import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.homo_utils import *
from transformer.epipolar_transformer import EpipolarTransformer
from networks.layers_op import convbn, convbnrelu, convbn_3d, convbnrelu_3d, convbntanh_3d


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = convbn(in_channels, out_channels, 3, 1, 1, 1)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


def depthlayer(logits, depth_values):
    prob_volume = torch.nn.functional.softmax(logits, dim=1)
    depth = torch.sum(prob_volume * depth_values, dim=1, keepdim=True)
    prob, _ = torch.max(prob_volume, dim=1, keepdim=True)

    return depth, prob


class DepthHybridDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_output_channels=1, use_skips=True,
                 ndepths=64, depth_max=10.0, IF_EST_transformer=True):
        super(DepthHybridDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.IF_EST_transformer = IF_EST_transformer
        self.upsample_mode = 'nearest'

        self.num_ch_enc = num_ch_enc  # [64, 64, 128, 256, 512]
        self.num_ch_dec = np.array([16, 32, ndepths, 128, 256])

        self.ndepths = ndepths
        self.depth_max = depth_max

        self.pixel_grid = None

        # decoder
        self.upconv_4_0 = ConvBlock(self.num_ch_enc[-1], self.num_ch_dec[4])
        self.upconv_4_1 = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])

        self.upconv_3_0 = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])
        self.upconv_3_1 = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])

        self.upconv_2_0 = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])
        self.upconv_2_1 = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.ndepths)

        self.upconv_1_0 = ConvBlock(self.num_ch_dec[2] + self.ndepths, self.num_ch_dec[1])
        self.upconv_1_1 = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
        self.dispconv_1 = nn.Conv2d(self.num_ch_dec[1], self.num_output_channels, 3, 1, 1, 1, bias=True)

        self.upconv_0_0 = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])
        self.upconv_0_1 = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])
        self.dispconv_0 = nn.Conv2d(self.num_ch_dec[0], self.num_output_channels, 3, 1, 1, 1, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        base_channels = 32
        if self.IF_EST_transformer:
            self.epipolar_transformer = EpipolarTransformer(base_channels // 2, base_channels // 2, 3)

        self.dres0 = nn.Sequential(convbnrelu_3d(base_channels, base_channels, 3, 1, 1),
                                   convbnrelu_3d(base_channels, base_channels, 3, 1, 1))

        self.dres1 = nn.Sequential(convbnrelu_3d(base_channels, base_channels, 3, 1, 1),
                                   convbnrelu_3d(base_channels, base_channels, 3, 1, 1))
        
        self.dres2 = nn.Sequential(convbnrelu_3d(base_channels + 1, base_channels + 1, 3, 1, 1))

        self.key_layer = nn.Sequential(convbnrelu_3d(base_channels + 1, base_channels // 2, 3, 1, 1))
        self.value_layer = nn.Sequential(convbntanh_3d(base_channels + 1, base_channels // 2, 3, 1, 1))

#         self.dres2 = nn.Sequential(convbnrelu_3d(base_channels, 1, 3, 1, 1))

#         self.fusion = nn.Sequential(convbnrelu_3d(2, base_channels // 2, 3, 1, 1),
#                                     convbnrelu_3d(base_channels // 2, base_channels // 2, 3, 1, 1),
#                                     convbnrelu_3d(base_channels // 2, base_channels, 3, 1, 1))

#         self.key_layer = nn.Sequential(convbnrelu_3d(base_channels, base_channels // 2, 3, 1, 1))
#         self.value_layer = nn.Sequential(convbntanh_3d(base_channels, base_channels // 2, 3, 1, 1))

        self.stereo_head0 = nn.Sequential(
            convbnrelu_3d(base_channels // 2, base_channels // 2, 3, 1, 1),
            nn.Conv3d(base_channels // 2, 1, kernel_size=1, padding=0, stride=1, bias=True)
        )

        self.stereo_head1 = nn.Sequential(
            convbnrelu_3d(base_channels // 2, base_channels // 2, 3, 1, 1),
            nn.Conv3d(base_channels // 2, 1, kernel_size=1, padding=0, stride=1, bias=True)
        )

    def scale_cam_intr(self, cam_intr, scale):
        cam_intr_new = cam_intr.clone()
        cam_intr_new[:, :2, :] *= scale

        return cam_intr_new

    def collapse_num(self, x):
        if len(x.shape) == 5:
            B, NUM, C, H, W = x.shape
            x = x.view(B * NUM, C, H, W)
        elif len(x.shape) == 6:
            B, NUM, C, D, H, W = x.shape
            x = x.view(B * NUM, C, D, H, W)
        return x

    def expand_num(self, x, NUM):
        if len(x.shape) == 4:
            B_NUM, C, H, W = x.shape
            x = x.view(-1, NUM, C, H, W)
        elif len(x.shape) == 5:
            B_NUM, C, D, H, W = x.shape
            x = x.view(-1, NUM, C, D, H, W)
        return x

    def forward_transformer(self, costvolumes, semantic_features, cam_poses, cam_intr,
                            depth_values, depth_min, depth_interval,
                            pre_costs=None, pre_cam_poses=None):
        """
        try to make it faster
        :param costvolumes: list of [N,C,D,H,W]
        :param cam_poses: list of [N,4,4]
        :param cam_intr: [N,3,3]
        :param depth_values: [N, ndepths, 1, 1]
        :return:
        """
        num = len(costvolumes)

        B, C, D, H, W = costvolumes[0].shape

        depth_values_lowres = depth_values.repeat(1, 1, H, W)
        depth_values_highres = depth_values.repeat(1, 1, 4 * H, 4 * W)

        outputs = {}

        if self.pixel_grid is None:
            self.pixel_grid = set_id_grid(H, W).to(costvolumes[0].dtype).to(costvolumes[0].device)  # [1, 3, H, W]
            self.pixel_grid = self.pixel_grid.view(1, 3, 1, H * W).repeat(B, 1, D, 1)  # [B, 3, D, H*W]

        # scale 4
        x = self.upconv_4_0(semantic_features[4])
        x = [upsample(x)]
        if self.use_skips:
            x += [semantic_features[3]]
        x = torch.cat(x, 1)
        x = self.upconv_4_1(x)

        # scale 3
        x = self.upconv_3_0(x)
        x = [upsample(x)]
        if self.use_skips:
            x += [semantic_features[2]]
        x = torch.cat(x, 1)
        x = self.upconv_3_1(x)

        # scale 2
        x = self.upconv_2_0(x)
        x = [upsample(x)]
        if self.use_skips:
            x += [semantic_features[1]]
        x = torch.cat(x, 1)
        semantic_vs = self.upconv_2_1(x)  # after relu, [B*num, C, H, W]

        # stack cost volumes together
        costvolumes = torch.stack(costvolumes, dim=1)
        costvolumes = self.collapse_num(costvolumes)
        # 3D matching guidance features
        matching_x = self.dres0(costvolumes)
        matching_x = self.dres1(matching_x)

#         x = self.dres2(matching_x)
#         x = self.fusion(torch.cat([semantic_vs.unsqueeze(1), x], dim=1))  # [B*num,32,D,H,W]
        x = torch.cat([semantic_vs.unsqueeze(1), matching_x], dim=1)  # [B*num,33,D,H,W]
        x = self.dres2(x)

        value = self.value_layer(x)
        key = self.key_layer(x)
        init_logits_ = self.stereo_head0(value).squeeze(1)  # [B*num,D,H,W]

        init_logits = F.interpolate(init_logits_, scale_factor=4)

        pred_depth_s3, pred_prob_s3 = depthlayer(init_logits, depth_values_highres)
        pred_depth_s3 = self.expand_num(pred_depth_s3, num)  # [B, num,1,H,W]
        pred_prob_s3 = self.expand_num(pred_prob_s3, num)
        for img_idx in range(num):
            outputs[("depth", img_idx, 3)] = pred_depth_s3[:, img_idx, :, :, :]
            outputs[("init_prob", img_idx)] = pred_prob_s3[:, img_idx, :, :, :]

        value = self.expand_num(value, num)
        key = self.expand_num(key, num)
        values = [value[:, img_idx, :, :, :, :] for img_idx in range(num)]
        keys = [key[:, img_idx, :, :, :, :] for img_idx in range(num)]
        detached_values = [value.detach() for value in values]
        detached_keys = [key.detach() for key in keys]

        ######################################################################
        # transformer
        if pre_costs is not None:
            cam_poses += pre_cam_poses
            values += pre_costs["values"]
            keys += pre_costs["keys"]
            pre_num = len(pre_cam_poses)
        else:
            pre_num = 0

        all_fused_logits = []
        for i in range(num):
            ref_cam_pose = cam_poses[i]
            warped_keys = []
            warped_values = []
            for j in range(num + pre_num):
                if i != j:
                    rel_pose = torch.matmul(cam_poses[j], torch.inverse(ref_cam_pose))

                    warped_key_ = warp_volume(keys[j], depth_values_lowres.view(B, 1, D, H * W),
                                              rel_pose, cam_intr,
                                              self.pixel_grid, depth_min, depth_interval)  # [B,C,D,H,W]

                    warped_value_ = warp_volume(values[j], depth_values_lowres.view(B, 1, D, H * W),
                                                rel_pose, cam_intr,
                                                self.pixel_grid, depth_min, depth_interval)  # [B,C,D,H,W]

                    warped_keys.append(warped_key_)
                    warped_values.append(warped_value_)

            fused_cost = self.epipolar_transformer(
                target_key=keys[i], warped_keys=warped_keys,
                target_value=values[i], warped_values=warped_values
            )

            values[i] = fused_cost
            detached_values[i] = fused_cost.detach()

            fused_logits_ = self.stereo_head1(fused_cost).squeeze(1)
            all_fused_logits.append(fused_logits_)

            fused_logits = F.interpolate(fused_logits_, scale_factor=4)
            outputs[("depth", i, 2)], outputs[("fused_prob", i)] = depthlayer(fused_logits, depth_values_highres)

        ######################################################################
        # depth refinement
        all_fused_logits = torch.stack(all_fused_logits, dim=1)  # [B, NUM, D, H, W]
        all_fused_logits = self.collapse_num(all_fused_logits)  # [B*NUM, D, H, W]

        # scale 1
        x = self.upconv_1_0(torch.cat([semantic_vs, self.relu(all_fused_logits)], dim=1))
        x = [upsample(x)]
        if self.use_skips:
            x += [semantic_features[0]]
        x = torch.cat(x, 1)
        x = self.upconv_1_1(x)

        pred_depth_s1 = F.interpolate(self.depth_max * self.sigmoid(self.dispconv_1(x)),
                                      scale_factor=2)
        pred_depth_s1 = self.expand_num(pred_depth_s1, num)  # [B, num,1,H,W]
        for img_idx in range(num):
            outputs[("depth", img_idx, 1)] = pred_depth_s1[:, img_idx, :, :, :]

        # scale 0
        x = self.upconv_0_0(x)
        x = [upsample(x)]
        x = torch.cat(x, 1)
        x = self.upconv_0_1(x)

        pred_depth_s0 = self.depth_max * self.sigmoid(self.dispconv_0(x))
        pred_depth_s0 = self.expand_num(pred_depth_s0, num)  # [B, num,1,H,W]
        for img_idx in range(num):
            outputs[("depth", img_idx, 0)] = pred_depth_s0[:, img_idx, :, :, :]

        return outputs, {"keys": detached_keys[-1:], "values": detached_values[-1:]}, cam_poses[-1:]

    def forward_notransformer(self, costvolumes, semantic_features, cam_poses, cam_intr,
                              depth_values, depth_min, depth_interval,
                              pre_costs=None, pre_cam_poses=None, if_trans_weight=True):
        """

        :param costvolumes: list of [N,C,D,H,W]
        :param cam_poses: list of [N,4,4]
        :param cam_intr: [N,3,3]
        :param depth_values: [N, ndepths, H, W]
        :return:
        """
        num = len(costvolumes)

        B, C, D, H, W = costvolumes[0].shape

        depth_values_lowres = depth_values.repeat(1, 1, H, W)
        depth_values_highres = depth_values.repeat(1, 1, 4 * H, 4 * W)

        outputs = {}

        if self.pixel_grid is None:
            self.pixel_grid = set_id_grid(H, W).to(costvolumes[0].dtype).to(costvolumes[0].device)  # [1, 3, H, W]
            self.pixel_grid = self.pixel_grid.view(1, 3, 1, H * W).repeat(B, 1, D, 1)  # [B, 3, D, H*W]

        # scale 4
        x = self.upconv_4_0(semantic_features[4])
        x = [upsample(x)]
        if self.use_skips:
            x += [semantic_features[3]]
        x = torch.cat(x, 1)
        x = self.upconv_4_1(x)

        # scale 3
        x = self.upconv_3_0(x)
        x = [upsample(x)]
        if self.use_skips:
            x += [semantic_features[2]]
        x = torch.cat(x, 1)
        x = self.upconv_3_1(x)

        # scale 2
        x = self.upconv_2_0(x)
        x = [upsample(x)]
        if self.use_skips:
            x += [semantic_features[1]]
        x = torch.cat(x, 1)
        semantic_vs = self.upconv_2_1(x)  # after relu, [B*num, C, H, W]

        # stack cost volumes together
        costvolumes = torch.stack(costvolumes, dim=1)
        costvolumes = self.collapse_num(costvolumes)
        # 3D matching guidance features
        matching_x = self.dres0(costvolumes)
        matching_x = self.dres1(matching_x)

#         x = self.dres2(matching_x)
#         # fuse matching feature and semantic feature
#         x = self.fusion(torch.cat([semantic_vs.unsqueeze(1), x], dim=1))  # [B*num,32,D,H,W]
        x = torch.cat([semantic_vs.unsqueeze(1), matching_x], dim=1)  # [B*num,33,D,H,W]
        x = self.dres2(x)

        value = self.value_layer(x)
        key = self.key_layer(x)
        init_logits_ = self.stereo_head0(value).squeeze(1)  # [B*num,D,H,W]

        init_logits = F.interpolate(init_logits_, scale_factor=4)

        pred_depth_s3, pred_prob_s3 = depthlayer(init_logits, depth_values_highres)
        pred_depth_s3 = self.expand_num(pred_depth_s3, num)  # [B, num,1,H,W]
        pred_prob_s3 = self.expand_num(pred_prob_s3, num)
        for img_idx in range(num):
            outputs[("depth", img_idx, 3)] = pred_depth_s3[:, img_idx, :, :, :]
            outputs[("init_prob", img_idx)] = pred_prob_s3[:, img_idx, :, :, :]

        value_expand = self.expand_num(value, num)
        key_expand = self.expand_num(key, num)
        values = [value_expand[:, img_idx, :, :, :, :] for img_idx in range(num)]
        keys = [key_expand[:, img_idx, :, :, :, :] for img_idx in range(num)]
        detached_values = [value.detach() for value in values]
        detached_keys = [key.detach() for key in keys]

        ######################################################################

        all_fused_logits = self.stereo_head1(value).squeeze(1)

        fused_logits = F.interpolate(all_fused_logits, scale_factor=4)

        pred_depth_s2, pred_prob_s2 = depthlayer(fused_logits, depth_values_highres)
        pred_depth_s2 = self.expand_num(pred_depth_s2, num)  # [B, num,1,H,W]
        pred_prob_s2 = self.expand_num(pred_prob_s2, num)
        for img_idx in range(num):
            outputs[("depth", img_idx, 2)] = pred_depth_s2[:, img_idx, :, :, :]
            outputs[("fused_prob", img_idx)] = pred_prob_s2[:, img_idx, :, :, :]

        ######################################################################
        # depth refinement

        # scale 1
        x = self.upconv_1_0(torch.cat([semantic_vs, self.relu(all_fused_logits)], dim=1))
        x = [upsample(x)]
        if self.use_skips:
            x += [semantic_features[0]]
        x = torch.cat(x, 1)
        x = self.upconv_1_1(x)

        pred_depth_s1 = F.interpolate(self.depth_max * self.sigmoid(self.dispconv_1(x)),
                                      scale_factor=2)
        pred_depth_s1 = self.expand_num(pred_depth_s1, num)  # [B, num,1,H,W]
        for img_idx in range(num):
            outputs[("depth", img_idx, 1)] = pred_depth_s1[:, img_idx, :, :, :]

        # scale 0
        x = self.upconv_0_0(x)
        x = [upsample(x)]
        x = torch.cat(x, 1)
        x = self.upconv_0_1(x)
        # outputs[("depth", img_idx, 0)] = self.depth_max * self.sigmoid(self.dispconv_0(x))

        pred_depth_s0 = self.depth_max * self.sigmoid(self.dispconv_0(x))
        pred_depth_s0 = self.expand_num(pred_depth_s0, num)  # [B, num,1,H,W]
        for img_idx in range(num):
            outputs[("depth", img_idx, 0)] = pred_depth_s0[:, img_idx, :, :, :]

        return outputs, {"keys": detached_keys[-1:], "values": detached_values[-1:]}, cam_poses[-1:]

    def forward(self, costvolumes, semantic_features, cam_poses, cam_intr,
                depth_values, depth_min, depth_interval,
                pre_costs=None, pre_cam_poses=None, mode="train"):

        flag = self.IF_EST_transformer & (pre_costs is not None or mode == "train")

        if flag:
            return self.forward_transformer(costvolumes, semantic_features, cam_poses, cam_intr,
                                            depth_values, depth_min, depth_interval,
                                            pre_costs, pre_cam_poses)
        else:
            return self.forward_notransformer(costvolumes, semantic_features, cam_poses, cam_intr,
                                              depth_values, depth_min, depth_interval,
                                              pre_costs, pre_cam_poses)
