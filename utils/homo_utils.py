from __future__ import division
import torch
from torch.autograd import Variable
import numpy as np


def set_id_grid(h, w):
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(1, h, w)).to(torch.float32)  # [1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(1, h, w)).to(torch.float32)  # [1, H, W]
    ones = Variable(torch.ones(1, h, w)).to(torch.float32)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

    return pixel_coords


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert (all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected),
                                                                              list(input.size()))


def cam2cam(cam_1_coords, extrinsic):
    '''Transform 3D coordinates from cam 1 to cam 2
    :param cam_1_coords: [batch, 4, height, width]
    :param extrinsic: [batch, 4, 4]  extrinsic (the position of the origin of the world coordinate system expressed
    in coordinates of the camera-centered coordinate system. world coords w.r.t. camera coords)
    :return: cam coordinates in cam 2 [batch, 4, height, width]
    '''
    batch, _, height, width = cam_1_coords.shape
    cam_1_coords_flat = cam_1_coords.view(batch, 4, -1)
    cam_2_coords = torch.bmm(extrinsic, cam_1_coords_flat)
    cam_2_coords = cam_2_coords.view(batch, 4, height, width)
    return cam_2_coords


def pixel2cam(depth, intrinsics, pixel_coords, is_homogeneous=True):
    """Transform coordinates in the pixel frame to the camera frame. Camera model: [X,Y,Z]^T = D * K^-1 * [u,v,1]^T
    Args:
        depth: depth maps -- [B, 1, H, W]
        intrinsics: intrinsics matrix for each element of batch -- [B, 3, 3]
        pixel_coords: the original pixel coordinates [1, 3, h, w]
        is_homogeneous: return in homogeneous coordinates
    Returns:
        array of (x,y,z, (1)) cam coordinates -- [B, 3 (4 if homogeneous), H, W]
    """
    b, _, h, w = depth.size()
    intrinsics_inv = torch.inverse(intrinsics)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w) \
        .contiguous().view(b, 3, -1).to(depth.device)  # [B, 3, H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w) * depth.repeat(1, 3, 1, 1)

    tt = depth[:, 0, 62, :]

    if is_homogeneous:
        ones = torch.ones((b, 1, h, w)).to(depth.dtype).to(depth.device)
        cam_coords = torch.cat([cam_coords, ones], dim=1)  # [b, 4, h, w]

    return cam_coords


# def cam2pixel(cam_coords, intrinsics):
#     '''Transforms coordinates in a camera frame to the pixel frame.
#     :param cam_coords: [batch, 4, height, width]
#     :param intrinsics: [batch, 3, 3]
#     :return: Pixel coordinates projected from the camera frame [batch, height, width, 2]
#     '''
#     b, _, h, w = cam_coords.size()
#     zeros = torch.zeros((b, 3, 1)).to(intrinsics.dtype).to(intrinsics.device)
#     intrinsics_3x4 = torch.cat([intrinsics, zeros], dim=2)
#     filter = torch.tensor([0, 0, 0, 1]).view(1, 1, 4).to(intrinsics.dtype).to(intrinsics.device)
#     filter = filter.repeat(b, 1, 1)
#     intrinsics_4x4 = torch.cat([intrinsics_3x4, filter], dim=1)  # [b, 4, 4]
#
#     world_coords = cam_coords.view(b, 4, -1)  # [b,4,h*w]
#     unnorm_pixel_coords = torch.bmm(intrinsics_4x4, world_coords)  # [b,4,h*w]
#     x = unnorm_pixel_coords[:, 0, :]
#     y = unnorm_pixel_coords[:, 1, :]
#     z = unnorm_pixel_coords[:, 2, :]
#     x_norm = x / (z + 1e-10)
#     y_norm = y / (z + 1e-10)
#     pixel_coords = torch.stack([x_norm, y_norm], dim=2)  # [b,h*w, 2]
#     return pixel_coords.view(b, h, w, 2)

def cam2pixel(cam_coords, intrinsics):
    '''Transforms coordinates in a camera frame to the pixel frame.
    :param cam_coords: [batch, 4, height, width]
    :param intrinsics: [batch, 3, 3]
    :return: Pixel coordinates projected from the camera frame [batch, height, width, 2]
    '''
    b, _, h, w = cam_coords.size()

    world_coords = cam_coords[:, :3, :, :].view(b, 3, -1)  # [b,3,h*w]
    unnorm_pixel_coords = torch.bmm(intrinsics, world_coords)  # [b,3,h*w]
    x = unnorm_pixel_coords[:, 0, :]
    y = unnorm_pixel_coords[:, 1, :]
    z = unnorm_pixel_coords[:, 2, :]
    x_norm = x / (z + 1e-10)
    y_norm = y / (z + 1e-10)
    pixel_coords = torch.stack([x_norm, y_norm], dim=2)  # [b,h*w, 2]
    return pixel_coords.view(b, h, w, 2)


def cam2pixel_depth(cam_coords, intrinsics):
    '''Transforms coordinates in a camera frame to the pixel frame.
    :param cam_coords: [batch, 4, height, width]
    :param intrinsics: [batch, 3, 3]
    :return: Pixel coordinates projected from the camera frame [batch, height, width, 2]
    '''
    b, _, h, w = cam_coords.size()

    world_coords = cam_coords[:, :3, :, :].view(b, 3, -1)  # [b,3,h*w]
    unnorm_pixel_coords = torch.bmm(intrinsics, world_coords)  # [b,3,h*w]
    x = unnorm_pixel_coords[:, 0, :]
    y = unnorm_pixel_coords[:, 1, :]
    z = unnorm_pixel_coords[:, 2, :]
    x_norm = x / (z + 1e-10)
    y_norm = y / (z + 1e-10)

    X_norm_ = 2 * x_norm / (
            w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm_ = 2 * y_norm / (h - 1) - 1  # Idem [B, H, W]

    X_mask_ = ((X_norm_ > 1) + (X_norm_ < -1)).detach()
    Y_mask_ = ((Y_norm_ > 1) + (Y_norm_ < -1)).detach()

    invalid_mask = X_mask_ | Y_mask_  # [ b, h, w]
    valid_mask = (~invalid_mask).view(b, 1, h, w)

    pixel_coords = torch.stack([x_norm, y_norm, z], dim=2)  # [b,h*w, 3]
    return pixel_coords.view(b, h, w, 3), valid_mask


def normalize_pixel_coords(pixel_coords, padding_mode='zeros', image_height=None, image_width=None):
    """
    normalize pixel coordintes into arrays (-1, 1) for torch.nn.functional.grid_sample
    :param pixel_coords: [b, h, w, 2]
    :param padding_mode:
    :return: normalized to (-1, 1) [b, h, w, 2]
    """
    if image_height is None:
        b, h, w, _ = pixel_coords.shape
        image_height = h
        image_width = w
    else:
        b, h, w, _ = pixel_coords.shape

    X = pixel_coords[:, :, :, 0]
    Y = pixel_coords[:, :, :, 1]

    X_norm = 2 * X / (
            image_width - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (image_height - 1) - 1  # Idem [B, H, W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

        invalid_mask = X_mask | Y_mask  # [ b, h, w]
        valid_mask = (~invalid_mask).view(b, 1, h, w)

    pixel_coords_norm = torch.stack([X_norm, Y_norm], dim=3)  # [B, H, W, 2]
    return pixel_coords_norm, valid_mask


def normalize_pixel_coords_volume(pixel_coords, depth_min, depth_interval, padding_mode='zeros',
                                  image_height=None, image_width=None, ndepths=None, disp_min=None, disp_interval=None):
    """
    normalize pixel coordintes into arrays (-1, 1) for torch.nn.functional.grid_sample
    :param pixel_coords: [b, ndepths, H*W, 3]
    :param padding_mode:
    :return: normalized to (-1, 1) [b, ndepths, h, w, 3]
    """
    batch_size = pixel_coords.shape[0]
    X = pixel_coords[:, :, :, 0]
    Y = pixel_coords[:, :, :, 1]
    Z = pixel_coords[:, :, :, 2]

    X_norm = 2 * X / (
            image_width - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (image_height - 1) - 1  # Idem [B, H, W]

    if disp_min is None:
        Z_norm = 2 * ((Z - depth_min) / depth_interval) / (ndepths - 1) - 1.
    else:
        # the cost volume use disparity planes not depth planes
        Z_norm = 2 * ((1. / (Z + 1e-10) - disp_min) / disp_interval) / (ndepths - 1) - 1.
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2
        Z_mask = ((Z_norm > 1) + (Z_norm < -1)).detach()
        Z_norm[Z_mask] = 2

        invalid_mask = X_mask | Y_mask | Z_mask  # [ b, ndepths, h*w]
        valid_mask = (~invalid_mask).view(batch_size, 1, ndepths, image_height, image_width)

    pixel_coords_norm = torch.stack([X_norm, Y_norm, Z_norm], dim=3)  # [B, ndepths, H*W, 3]
    pixel_coords_norm = pixel_coords_norm.view(batch_size, ndepths, image_height, image_width, 3)
    return pixel_coords_norm, valid_mask


def inverse_warp(feat, depth, pose, intrinsics, pixel_coords, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        feat: the source feature (where to sample pixels) -- [B, CH, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        pose: Pose matrix (camera 2 world) from source to target -- [B, 4, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        pixel_coords:  [b, 3, h, w] original homogeneous pixel coordinates of target image
    Returns:
        Source image warped to the target image plane
    """
    # check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B44')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, feat_height, feat_width = feat.size()

    tgt_cam_coords = pixel2cam(depth, intrinsics, pixel_coords, is_homogeneous=True)  # [b, 4, h, w]

    src_cam_coords = cam2cam(tgt_cam_coords, torch.inverse(pose))  # change pose to extrinsic matrix (world 2 camera)

    src_pixel_coords = cam2pixel(src_cam_coords, intrinsics)

    src_pixel_coords_norm, valid_mask = normalize_pixel_coords(src_pixel_coords)  # [B,H,W,2]

    projected_feat = torch.nn.functional.grid_sample(feat, src_pixel_coords_norm, padding_mode=padding_mode)

    return projected_feat


def warp_volume(feat_volume, depth, pose, cam_intr, pixel_coords,
                depth_min, depth_interval, padding_mode='zeros', padding_value=0., disp_min=None, disp_interval=None,
                inter_mode='bilinear'):
    """

    :param feat_volume: [N,C,D,H,W]
    :param depth: [N,1,D,H*W]
    :param pose: [N,4,4]
    :param cam_intr: [N,3,3]
    :param pixel_coords: [N,3,D,H*W]
    :param padding_mode:
    :return:
    """
    N, C, D, H, W = feat_volume.shape

    tgt_cam_coords = pixel2cam(depth, cam_intr, pixel_coords, is_homogeneous=True)  # [b, 4, ndepths, h*w]

    # [b, 4, ndepths, h*w]
    src_cam_coords = cam2cam(tgt_cam_coords, torch.inverse(pose))  # change pose to extrinsic matrix (world 2 camera)

    src_pixel_coords, _ = cam2pixel_depth(src_cam_coords, cam_intr)

    src_pixel_coords_norm, valid_mask = normalize_pixel_coords_volume(src_pixel_coords,
                                                                      depth_min,
                                                                      depth_interval,
                                                                      image_height=H,
                                                                      image_width=W,
                                                                      ndepths=D,
                                                                      disp_min=disp_min,
                                                                      disp_interval=disp_interval)  # [B,ndepths, H,W,3]

    if padding_mode == 'border':
        feat_volume_th_ = _set_vol_border(feat_volume, padding_value)
        projected_volume = torch.nn.functional.grid_sample(feat_volume_th_, src_pixel_coords_norm, inter_mode,
                                                           padding_mode=padding_mode)
    else:
        projected_volume = torch.nn.functional.grid_sample(feat_volume, src_pixel_coords_norm, inter_mode,
                                                           padding_mode=padding_mode)

    return projected_volume


def warp_depth(depth, pose, cam_intr, pixel_coords):
    """
    warp the depth map of reference image into src view, get the depth map of src view
    :param depth: [N,1,H,W] the depth map of reference image
    :param pose: [N,4,4] relative camera pose (Src_pose @ inv(ref_pose)), not camera extrinsic
    :param cam_intr: [N,3,3]
    :param pixel_coords: [N,3,H,W]
    :param padding_mode:
    :return:
    """
    N, C, H, W = depth.shape
    tgt_cam_coords = pixel2cam(depth, cam_intr, pixel_coords, is_homogeneous=True)  # [b, 4, h,w]

    # [b, 4, ndepths, h*w]
    src_cam_coords = cam2cam(tgt_cam_coords, torch.inverse(pose))  # change pose to extrinsic matrix (world 2 camera)

    src_pixel_coords, mask = cam2pixel_depth(src_cam_coords, cam_intr)

    warped_depth = src_pixel_coords[:, :, :, 2]

    return warped_depth.unsqueeze(1), mask


def _set_vol_border(vol, border_val):
    '''
    inputs:
    vol - a torch tensor in 3D: N x C x D x H x W
    border_val - a float, the border value
    '''
    vol_ = vol + 0.
    vol_[:, :, 0, :, :] = border_val
    vol_[:, :, :, 0, :] = border_val
    vol_[:, :, :, :, 0] = border_val
    vol_[:, :, -1, :, :] = border_val
    vol_[:, :, :, -1, :] = border_val
    vol_[:, :, :, :, -1] = border_val

    return vol_


def skew(phi):
    '''
    :param phi: [batch, 3]
    :return: Phi : [batch, 3, 3 ]
    '''
    b, _ = phi.shape
    phi_0 = phi[:, 0:1]
    phi_1 = phi[:, 1:2]
    phi_2 = phi[:, 2:3]
    zero = torch.zeros_like(phi_0).to(phi.dtype).to(phi.device)
    Phi = torch.cat([zero, -phi_2, phi_1, phi_2, zero, -phi_0, -phi_1, phi_0, zero], dim=-1)
    Phi = Phi.view(b, 3, 3)
    return Phi


def expMap(ksai):
    '''exponetial mapping.
    :param ksai: [Batch, 6]
    :return: SE3: [Batch, 4, 4]
    '''
    b, _ = ksai.shape
    omega = ksai[:, :3]  # rotation
    upsilon = ksai[:, 3:]  # translation

    theta = torch.norm(omega, dim=-1, keepdim=True)  # [b, 1]
    theta = theta.view(b, 1, 1).repeat(1, 3, 3)

    Omega = skew(omega)  # [b,3,3]
    Omega2 = torch.bmm(Omega, Omega)

    identities = torch.eye(3).view(1, 3, 3).repeat(b, 1, 1).to(ksai.dtype).to(ksai.device)  # [b, 3, 3]

    R = identities + \
        torch.sin(theta) * Omega / theta + \
        (1 - torch.cos(theta)) * Omega2 / (theta * theta)
    V = identities + \
        (1 - torch.cos(theta)) * Omega / (theta * theta) + \
        (theta - torch.sin(theta)) * Omega2 / (theta * theta * theta)
    t = torch.bmm(V, upsilon.view(b, 3, 1))  # [b, 3, 1]

    T34 = torch.cat([R, t], dim=-1)
    brow = torch.tensor([0., 0., 0., 1.]).view(1, 1, 4).repeat(b, 1, 1).to(T34.dtype).to(T34.device)  # [b, 1, 4]
    T44 = torch.cat([T34, brow], dim=1)
    return T44


def logMap(SE3):
    '''logarithmic mapping
    :param SE3: [B, 4, 4]
    :return: ksai [B, 6], trn after rot
    '''

    def deltaR(R):
        '''
        :param R:  [B, 3, 3]
        :return:
        '''
        v_0 = R[:, 2, 1] - R[:, 1, 2]
        v_1 = R[:, 0, 2] - R[:, 2, 0]
        v_2 = R[:, 1, 0] - R[:, 0, 1]
        v = torch.stack([v_0, v_1, v_2], dim=-1)  # [b, 3]
        return v

    b, _, _ = SE3.shape
    _R = SE3[:, :3, :3]
    _t = SE3[:, :3, 3:]
    d = 0.5 * (_R[:, 0, 0] + _R[:, 1, 1] + _R[:, 2, 2] - 1)  # [b]
    d = d.view(b, 1)
    dR = deltaR(_R)  # [b, 3]
    theta = torch.acos(d)  # [b, 1]
    omega = theta * dR / (2 * torch.sqrt(1 - d * d))
    Omega = skew(omega)
    identities = torch.eye(3).view(1, 3, 3).repeat(b, 1, 1).to(SE3.dtype).to(SE3.device)  # [b, 3, 3]
    V_inv = identities - 0.5 * Omega + \
            (1 - theta / (2 * torch.tan(theta / 2))) * torch.bmm(Omega, Omega) / (theta * theta)
    upsilon = torch.bmm(V_inv, _t)
    upsilon = upsilon.view(b, -1)
    ksai = torch.concat([omega, upsilon], dim=-1)
    return ksai


def mat2euler_np(rot_M):
    '''Conver rotation matrix to euler angle X->Y->Z, numpy version
    :param rot_M:
    :return:
    '''
    r11 = rot_M[0][0]
    r12 = rot_M[0][1]
    r13 = rot_M[0][2]

    r21 = rot_M[1][0]
    r22 = rot_M[1][1]
    r23 = rot_M[1][2]

    r31 = rot_M[2][0]
    r32 = rot_M[2][1]
    r33 = rot_M[2][2]

    rx = np.arctan2(-r23, r33)
    cy = np.sqrt(r11 * r11 + r12 * r12)
    ry = np.arctan2(r13, cy)
    rz = np.arctan2(-r12, r11)

    # return  tf.stack( [ rx, ry, rz ] )
    return np.stack([rx, ry, rz])


def quat2mat(q):
    '''
    :param q:
    :return:
    '''

    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array(
        [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :] + 1e-8)  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1

        # mask out of range values, otherwise the warped volume will have Nan values
        X_mask = ((proj_x_normalized > 1) + (proj_x_normalized < -1)).detach()
        proj_x_normalized[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((proj_y_normalized > 1) + (proj_y_normalized < -1)).detach()
        proj_y_normalized[Y_mask] = 2

        invalid_mask = X_mask | Y_mask  # [ b, h, w]
        valid_mask = (~invalid_mask).view(batch, num_depth, height, width)

        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = torch.nn.functional.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2),
                                                     mode='bilinear',
                                                     padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea
