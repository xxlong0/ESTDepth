import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from data.general_eval_seq import SevenScenesSeq
from metric import *

from utils.misc_utils import *
from utils.utils import *
from hybrid_models.model_hybrid import DepthNetHybrid
import os
import pdb
import time

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of Cascade Cost Volume MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--device', default='cuda', help='select model')

parser.add_argument('--datapath', help='train datapath')
parser.add_argument('--evalpath', help='the evaluation output dir')
parser.add_argument('--trainlist', help='train list', default="./data/scannet_split/train_split.txt")
parser.add_argument('--testlist', help='test list', default="./data/scannet_split/test_split.txt")

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2",
                    help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=4e-4, help='weight decay')

parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--seq_len', type=int, default=7, help='the length of each img seq')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--pin_m', action='store_true', help='data loader pin memory')
parser.add_argument("--local_rank", type=int, default=0)

parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex')
parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str, default="O0")
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

parser.add_argument('--depth_min', type=float, default=0.01, help='the minimum depth value')
parser.add_argument('--depth_max', type=float, default=10.0, help='the minimum depth value')
parser.add_argument('--ndepths', type=int, default=64, help='the minimum depth value')

parser.add_argument('--lwindow', type=int, default=3, help='length of local window')

parser.add_argument('--resnet', type=int, default=50, help='resnet 18 or 50 as backbone')

parser.add_argument('--IF_EST_transformer', type=str, default="True", help='IF_EST_transformer')

parser.add_argument('--save_init_depth', type=str, default="True")
parser.add_argument('--save_refined_depth', type=str, default="True")
parser.add_argument('--save_init_prob', type=str, default="True")
parser.add_argument('--save_refined_prob', type=str, default="True")

parser.add_argument('--eval_dataset', type=str, default="scannet", help='scannet or 7scenes')

parser.add_argument('--memory_size', type=int, default=2, help='the number of estimation stored in memory')

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1


def lw2batch(local_sample_window, local_pre_costs_window,
             local_pre_cam_poses_window):
    sample = {
        'imgs': [],
        'img_raws': [],
        'dmaps': [],
        'dmasks': [],
        'cam_poses': [],
    }

    for i in range(len(local_sample_window)):
        sample['imgs'].append(local_sample_window[i]['img'])
        sample['img_raws'].append(local_sample_window[i]['img_raw'])
        sample['dmaps'].append(local_sample_window[i]['dmap'])
        sample['dmasks'].append(local_sample_window[i]['dmask'])
        sample['cam_poses'].append(local_sample_window[i]['cam_pose'])

    for key, value in sample.items():
        sample[key] = torch.stack(sample[key], dim=1)

    sample['img_paths'] = []
    for i in range(len(local_sample_window)):
        sample['img_paths'].append(local_sample_window[i]['img_path'])

    sample['cam_intr'] = local_sample_window[i]['cam_intr']

    if local_pre_cam_poses_window:

        keys = list(local_pre_costs_window[0].keys())
        pre_costs = {
            keys[0]: [],
            keys[1]: []
        }

        pre_cam_poses = []

        for i in range(len(local_pre_costs_window)):
            pre_costs[keys[0]].append(local_pre_costs_window[i][keys[0]][0])
            pre_costs[keys[1]].append(local_pre_costs_window[i][keys[1]][0])
            pre_cam_poses.append(local_pre_cam_poses_window[i][0])

    else:
        return sample, None, None

    return sample, pre_costs, pre_cam_poses


@make_nograd_func
def test_scannet_seq(model, args):
    MEMORY_WINDOW = args.memory_size
    INTERVAL = 10
    model.eval()
    # data loader
    # dataset, dataloader
    times = []

    test_dataset = SevenScenesSeq(args.datapath, frame_interval=INTERVAL, eval_dataset=args.eval_dataset)

    if args.IF_EST_transformer == "True":
        evaluation_dir = args.evalpath + "_M" + str(MEMORY_WINDOW)
    else:
        evaluation_dir = args.evalpath

    # main loop
    scenes_file = open(args.testlist, 'r')

    for scene in scenes_file.readlines():
        scene = scene.rstrip()
        print(scene)
        rgb_dir = os.path.join(evaluation_dir, scene, 'rgb')
        init_depth_dir = os.path.join(evaluation_dir, scene, 'init_depth')
        init_prob_dir = os.path.join(evaluation_dir, scene, 'init_prob')

        refined_depth_dir = os.path.join(evaluation_dir, scene, 'refined_depth')
        refined_prob_dir = os.path.join(evaluation_dir, scene, 'refined_prob')

        dirs = [rgb_dir, init_depth_dir, init_prob_dir, refined_depth_dir, refined_prob_dir]

        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

        test_dataset.reset(scene)

        pre_costs = None
        pre_cam_poses = None
        for index, sample in enumerate(test_dataset):
            if index == 0:
                print("new squence starts!")
                local_sample_window = []
                local_pre_costs_window = []
                local_pre_cam_poses_window = []

            local_sample_window.append(sample)

            if len(local_sample_window) == args.lwindow:

                sample_window, pre_costs, pre_cam_poses = lw2batch(local_sample_window, local_pre_costs_window,
                                                                   local_pre_cam_poses_window)

                sample_cuda = tocuda(sample_window)

                # print(sample["img_path"])
                start_time = time.time()
                outputs, pre_costs, pre_cam_poses = model(sample_cuda["imgs"], sample_cuda["cam_poses"],
                                                          sample_cuda["cam_intr"],
                                                          sample_cuda, pre_costs, pre_cam_poses,
                                                          mode='val')
                times.append(time.time() - start_time)
                local_pre_costs_window.append(pre_costs)
                local_pre_cam_poses_window.append(pre_cam_poses)
            else:
                continue

            local_sample_window.pop(0)
            if len(local_pre_costs_window) > MEMORY_WINDOW:
                local_pre_cam_poses_window.pop(0)
                local_pre_costs_window.pop(0)

            intrinsic = sample_cuda["cam_intr"]

            dmaps_gt = sample_cuda['dmaps']
            dmasks_gt = sample_cuda['dmasks']

            with torch.no_grad():
                img_i = 0
                rgb_basename = os.path.basename(sample_window["img_paths"][args.lwindow // 2])
                _, img_ext = os.path.splitext(rgb_basename)
                rgb_filepath = os.path.join(rgb_dir, rgb_basename)
                # cv2.imwrite(rgb_filepath,
                #             cv2.cvtColor(sample["img_raws"][:, img_i + 1, :, :, :].squeeze().numpy(),
                #                          cv2.COLOR_RGB2BGR))

                if args.save_init_depth == "True":
                    init_depth = np.float16(outputs[("depth", img_i, 2)].squeeze(1).cpu().numpy())
                    init_depth_filepath = os.path.join(init_depth_dir,
                                                       rgb_basename.replace(img_ext, ".npy"))
                    np.save(init_depth_filepath, init_depth)

                    init_depth_color = colorize_depth(outputs[("depth", img_i, 2)].squeeze(1),
                                                      max_depth=5.0).permute(0, 2, 3, 1).squeeze().cpu().numpy()
                    init_depth_color_filepath = os.path.join(init_depth_dir,
                                                             rgb_basename.replace(img_ext,
                                                                                  ".jpg"))
                    cv2.imwrite(init_depth_color_filepath,
                                cv2.cvtColor(np.uint8(init_depth_color), cv2.COLOR_RGB2BGR))
                if args.save_init_prob == "True":
                    init_prob = colorize_probmap(outputs[("init_prob", img_i)].squeeze(1)).permute(0, 2, 3,
                                                                                                   1).squeeze().cpu().numpy()
                    init_prob_filepath = os.path.join(init_prob_dir,
                                                      rgb_basename.replace(img_ext, ".jpg"))
                    cv2.imwrite(init_prob_filepath, cv2.cvtColor(np.uint8(init_prob), cv2.COLOR_RGB2BGR))
                    np.save(init_prob_filepath.replace('jpg', 'npy'),
                            np.float16(outputs[("init_prob", img_i)].squeeze().cpu().numpy()))

                if args.save_refined_depth == "True":
                    refined_depth = np.float16(outputs[("depth", img_i, 0)].squeeze(1).cpu().numpy())
                    refined_depth_filepath = os.path.join(refined_depth_dir,
                                                          rgb_basename.replace(img_ext,
                                                                               ".npy"))
                    np.save(refined_depth_filepath, refined_depth)

                    refined_depth_color = colorize_depth(outputs[("depth", img_i, 0)].squeeze(1),
                                                         max_depth=5.0).permute(0, 2, 3,
                                                                                1).squeeze().cpu().numpy()
                    refined_depth_color_filepath = os.path.join(refined_depth_dir,
                                                                rgb_basename.replace(img_ext,
                                                                                     ".jpg"))
                    cv2.imwrite(refined_depth_color_filepath,
                                cv2.cvtColor(np.uint8(refined_depth_color), cv2.COLOR_RGB2BGR))

                if args.save_refined_prob == "True":
                    refined_prob = colorize_probmap(outputs[("fused_prob", img_i)].squeeze(1)).permute(0, 2,
                                                                                                       3,
                                                                                                       1).squeeze().cpu().numpy()
                    refined_prob_filepath = os.path.join(refined_prob_dir,
                                                         rgb_basename.replace(img_ext,
                                                                              ".jpg"))
                    cv2.imwrite(refined_prob_filepath,
                                cv2.cvtColor(np.uint8(refined_prob), cv2.COLOR_RGB2BGR))

                    np.save(refined_prob_filepath.replace('jpg', 'npy'),
                            np.float16(outputs[("fused_prob", img_i)].squeeze().cpu().numpy()))
    print("inference time", np.mean(times))


@make_nograd_func
def test_sevenscenes_seq(model, args):
    MEMORY_WINDOW = args.memory_size
    INTERVAL = 10

    model.eval()
    # data loader
    # dataset, dataloader

    test_dataset = SevenScenesSeq(args.datapath, frame_interval=INTERVAL, eval_dataset=args.eval_dataset)
    if args.IF_EST_transformer == "True":
        evaluation_dir = args.evalpath + "_M" + str(MEMORY_WINDOW)
    else:
        evaluation_dir = args.evalpath

    for (scene, seq) in test_dataset.test_seqs_list:
        print(scene, seq)

        rgb_dir = os.path.join(evaluation_dir, scene, seq, 'rgb')
        init_depth_dir = os.path.join(evaluation_dir, scene, seq, 'init_depth')
        init_prob_dir = os.path.join(evaluation_dir, scene, seq, 'init_prob')

        refined_depth_dir = os.path.join(evaluation_dir, scene, seq, 'refined_depth')
        refined_prob_dir = os.path.join(evaluation_dir, scene, seq, 'refined_prob')

        dirs = [rgb_dir, init_depth_dir, init_prob_dir, refined_depth_dir, refined_prob_dir]

        if os.path.exists(refined_depth_dir):
            continue

        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

        test_dataset.reset(scene, seq)

        pre_costs = None
        pre_cam_poses = None
        for index, sample in enumerate(test_dataset):
            if index == 0:
                print("new squence starts!")
                local_sample_window = []
                local_pre_costs_window = []
                local_pre_cam_poses_window = []

            local_sample_window.append(sample)

            if len(local_sample_window) == args.lwindow:

                sample_window, pre_costs, pre_cam_poses = lw2batch(local_sample_window, local_pre_costs_window,
                                                                   local_pre_cam_poses_window)

                sample_cuda = tocuda(sample_window)

                # print(sample["img_path"])

                outputs, pre_costs, pre_cam_poses = model(sample_cuda["imgs"], sample_cuda["cam_poses"],
                                                          sample_cuda["cam_intr"],
                                                          sample_cuda,
                                                          pre_costs,
                                                          pre_cam_poses,
                                                          # None, None,
                                                          mode='val')

                local_pre_costs_window.append(pre_costs)
                local_pre_cam_poses_window.append(pre_cam_poses)
            else:
                continue

            local_sample_window.pop(0)
            if len(local_pre_costs_window) > MEMORY_WINDOW:
                local_pre_cam_poses_window.pop(0)
                local_pre_costs_window.pop(0)

            intrinsic = sample_cuda["cam_intr"]

            dmaps_gt = sample_cuda['dmaps']
            dmasks_gt = sample_cuda['dmasks']

            with torch.no_grad():
                img_i = 0
                rgb_basename = os.path.basename(sample_window["img_paths"][args.lwindow // 2])
                _, img_ext = os.path.splitext(rgb_basename)
                rgb_filepath = os.path.join(rgb_dir, rgb_basename)
                # cv2.imwrite(rgb_filepath,
                #             cv2.cvtColor(sample["img_raws"][:, img_i + 1, :, :, :].squeeze().numpy(),
                #                          cv2.COLOR_RGB2BGR))

                if args.save_init_depth == "True":
                    init_depth = np.float16(outputs[("depth", img_i, 2)].squeeze(1).cpu().numpy())
                    init_depth_filepath = os.path.join(init_depth_dir,
                                                       rgb_basename.replace(img_ext, ".npy"))
                    np.save(init_depth_filepath, init_depth)

                    init_depth_color = colorize_depth(outputs[("depth", img_i, 2)].squeeze(1),
                                                      max_depth=5.0).permute(0, 2, 3, 1).squeeze().cpu().numpy()
                    init_depth_color_filepath = os.path.join(init_depth_dir,
                                                             rgb_basename.replace(img_ext,
                                                                                  ".jpg"))
                    cv2.imwrite(init_depth_color_filepath,
                                cv2.cvtColor(np.uint8(init_depth_color), cv2.COLOR_RGB2BGR))
                if args.save_init_prob == "True":
                    init_prob = colorize_probmap(outputs[("init_prob", img_i)].squeeze(1)).permute(0, 2, 3,
                                                                                                   1).squeeze().cpu().numpy()
                    init_prob_filepath = os.path.join(init_prob_dir,
                                                      rgb_basename.replace("color" + img_ext, "init_prob.jpg"))
                    cv2.imwrite(init_prob_filepath, cv2.cvtColor(np.uint8(init_prob), cv2.COLOR_RGB2BGR))
                    np.save(init_prob_filepath.replace('png', 'npy'),
                            np.float16(outputs[("init_prob", img_i)].squeeze().cpu().numpy()))

                if args.save_refined_depth == "True":
                    refined_depth = np.float16(outputs[("depth", img_i, 0)].squeeze(1).cpu().numpy())
                    refined_depth_filepath = os.path.join(refined_depth_dir,
                                                          rgb_basename.replace(img_ext,
                                                                               ".npy"))
                    np.save(refined_depth_filepath, refined_depth)

                    refined_depth_color = colorize_depth(outputs[("depth", img_i, 0)].squeeze(1),
                                                         max_depth=5.0).permute(0, 2, 3,
                                                                                1).squeeze().cpu().numpy()
                    refined_depth_color_filepath = os.path.join(refined_depth_dir,
                                                                rgb_basename.replace(img_ext,
                                                                                     ".jpg"))
                    cv2.imwrite(refined_depth_color_filepath,
                                cv2.cvtColor(np.uint8(refined_depth_color), cv2.COLOR_RGB2BGR))

                if args.save_refined_prob == "True":
                    refined_prob = colorize_probmap(outputs[("fused_prob", img_i)].squeeze(1)).permute(0, 2,
                                                                                                       3,
                                                                                                       1).squeeze().cpu().numpy()
                    refined_prob_filepath = os.path.join(refined_prob_dir,
                                                         rgb_basename.replace("color" + img_ext,
                                                                              "refined_prob.jpg"))
                    cv2.imwrite(refined_prob_filepath,
                                cv2.cvtColor(np.uint8(refined_prob), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':

    # parse arguments and check
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = torch.device(args.device)

    # model, optimizer

    print("hybrid model EST transformer: ", args.IF_EST_transformer == "True")
    model = DepthNetHybrid(ndepths=args.ndepths, depth_min=args.depth_min,
                           depth_max=args.depth_max, resnet=args.resnet,
                           IF_EST_transformer=args.IF_EST_transformer == "True", )

    model.to(device)

    if args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))

        model.load_state_dict(state_dict['model'])

    if args.eval_dataset == "scannet":
        test_scannet_seq(model, args)
    elif args.eval_dataset == "7scenes" or args.eval_dataset == "sun3d":
        test_sevenscenes_seq(model, args)
    else:
        print("wrong dataset")
