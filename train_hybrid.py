import argparse, time, gc, datetime, sys
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from hybrid_models.model_hybrid import *
from utils.utils import *
import torch.distributed as dist

from data.scannet import *

from utils.misc_utils import *

from natsort import natsorted

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of Epipolar Spatio-Temporal depth network')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--device', default='cuda', help='select model')

parser.add_argument('--datapath', help='train datapath')
parser.add_argument('--testdatapath', help='test datapath')
parser.add_argument('--trainlist', help='train list', default="./data/scannet_split/train_split.txt")
parser.add_argument('--testlist', help='test list', default="./data/scannet_split/test_split.txt")

parser.add_argument('--reloadscan', help='reload the scannet dataset', default="True")

parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0004, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="1,3,5:2",
                    help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=4e-4, help='weight decay')

parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--seq_len', type=int, default=5, help='the length of each img seq')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--restore_part', type=str, default=None, help='restore part of network from checkpoints, '
                                                                   'such as feature')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=1, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')

parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed')
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

parser.add_argument('--lwindow', type=int, default=3, help='the num of frames in local window')
parser.add_argument('--resnet', type=int, default=50, help='resnet 18 or 50 as backbone')

parser.add_argument('--IF_EST_transformer', type=str, default="True", help='IF_EST_transformer')

parser.add_argument('--fix_matchingFeature', type=str, default="False", help='fix_matchingFeature')
parser.add_argument('--fix_semanticFeature', type=str, default="False", help='fix_semanticFeature')

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1


# is_distributed = False


# main function
def train(model, optimizer, TrainImgLoader, train_sampler, start_epoch, args):
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0 / 3, warmup_iters=500,
                                     last_epoch=len(TrainImgLoader) * start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        if train_sampler is not None:
            train_sampler.set_epoch(
                epoch_idx)  # if don't use this, the sampler to generate same ordering data each epoch

        if epoch_idx >= start_epoch + 1:
            args.summary_freq = 100

        if epoch_idx >= 3:
            clip_grad_value = 1.
        else:
            clip_grad_value = 10.

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            # print(batch_idx, dist.get_rank(), sample['start_frame'])

            scalar_outputs, image_outputs, pre_outputs = train_sample(
                args, sample, model, optimizer, clip_grad_value)

            lr_scheduler.step()
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    if global_step % (args.summary_freq * 10) == 0:
                        save_images(logger, 'train', image_outputs, global_step)

                    print(
                        "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, "
                        "loss_3 = {:.3f}, loss_2 = {:.3f}, loss_1 = {:.3f}, loss_0 = {:.3f},  "
                        "delta_3 = {:.4f},  delta_2 = {:.4f}, delta_1 = {:.4f},  delta_0 = {:.4f},"
                        "thred_3 = {:.3f}, thred_2 = {:.3f}, thred_1 = {:.3f}, thred_0 = {:.3f}, "
                        "time={:.3f}".format(
                            epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                            optimizer.param_groups[0]["lr"], scalar_outputs["loss"],
                            scalar_outputs["loss_3"], scalar_outputs["loss_2"],
                            scalar_outputs["loss_1"], scalar_outputs["loss_0"],

                            scalar_outputs["delta_3"], scalar_outputs["delta_2"],
                            scalar_outputs["delta_1"], scalar_outputs["delta_0"],

                            scalar_outputs["thred_3"], scalar_outputs["thred_2"],
                            scalar_outputs["thred_1"], scalar_outputs["thred_0"],
                            time.time() - start_time)
                    )

                del scalar_outputs, image_outputs

                if global_step % 5000 == 0:
                    torch.save({
                        'epoch': epoch_idx,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        "{}/model_epoch{:0>6}_iter{:0>8}.ckpt".format(args.logdir, epoch_idx, global_step))

        # checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()


def train_sample(args, sample, model, optimizer, grad_clip_value=5):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)

    dmaps_gt = sample_cuda['dmaps']

    views_num_for_loss = dmaps_gt.shape[1] - 2

    # print(sample["img_path"])

    with torch.autograd.set_detect_anomaly(True):
        outputs, losses = model(sample_cuda["imgs"], sample_cuda["cam_poses"],
                                sample_cuda["cam_intr"],
                                sample_cuda,
                                mode='train')

        # if losses['loss'] > 0.2 and optimizer.param_groups[0]["lr"] < 1e-5:
        #     print(sample["img_path"])

        if is_distributed and args.using_apex:
            with amp.scale_loss(losses['loss'], optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            losses['loss'].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value, norm_type=2)  # use small gradient
        optimizer.step()

    image_outputs = {}

    depth_scales = [0, 1, 2, 3]

    for img_i in range(views_num_for_loss):
        image_outputs["depth_gt_img{}".format(img_i)] = colorize_depth(
            dmaps_gt[:, img_i + 1, 0, :, :], max_depth=5.0)

        for depth_scale in depth_scales:
            image_outputs["depth_s{}_img{}".format(depth_scale, img_i)] = colorize_depth(
                outputs[("depth", img_i, depth_scale)].squeeze(1), max_depth=5.0)

        image_outputs["init_prob_img{}".format(img_i)] = colorize_probmap(
            outputs[("init_prob", img_i)].squeeze(1))

        image_outputs["fused_prob_img{}".format(img_i)] = colorize_probmap(
            outputs[("fused_prob", img_i)].squeeze(1))

    for img_i in range(dmaps_gt.shape[1]):
        image_outputs["img" + str(img_i)] = sample['imgs'][:, img_i, :, :, :]

    if is_distributed:
        losses = reduce_scalar_outputs(losses)

    return tensor2float(losses), tensor2numpy(
        image_outputs), outputs


def write_images(outputs, save_dir, ref_name, max_depth=930):
    names = ref_name.split("/")
    save_dir = os.path.join(save_dir, names[-2], names[-1].strip(".png"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for key, value in outputs.items():
        if "depth" in key:
            value = colorize_depth(value.squeeze(), max_depth).permute(1, 2, 0).cpu().numpy()

        elif "conf" in key:
            value = value.squeeze().permute(1, 2, 0).cpu().numpy()

        elif "ref_img" in key:
            value = value.squeeze().permute(1, 2, 0).cpu().numpy() * 255
            value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
        else:
            value = value.squeeze().cpu().numpy()
            value = value / value.max() * 255

        cv2.imwrite(os.path.join(save_dir, key + ".png"), value)


if __name__ == '__main__':
    # parse arguments and check
    args = parser.parse_args()

    # using sync_bn by using nvidia-apex, need to install apex.
    if args.sync_bn:
        assert args.using_apex, "must set using apex and install nvidia-apex"
    if args.using_apex:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
            from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    set_random_seed(args.seed)
    device = torch.device(args.device)

    if (not is_distributed) or (dist.get_rank() == 0):
        # create logger for mode "train" and "testall"
        if args.mode == "train":
            current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            if not os.path.isdir(os.path.join(args.logdir, current_time_str)):
                os.makedirs(os.path.join(args.logdir, current_time_str))

            print("current time", current_time_str)
            print("creating new summary file")
            logger = SummaryWriter(os.path.join(args.logdir, current_time_str))

            eval_file = open(os.path.join(args.logdir, "eval_stamps.txt"), 'w+')
        print("argv:", sys.argv[1:])
        print_args(args)

    # model, optimizer

    print("using hybrid model")
    print("EST transformer: ", args.IF_EST_transformer == "True")
    model = DepthNetHybrid(ndepths=args.ndepths, depth_min=args.depth_min,
                           depth_max=args.depth_max, resnet=args.resnet,
                           IF_EST_transformer=args.IF_EST_transformer == "True")

    model.to(device)

    if args.sync_bn:
        import apex

        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    if args.fix_matchingFeature == "True":
        for name, p in model.named_parameters():
            if "matchingFeature" in name:
                print(name)
                p.requires_grad = False
    if args.fix_semanticFeature == "True":
        for name, p in model.named_parameters():
            if "semanticFeature" in name:
                print(name)
                p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999),
                           weight_decay=args.wd)

    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        # saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        saved_models = natsorted(saved_models)
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))

        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        pretrained_state = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        if args.restore_part is None:
            try:
                model.load_state_dict(pretrained_state['model'])
            except:
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_state['model'].items() if
                                   k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
        else:
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_state['model'].items() if
                               "semanticFeature" in k or "matchingFeature" in k}
            print(pretrained_dict.keys())
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)

    if (not is_distributed) or (dist.get_rank() == 0):
        print("start at epoch {}".format(start_epoch))
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.using_apex:
        # Initialize Amp
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale
                                          )

    if is_distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
        )
    else:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    # dataset, dataloader

    if is_distributed:
        train_dataset = ScannetDataset(args.datapath, args.trainlist, depth_min=args.depth_min,
                                       depth_max=args.depth_max, mode='train',
                                       n_frames=args.seq_len,
                                       reloadscan=args.reloadscan == "True")
    else:
        train_dataset = ScannetDataset(args.datapath, args.trainlist, depth_min=args.depth_min,
                                       depth_max=args.depth_max, mode='train',
                                       n_frames=args.seq_len)

    if (not is_distributed) or (dist.get_rank() == 0):
        print("load %d seqs in train_dataset" % (len(train_dataset)))
        # print("load %d seqs in test_dataset" % (len(test_dataset)))

    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank(), shuffle=True)
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=1,
                                    drop_last=True,
                                    pin_memory=args.pin_m)

    else:
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True,
                                    pin_memory=args.pin_m)
        train_sampler = None

    train(model, optimizer, TrainImgLoader, train_sampler, start_epoch, args)
