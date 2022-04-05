# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pickle

import torch
import torch.distributed as dist
from joblib import Parallel, delayed
from torch.utils.data.sampler import (
    SubsetRandomSampler,
    Sampler
)
from tqdm import tqdm
import numpy as np

from datasets.AVideoDataset import AVideoDataset
from utils import (
    init_distributed_mode,
    init_signal_handler,
    load_model_parameters
)
from model import load_model


class Subset_Sampler(Sampler):
    """
    Sample indices.
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_cluster_assignments_gpu(
        args,
        dataset,
        model,
        logger=None,
        device='cuda'
):
    # clear cache at beginning
    torch.cuda.empty_cache()
    model.eval()
    N = len(dataset)
    # this process deals only with a subset of the dataset
    local_nmb_data = N // args.world_size
    train_indices = torch.arange(
        args.rank * local_nmb_data,
        (args.rank + 1) * local_nmb_data
    ).int()
    # create subset sampler
    sampler = Subset_Sampler(train_indices)

    # we need a data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=None,
        shuffle=False,
        #drop_last=True  # New
    )

    # Ensure processes reach to end of optim clusters
    if args.distributed:
        dist.barrier()

    # use GAP features
    if args.headcount > 1:
        model.module.return_features = True
    aggregtensor = torch.cuda.DoubleTensor if args.headcount == 1 else torch.cuda.FloatTensor
    dtype = torch.float64 if args.headcount == 1 else torch.float32

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # print(f"{batch_idx}/{len(dataloader)}", end='\r', flush=True)
        # if batch_idx > 1:
        #     break
        # print(f"{batch_idx}/{len(dataloader)}", flush=True)
        # Get data
        # if batch_idx > len(dataloader)//10:
        #    break
        video, audio, label, _, _, filename = batch

        # Move to GPU
        video = video.cuda(non_blocking=True)
        audio = audio.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # Forward pass
        # feat_v, feat_a = model(video, audio) # OLD

        # Slow
        f_v, f_a = [], []
        for sec in range(len(video[0])):
            vid, aud = model(video[:, sec], audio[:, sec])
            f_v.append(vid)
            f_a.append(aud)
        feat_v = torch.stack(f_v, dim=0).mean(dim=0)
        feat_a = torch.stack(f_a, dim=0).mean(dim=0)

        """
        video_splits = torch.squeeze(video, dim=0).split(32)
        audio_splits = torch.squeeze(audio, dim=0).split(32)
        vids = torch.zeros((video.shape[1], 512), dtype=dtype, device=device)
        auds = torch.zeros((audio.shape[1], 512), dtype=dtype, device=device)
        f = 0
        for idx in range(len(video_splits)):
            vid, aud = model(video_splits[idx], audio_splits[idx])
            t = f + video_splits[idx].shape[0]
            vids[f:t] = vid
            auds[f:t] = aud
            f = t
        #f_v_fast, f_a_fast = model(torch.squeeze(video, dim=0), torch.squeeze(audio, dim=0))
        #feat_v = f_v_fast.mean(dim=0)
        #feat_a = f_a_fast.mean(dim=0)
        feat_v = vids.mean(dim=0)
        feat_a = auds.mean(dim=0)
        """

        # gather the features computed by all processes
        if args.distributed:
            all_feat_v_list = [aggregtensor(feat_v.size()) for src in range(args.world_size)]
            all_feat_a_list = [aggregtensor(feat_a.size()) for src in range(args.world_size)]
            all_labels_list = [torch.zeros(label.size(0), dtype=torch.long).cuda() for _ in range(args.world_size)]
            all_filenames_list = [torch.zeros(filename.size(0), dtype=torch.long).cuda() for _ in
                                  range(args.world_size)]

            dist.all_gather(all_feat_v_list, feat_v)
            dist.all_gather(all_feat_a_list, feat_a)
            dist.all_gather(all_labels_list, label)
            dist.all_gather(all_filenames_list, filename)
        else:
            all_feat_v_list = [feat_v]
            all_feat_a_list = [feat_a]
            all_labels_list = [label]
            all_filenames_list = [filename]

        # only main process stores all features
        if args.rank == 0:
            all_feat_v = torch.cat(all_feat_v_list)
            all_feat_a = torch.cat(all_feat_a_list)
            all_labels = torch.cat(all_labels_list)#.cpu()
            all_filenames = np.concatenate(all_filenames_list)

        if batch_idx == 0 and (args.rank == 0):
            fr = 0
            if len(feat_v.size()) > 1:
                K = feat_v.size(1)
            else:
                K = feat_v.size(0)
            PS_v = torch.zeros((N, K), dtype=dtype, device=device)
            PS_a = torch.zeros((N, K), dtype=dtype, device=device)
            labels = torch.zeros(N, dtype=torch.long, device=device)
            filenames = []

        # fill in arrays on main node
        if args.rank == 0:
            if len(all_feat_v.size()) > 1:
                to = fr + all_feat_v.shape[0]
            else:
                to = fr + 1
            PS_v[fr: to] = all_feat_v
            PS_a[fr: to] = all_feat_a
            labels[fr: to] = all_labels
            filenames[fr: to] = all_filenames
            fr = to

        if args.distributed:
            dist.barrier()

    # Dump results
    if args.rank == 0:
        # PS_v_heads, PS_a_heads = [], []
        # for h in range(args.headcount):
        #     head_a = getattr(model.module, f'mlp_a{h}')
        #     head_v = getattr(model.module, f'mlp_v{h}')
        #     PS_v_heads.append(head_v.forward(PS_v))
        #     PS_a_heads.append(head_a.forward(PS_a))
        # PS = [PS_v_heads, labels, PS_a_heads]
        PS = [PS_v, labels, PS_a, filenames]  # only save pre-mlp 512 dim features
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f'{args.exp_desc}.pkl')
        with open(save_path, 'wb') as handle:
            pickle.dump(PS, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Finished Dumping!")

    # Make other processes wait
    if args.distributed:
        dist.barrier()

    return


def parse_args():
    def str2bool(v):
        v = v.lower()
        if v in ('yes', 'true', 't', '1'):
            return True
        elif v in ('no', 'false', 'f', '0'):
            return False
        raise ValueError('Boolean argument needs to be true or false. '
                         'Instead, it is %s.' % v)

    import argparse
    parser = argparse.ArgumentParser(description='Video Cluster Fit')
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--output_dir', default='.', type=str,
                        help='path where to save')
    parser.add_argument('--weights_path', default='', type=str,
                        help='Path to weights file')
    parser.add_argument('--exp_desc', default='vggsound_clusters', type=str,
                        help='desc of exp')
    parser.add_argument('--pretrained', default='False', type='bool',
                        help="Use pre-trained models from the modelzoo")
    parser.add_argument('--dataset', default='vggsound', type=str,
                        choices=['kinetics', 'vggsound', 'kinetics_sound', 'ave', "audioset_zsl", "ucf", "activity"],
                        help='name of dataset')
    parser.add_argument("--root_dir", type=str, default="/path/to/dataset",
                        help="root dir of dataset")
    parser.add_argument('--mode', default='val', type=str,
                        help='mode of dataset')
    parser.add_argument('--num_data_samples', default=14032, type=int,
                        help='number of samples in dataset')

    # AUDIO UTILS
    parser.add_argument("--num_sec_aud", type=int, default=1,
                        help="number of seconds of audio")
    parser.add_argument("--aud_sample_rate", type=int, default=24000,
                        help="audio sample rate")
    parser.add_argument("--aud_spec_type", type=int, default=2,
                        help="audio spec type")
    parser.add_argument('--use_volume_jittering', type='bool', default='False',
                        help='use volume jittering')
    parser.add_argument('--use_audio_temp_jittering', type='bool', default='False',
                        help='use audio temporal jittering')
    parser.add_argument('--z_normalize', type='bool', default='True',
                        help='z-normalize the audio')

    ### DATA
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--workers', default=10, type=int,
                        help='number of data loading workers (default: 16)')

    ### MODEL
    parser.add_argument("--vid_base_arch", default="r2plus1d_18", type=str,
                        help="video architecture", choices=['r2plus1d_18'])
    parser.add_argument("--aud_base_arch", default="resnet9", type=str,
                        help="audio architecture", choices=['resnet9', 'resnet18'])
    parser.add_argument('--use_mlp', type='bool', default='True',
                        help='use MLP head')
    parser.add_argument('--norm_feat', type='bool', default='False',
                        help='normalize pre-mlp features')
    parser.add_argument("--num_clusters", default=256, type=int,
                        help="final layer dimension in projection head")
    parser.add_argument("--headcount", default=1, type=int,
                        help="number of heads")

    # distributed training parameters
    parser.add_argument("--dist_url", default="env://", type=str,
                        help="""url used to set up distributed
                        training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=-1, type=int, help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int,
                        help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")
    parser.add_argument("--distributed", default='False', type='bool',
                        help="in distributed mode")

    # own arguments
    parser.add_argument("--run", choices=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], default=-1, type=int,
                        help="Number of runs (out of 7) for parallel evaluation.")

    args = parser.parse_args()
    return args


def valid_audio(index, args):
    dataset = AVideoDataset(
        ds_name=args.dataset,
        root_dir=args.root_dir,
        mode=args.mode,
        num_frames=30,
        sample_rate=1,
        train_crop_size=112,
        num_data_samples=args.num_data_samples,
        target_fps=30,
        decode_audio=True,
        num_sec=args.num_sec_aud,
        aud_sample_rate=args.aud_sample_rate,
        aud_spec_type=args.aud_spec_type,
        use_volume_jittering=args.use_volume_jittering,
        use_temporal_jittering=args.use_audio_temp_jittering,
        z_normalize=args.z_normalize,
        center_crop=True,
        temp_jitter=False,
    )
    try:
        tmp = dataset[index]
        print(f"{index}: True", end='\r', flush=True)
        return True
    except Exception as e:
        print(f"Index: {index}")
        print(f"{dataset._path_to_videos[index]}")
        print(e)
        print("-------------------------------------------------------------------")
        return False


def filter_audios(args):
    print("START FILTER AUDIOS")
    all_indices = Parallel(n_jobs=30)(
        delayed(valid_audio)(aud_idx, args) for aud_idx in range(len(dataset)))
    invalid_indices = [i for i, val in enumerate(all_indices) if not val]
    print(invalid_indices)
    return dataset._path_to_videos[invalid_indices]
    # return valid_indices


if __name__ == '__main__':

    # parse args
    args = parse_args()

    # Init distributed mode
    if args.distributed:
        init_distributed_mode(args)
        init_signal_handler()
    else:
        args.rank = 0
        args.world_size = 1

    # Set up dataset hyper-params
    if args.dataset == 'vggsound':
        args.num_clusters = 309
        if args.mode == 'train':
            args.num_data_samples = 170752
        else:
            args.num_data_samples = 14032
    elif args.dataset == 'kinetics':
        args.num_clusters = 400
        if args.mode == 'train':
            args.num_data_samples = 230976
        else:
            args.num_data_samples = 18968
    elif args.dataset == 'kinetics_sound':
        args.num_clusters = 32
        if args.mode == 'train':
            args.num_data_samples = 22408
        else:
            args.num_data_samples = 22408
    elif args.dataset == 'ave':
        args.num_clusters = 28
        if args.mode == 'train':
            args.num_data_samples = 3328
        else:
            args.num_data_samples = 3328
    elif args.dataset == "audioset_zsl":
        args.num_clusters = 309
        if args.mode == "train":
            args.num_data_samples = 93865
        elif args.mode == "val":
            args.num_data_samples = 31271
        else:
            args.num_data_samples = 31280
    elif args.dataset == "ucf":
        args.num_clusters = 101
        args.num_data_samples = 13521
    elif args.dataset == "activity":
        args.num_clusters = 200
        args.num_data_samples = 20694

    # Get dataset
    dataset = AVideoDataset(
        ds_name=args.dataset,
        root_dir=args.root_dir,
        mode=args.mode,
        num_frames=30,
        sample_rate=1,
        train_crop_size=112,
        num_data_samples=args.num_data_samples,
        target_fps=30,
        decode_audio=True,
        num_sec=args.num_sec_aud,
        aud_sample_rate=args.aud_sample_rate,
        aud_spec_type=args.aud_spec_type,
        use_volume_jittering=args.use_volume_jittering,
        use_temporal_jittering=args.use_audio_temp_jittering,
        z_normalize=args.z_normalize,
        center_crop=True,
        temp_jitter=False,
        run=args.run
    )

    weight_path_type = type(args.weights_path)
    if weight_path_type == str:
        weight_path_not_none = args.weights_path != 'None'
    else:
        weight_path_not_none = args.weights_path is not None

    #  Load model
    args.headcount = args.headcount if weight_path_not_none else 1
    model = load_model(
        vid_base_arch=args.vid_base_arch,
        aud_base_arch=args.aud_base_arch,
        pretrained=args.pretrained,
        norm_feat=args.norm_feat,
        use_mlp=args.use_mlp,
        headcount=args.headcount,
        num_classes=args.num_clusters,
    )

    # Load model weights
    to_restore = {'epoch': 0}
    if not args.pretrained:
        if weight_path_not_none:
            print("Loading model weights")
            if os.path.exists(args.weights_path):
                ckpt_dict = torch.load(args.weights_path)
                model_weights = ckpt_dict["model"]
                epoch = ckpt_dict["epoch"]
                print(f"Epoch checkpoint: {epoch}")
                load_model_parameters(model, model_weights)
        else:
            print("Random weights")

    # Put model in distributed mode
    model = model.cuda()
    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False
        )
    else:
        model = torch.nn.DataParallel(model)

    # Get cluster assignments
    with torch.no_grad():
        # print(filter_audios(args))
        get_cluster_assignments_gpu(args, dataset, model)
