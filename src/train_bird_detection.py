r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time
from pathlib import Path

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from references.coco_utils import get_coco, get_coco_kp

from references.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from references.engine import train_one_epoch, evaluate

from references import utils
from references import transforms as T

from datasets.bird_dataset import BirdDataset
from modeling.maskrcnn_resnet50_fpn import get_model, get_model_attention

import torch
import numpy as np
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)



def get_transform(train):
    transforms = []
    # if train:
        # transforms = [T.RandomColorJitter(), T.RandomGrayscale()]
        # transforms = [T.RandomColorJitter()]
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset = BirdDataset(name=args.dataset, transforms=get_transform(True), train=True, small_set=args.small_set,
        only_instance=args.only_instance)
    dataset_test = BirdDataset(name=args.dataset, transforms=get_transform(False), train=False, small_set=args.small_set,
        only_instance=args.only_instance)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    if args.model == 'normal':
        print('normal model')
        model = get_model_attention(num_classes=args.num_classes,
            use_focal_loss=args.use_focal_loss, focal_gamma=args.focal_gamma,
            use_attention=False)
    elif args.model == 'attention':
        print('attention model')
        model = get_model_attention(num_classes=args.num_classes,
                                    attention_head_output_channels=args.num_parts,
                                    use_focal_loss=args.use_focal_loss, focal_gamma=args.focal_gamma,
                                    use_attention=True)
    elif args.model == 'attention_transformer':
        print('attention transformer model')
        model = get_model_attention(transformer=True, num_classes=args.num_classes,
                                    attention_head_output_channels=args.num_parts,
                                    use_focal_loss=args.use_focal_loss, focal_gamma=args.focal_gamma,
                                    use_attention=True)
    else:
        raise Exception("'model' must be 'normal' or 'attention' or 'attention_transformer'")
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        print("load resume")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.ft:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluator = evaluate(model, data_loader_test, device=device, epoch=0, name=Path(args.output_dir).name, do_record=True)
        return evaluator

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq,
                        name=Path(args.output_dir).name, use_aug=args.use_aug)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        if not args.no_eval:
            do_record = False
            if epoch == args.epochs - 1:
                do_record = True
            evaluate(model, data_loader_test, epoch=epoch, device=device, name=Path(args.output_dir).name, do_record=do_record)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--model', default='normal', help='model name')
    parser.add_argument('--num-parts', default=8, type=int, help='parts used by attention model')
    parser.add_argument('--use-aug', action='store_true', help='whether to use aug')
    parser.add_argument('--num-classes', default=4, type=int, help='number of classes to identify')
    parser.add_argument('--use-focal-loss', action="store_true", help="whether to use focal loss")
    parser.add_argument('--focal-gamma', default=2., type=float, help='focal gamma')
    parser.add_argument('--dataset', default='real', help='dataset name')
    parser.add_argument('--small-set', action='store_true', help='small set for synthesized dataset')
    parser.add_argument('--only-instance', action='store_true', help='detect instance only')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default="/media/data1/mx_model/bird_detection/bird_detection/ex0", help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--ft', action="store_true", help='fine tune')
    parser.add_argument('--no-eval', action='store_true', help='no evaluation')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    evaluator = main(args)
