import math
import sys
import time
import torch
from pathlib import Path

import torchvision.models.detection.mask_rcnn
import numpy as np

from references.coco_utils import get_coco_api_from_dataset
from references.coco_eval import CocoEvaluator
import references.utils as utils


def train_one_epoch_cls(model, optimizer, data_loader, device, epoch, print_freq, name='ex0'):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ", name=name, agent='train')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # images = list(image.to(device) for image in images)
        images = images.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = targets.to(device)

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, name='ex0', use_aug=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ", name=name)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    if use_aug:
        model_without_ddp = model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = model.module
        if epoch <= 1:
            model_without_ddp.roi_heads.use_aug = False
        else:
            model_without_ddp.roi_heads.use_aug = True

    for images, targets in metric_logger.log_every(data_loader, print_freq, header, epoch):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def get_coco_eval_bbox(coco_evaluator, ap=1, iouThr=None, catIds=None, areaRng='all', maxDets=100):
    coco_eval = coco_evaluator.coco_eval['bbox']
    aind = [i for i, aRng in enumerate(coco_eval.params.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(coco_eval.params.maxDets) if mDet == maxDets]

    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = coco_eval.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == coco_eval.params.iouThrs)[0]
            s = s[t]
        if catIds is not None:
            t = np.where(catIds == np.array(coco_eval.params.catIds))[0][0]
            s = s[:, :, t]
            s = s[:, :, None, :, :]
        s = s[:,:,:,aind,mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = coco_eval.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == coco_eval.params.iouThrs)[0]
            s = s[t]
        if catIds is not None:
            t = np.where(catIds == np.array(coco_eval.params.catIds))[0][0]
            s = s[:, t]
            s = s[:, None, :, :]
        s = s[:,:,aind,mind]
    if len(s[s>-1])==0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s>-1])

    return mean_s

@torch.no_grad()
def evaluate(model, data_loader, device, epoch, name='ex0', do_record=False):
    record_path = Path("/media/data1/mx_log")/f"bird_detection_records/{name}"

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", name=name, task='val')
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    if utils.get_rank() == 0:
        for catIds in [None] + coco_evaluator.coco_eval['bbox'].params.catIds:
            p0 = get_coco_eval_bbox(coco_evaluator, 1, catIds=catIds,)
            p1 = get_coco_eval_bbox(coco_evaluator, 1, catIds=catIds, iouThr=.5, maxDets=coco_evaluator.coco_eval['bbox'].params.maxDets[2])
            p2 = get_coco_eval_bbox(coco_evaluator, 1, catIds=catIds, iouThr=.75, maxDets=coco_evaluator.coco_eval['bbox'].params.maxDets[2])
            p3 = get_coco_eval_bbox(coco_evaluator, 1, catIds=catIds, areaRng='small', maxDets=coco_evaluator.coco_eval['bbox'].params.maxDets[2])
            p4 = get_coco_eval_bbox(coco_evaluator, 1, catIds=catIds, areaRng='medium', maxDets=coco_evaluator.coco_eval['bbox'].params.maxDets[2])
            p5 = get_coco_eval_bbox(coco_evaluator, 1, catIds=catIds, areaRng='large', maxDets=coco_evaluator.coco_eval['bbox'].params.maxDets[2])
            r0 = get_coco_eval_bbox(coco_evaluator, 0, catIds=catIds, maxDets=coco_evaluator.coco_eval['bbox'].params.maxDets[0])
            r1 = get_coco_eval_bbox(coco_evaluator, 0, catIds=catIds, maxDets=coco_evaluator.coco_eval['bbox'].params.maxDets[1])
            r2 = get_coco_eval_bbox(coco_evaluator, 0, catIds=catIds, maxDets=coco_evaluator.coco_eval['bbox'].params.maxDets[2])
            r3 = get_coco_eval_bbox(coco_evaluator, 0, catIds=catIds, areaRng='small', maxDets=coco_evaluator.coco_eval['bbox'].params.maxDets[2])
            r4 = get_coco_eval_bbox(coco_evaluator, 0, catIds=catIds, areaRng='medium', maxDets=coco_evaluator.coco_eval['bbox'].params.maxDets[2])
            r5 = get_coco_eval_bbox(coco_evaluator, 0, catIds=catIds, areaRng='large', maxDets=coco_evaluator.coco_eval['bbox'].params.maxDets[2])

            if catIds is None:
                catIds = 'all'
                if do_record and utils.get_rank() == 0:
                    print("RECORD!!!!!")
                    with record_path.open("a") as f:
                        f.write(f"{p0}, {p1}, {p2}\n")
            metric_logger.tb_writer.add_scalar(f"val/ap_05_95_c{catIds}_aall_m100", p0, epoch)
            metric_logger.tb_writer.add_scalar(f"val/ap_05_c{catIds}_aall_m100", p1, epoch)
            metric_logger.tb_writer.add_scalar(f"val/ap_75_c{catIds}_aall_m100", p2, epoch)
            metric_logger.tb_writer.add_scalar(f"val/ap_05_95_c{catIds}_asm_m100", p3, epoch)
            metric_logger.tb_writer.add_scalar(f"val/ap_05_95_c{catIds}_amd_m100", p4, epoch)
            metric_logger.tb_writer.add_scalar(f"val/ap_05_95_c{catIds}_alg_m100", p5, epoch)

            metric_logger.tb_writer.add_scalar(f"val/ar_05_95_c{catIds}_aall_m1", r0, epoch)
            metric_logger.tb_writer.add_scalar(f"val/ar_05_95_c{catIds}_aall_m10", r1, epoch)
            metric_logger.tb_writer.add_scalar(f"val/ar_05_95_c{catIds}_aall_m100", r2, epoch)
            metric_logger.tb_writer.add_scalar(f"val/ar_05_95_c{catIds}_asm_m100", r3, epoch)
            metric_logger.tb_writer.add_scalar(f"val/ar_05_95_c{catIds}_amd_m100", r4, epoch)
            metric_logger.tb_writer.add_scalar(f"val/ar_05_95_c{catIds}_alg_m100", r5, epoch)

    torch.set_num_threads(n_threads)
    return coco_evaluator


@torch.no_grad()
def evaluate_cls(model, data_loader, device, name='ex0', epoch_num=0):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", name=name, agent='val')
    header = 'Test:'

    correct1 = torch.LongTensor([0]).squeeze().to(device)
    correct2 = torch.LongTensor([0]).squeeze().to(device)
    total1 = torch.LongTensor([0]).squeeze().to(device)
    total2 = torch.LongTensor([0]).squeeze().to(device)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        # image = list(img.to(device) for img in image)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        image = image.to(device)
        targets = targets.to(device)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        model_time = time.time() - model_time

        evaluator_time = time.time()
        correct1 += (outputs[targets==0] == targets[targets==0]).sum()
        correct2 += (outputs[targets==1] == targets[targets==1]).sum()
        total1 += outputs[targets==0].size(0)
        total2 += outputs[targets==1].size(0)
        evaluator_time = time.time() - evaluator_time

        metric_logger.update(model_time=model_time)

    correct1 = utils.reduce_tensor(correct1, average=False)
    total1 = utils.reduce_tensor(total1, average=False)
    accuracy1 = correct1.item() / total1.item()
    correct2 = utils.reduce_tensor(correct2, average=False)
    total2 = utils.reduce_tensor(total2, average=False)
    accuracy2 = correct2.item() / total2.item()

    metric_logger.update(accuracy1=accuracy1)
    metric_logger.update(accuracy2=accuracy2)
    # metric_logger.update(accuracy3=correct3/total3)
    metric_logger.tb_writer.add_scalar(f'val/accuracy1', accuracy1, epoch_num)
    metric_logger.tb_writer.add_scalar(f'val/accuracy2', accuracy2, epoch_num)
    # metric_logger.tb_writer.add_scalar(f'val/accuracy3', correct3/total3, epoch_num)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # accumulate predictions from all images
    torch.set_num_threads(n_threads)