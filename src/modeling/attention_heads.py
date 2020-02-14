import random

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads as RoIHeadsOld

# use AttentionHead as well as TwoMLHead; AttentionHead for classification, TwoMLHead for box regression
class AttentionHead(nn.Module):
    def __init__(self, in_channels, out_channels, representation_size=1024, attention_threshold=0.8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attention_threshold = attention_threshold

        self.attention_conv = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc6 = nn.Linear(in_channels * out_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

        self.register_buffer('ck_buffer', torch.zeros(in_channels, out_channels))
    
    def get_attention_mask(self, attention_map):
        idx = random.randint(0, self.out_channels)
        attention_mask = attention_map[:, idx, :, :]
        attention_mask = (attention_mask - attention_mask.min()) / (attention_mask.max() - attention_mask.min())
        attention_mask = attention_mask > self.attention_threshold

        return attention_mask

    def attention_drop(self, attention_map, x):
        attention_mask = self.get_attention_mask(attention_map)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(x.shape)

        x[attention_mask] = torch.randn_like(x)[attention_mask]
        x = self.avg_pool(x.flatten(1, 2)).flatten(1).view(-1, self.in_channels, self.out_channels)

        return x


    def attention_crop(self, attention_map, x):
        new_x = []

        attention_mask = self.get_attention_mask(attention_map)
        for n, m in enumerate(attention_mask):
            l_i, l_j = torch.where(m == True)
            i_min = l_i.min()
            i_max = l_i.max()
            j_min = l_j.min()
            j_max = l_j.max()

            new_x.append(self.avg_pool(x[n, :, :, i_min:i_max+1, j_min:j_max+1]).unsqueeze(0))
        
        return torch.cat(new_x)

    def forward(self, x):
        attention_map = self.attention_conv(x)
        # (N, output_channels, H, W)
        x = attention_map.unsqueeze(1) * x.unsqueeze(2)
        # new x: (N, in_channels, out_channels, H, W)

        if self.training:
            # when training, Attention Guided Data Augmentation
            if random.random() <= 0.33:
                x = self.attention_crop(attention_map, x)
            elif random.random() <= 0.66:
                x = self.attention_drop(attention_map, x)
            else:
                x = x
            # (N, in_channels, out_channels)

            ck_loss = ((x.mean(0) - self.ck_buffer)**2).sum()
            # not use ck_loss when warm up
            self.ck_buffer = self.center_loss_buffer * 0.9 + x.mean(0) * 0.1
        else:
            x = self.avg_pool(x.flatten(1, 2)).flatten(1).view(-1, self.in_channels, self.out_channels)

        x = x.flatten(1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        if self.training:
            return x, ck_loss
        else:
            return x, None

class RoIHeads(RoIHeadsOld):
    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_head_attention,
                 box_predictor,
                 box_predictor_attention,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Mask
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None,
                 keypoint_roi_pool=None,
                 keypoint_head=None,
                 keypoint_predictor=None,
                 ):
        super().__init__(box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Mask
                 mask_roi_pool,
                 mask_head,
                 mask_predictor,
                 keypoint_roi_pool,
                 keypoint_head,
                 keypoint_predictor)
        
        self.box_head_attention = box_head_attention
        self.box_predictor_attention = box_predictor_attention

    def forward(self, features, proposals, image_shapes, targets=None):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        box_features_attention, ck_loss = self.box_head_attention(box_features)
        # class_logits, box_regression = self.box_predictor(box_features)
        class_logits, box_regression = self.box_predictor_attention(box_features, box_features_attention)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_ck": ck_loss,
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                mask_logits = torch.tensor(0)
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
                and self.keypoint_predictor is not None:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses
        
class FastRCNNPredictorAttention(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN with AttentionHead.
    Arguments:
        in_channels_two_ml_head (int): number of input channels from TwoMLPHead
        in_channels_attention_head (int): number of input channels from AttentionHead
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels_two_ml_head, in_channels_attention_head, num_classes):
        super(FastRCNNPredictorAttention, self).__init__()
        self.cls_score = nn.Linear(in_channels_attention_head, num_classes)
        self.bbox_pred = nn.Linear(in_channels_two_ml_head, num_classes * 4)

    def forward(self, x_two_ml_head, x_attention_head):
        for x in [x_two_ml_head, x_attention_head]:
            if x.dim() == 4:
                assert list(x.shape[2:]) == [1, 1]
        
        x_attention_head = x_attention_head.flatten(start_dim=1)
        scores = self.cls_score(x)

        x_two_ml_head = x_two_ml_head.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x_two_ml_head)

        return scores, bbox_deltas