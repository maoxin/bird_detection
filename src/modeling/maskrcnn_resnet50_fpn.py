import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN, resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url

from modeling.attention_heads import AttentionHead, AttentionHeadTransformer, FastRCNNPredictorAttention, RoIHeads

def get_model(num_classes=4, pretrained=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_model_attention(num_classes=4, pretrained=True, transformer=False):
    model = fasterrcnn_resnet50_fpn_attention(pretrained=pretrained, transformer=transformer)
    
    in_channels_two_ml_head = model.roi_heads.box_predictor.cls_score.in_features
    in_channels_attention_head = model.roi_heads.box_head_attention.in_channels * model.roi_heads.box_head_attention.out_channels
    model.roi_heads.box_predictor_attention = FastRCNNPredictorAttention(in_channels_two_ml_head, in_channels_attention_head, num_classes)

    model.roi_heads.box_predictor = None

    return model

class FasterRCNNAttention(FasterRCNN):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 attention_head_output_channels=8):
        super().__init__(backbone, num_classes=num_classes,
                 # transform parameters
                 min_size=min_size, max_size=max_size,
                 image_mean=image_mean, image_std=image_std,
                 # RPN parameters
                 rpn_anchor_generator=rpn_anchor_generator, rpn_head=rpn_head,
                 rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                 rpn_post_nms_top_n_train=rpn_post_nms_top_n_train, rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
                 rpn_nms_thresh=rpn_nms_thresh,
                 rpn_fg_iou_thresh=rpn_fg_iou_thresh, rpn_bg_iou_thresh=rpn_bg_iou_thresh,
                 rpn_batch_size_per_image=rpn_batch_size_per_image, rpn_positive_fraction=rpn_positive_fraction,
                 # Box parameters
                 box_roi_pool=box_roi_pool, box_head=box_head, box_predictor=box_predictor,
                 box_score_thresh=box_score_thresh, box_nms_thresh=box_nms_thresh, box_detections_per_img=box_detections_per_img,
                 box_fg_iou_thresh=box_fg_iou_thresh, box_bg_iou_thresh=box_bg_iou_thresh,
                 box_batch_size_per_image=box_batch_size_per_image, box_positive_fraction=box_positive_fraction,
                 bbox_reg_weights=bbox_reg_weights)

        attention_head_in_channels = self.backbone.out_channels
        box_head_attention = AttentionHead(attention_head_in_channels, attention_head_output_channels)
        
        in_channels_two_ml_head = self.roi_heads.box_predictor.cls_score.in_features
        in_channels_attention_head = box_head_attention.in_channels * box_head_attention.out_channels
        box_predictor_attention = FastRCNNPredictorAttention(in_channels_two_ml_head, in_channels_attention_head, num_classes)

        self.roi_heads = RoIHeads(
            # Box
            self.roi_heads.box_roi_pool, self.roi_heads.box_head, box_head_attention,
            self.roi_heads.box_predictor, box_predictor_attention,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

class FasterRCNNAttentionTransformer(FasterRCNN):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 attention_head_output_channels=8):
        super().__init__(backbone, num_classes=num_classes,
                 # transform parameters
                 min_size=min_size, max_size=max_size,
                 image_mean=image_mean, image_std=image_std,
                 # RPN parameters
                 rpn_anchor_generator=rpn_anchor_generator, rpn_head=rpn_head,
                 rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                 rpn_post_nms_top_n_train=rpn_post_nms_top_n_train, rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
                 rpn_nms_thresh=rpn_nms_thresh,
                 rpn_fg_iou_thresh=rpn_fg_iou_thresh, rpn_bg_iou_thresh=rpn_bg_iou_thresh,
                 rpn_batch_size_per_image=rpn_batch_size_per_image, rpn_positive_fraction=rpn_positive_fraction,
                 # Box parameters
                 box_roi_pool=box_roi_pool, box_head=box_head, box_predictor=box_predictor,
                 box_score_thresh=box_score_thresh, box_nms_thresh=box_nms_thresh, box_detections_per_img=box_detections_per_img,
                 box_fg_iou_thresh=box_fg_iou_thresh, box_bg_iou_thresh=box_bg_iou_thresh,
                 box_batch_size_per_image=box_batch_size_per_image, box_positive_fraction=box_positive_fraction,
                 bbox_reg_weights=bbox_reg_weights)

        attention_head_in_channels = self.backbone.out_channels
        box_head_attention = AttentionHeadTransformer(attention_head_in_channels, attention_head_output_channels)
        
        in_channels_two_ml_head = self.roi_heads.box_predictor.cls_score.in_features
        in_channels_attention_head = box_head_attention.in_channels * box_head_attention.out_channels
        box_predictor_attention = FastRCNNPredictorAttention(in_channels_two_ml_head, in_channels_attention_head, num_classes)

        self.roi_heads = RoIHeads(
            # Box
            self.roi_heads.box_roi_pool, self.roi_heads.box_head, box_head_attention,
            self.roi_heads.box_predictor, box_predictor_attention,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}
def fasterrcnn_resnet50_fpn_attention(pretrained=False, progress=True,
                                      num_classes=91, pretrained_backbone=True,
                                      transformer=False, **kwargs):
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    if not transformer:
        model = FasterRCNNAttention(backbone, num_classes, **kwargs)
    else:
        model = FasterRCNNAttentionTransformer(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)

    return model