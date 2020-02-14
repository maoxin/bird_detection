import random

import torch
from torch import nn
import torch.nn.functional as F

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
        self.f6 = nn.Linear(in_channels * out_channels, representation_size)
        self.f7 = nn.Linear(representation_size, representation_size)

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
            if random.random() <= 0.5:
                x = self.attention_crop(attention_map, x)
            else:
                x = self.attention_drop(attention_map, x)
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
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels_attention_head, num_classes)
        self.bbox_pred = nn.Linear(in_channels_two_ml_head, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas