import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ResNet50(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()
        self.extractor = resnet50(pretrained)
        self.extractor.fc = nn.Linear(self.extractor.fc.in_features, num_classes)

        # self.loss_func = nn.CrossEntropyLoss(weight=torch.tensor([0.03727246, 0.117885, 0.84484253]))
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        y = self.extractor(x)

        losses = {}
        if self.training:
            assert(labels is not None)
            losses['loss_classification'] = self.loss_func(y, labels)
            return losses
        else:
            return y.argmax(dim=1)

class ResNet50Attention(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, num_attention=32, use_transformer=False):
        super().__init__()
        self.extractor = ResNet50AttentionExtractor(pretrained, num_classes, num_attention, use_transformer)
        self.fc = nn.Linear(num_attention * 512, num_classes)

        # self.loss_func = nn.CrossEntropyLoss(weight=torch.tensor([0.03727246, 0.117885, 0.84484253]))
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        x, attention_map = self.extractor(x)
        y = self.fc(x)

        losses = {}
        if self.training:
            assert(labels is not None)
            losses['loss_classification'] = self.loss_func(y, labels)
            losses['ck_loss'] = self.extractor.ck_loss(x, labels)
            return losses
        else:
            return y.argmax(dim=1)

class ResNet50AttentionExtractor(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, num_attention=32, use_transformer=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_attention = num_attention
        self.use_transformer = use_transformer

        self.resnet = resnet50(True)
        self.resnet.avgpool = None
        self.resnet.fc = None

        self.attention_conv = nn.Conv2d(self.resnet.layer4[-1].conv3.out_channels, num_attention,
                                        kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.pool_conv = nn.Conv1d(self.attention_conv.in_channels, 512, 1)

        if use_transformer:
            transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=3)

        self.register_buffer('ck_buffer', torch.zeros(num_classes, num_attention, 512))

    def ck_loss(self, x, labels):
        x = x.view(-1, self.num_attention, 512)
        # (N, 32, 512)
        x = F.normalize(x, dim=2)

        # x = x[labels!=0]
        # labels = labels[labels!=0]

        ck_buffer = self.ck_buffer[labels]
        ck_loss = ((x - ck_buffer)**2).sum() / x.size(0)

        with torch.no_grad():
            for label in torch.unique(labels):
                self.ck_buffer[label] = self.ck_buffer[label] * 0.9 + x[labels==label].mean(0) * 0.1
        
        return ck_loss

    def resnet_forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        return x
    
    def forward(self, x):
        x = self.resnet_forward(x)
        # (N, 2048, 8, 8)
        attention_map = self.attention_conv(x)
        # (N, 32, 8, 8)

        x = attention_map.unsqueeze(2) * x.unsqueeze(1)
        # (N, 32, 2048, 8, 8)

        x = self.pool(x.flatten(1, 2)).flatten(1).view(-1, x.size(1), x.size(2))
        # (N, 32, 2048)
        x = self.pool_conv(x.transpose(1, 2)).transpose(1, 2)
        #(N, 32, 512)

        if self.use_transformer:
            x = F.relu(self.transformer_encoder(x))
            # (N, 32, 512)
        x = x.flatten(1)
        #(N, 32 * 512)

        return x, attention_map



    