import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualAttentionGate(nn.Module):
    def __init__(self, feature_channels, context_channels, hidden_channels, initial_gamma=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels + context_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(initial_gamma)))

    def forward(self, feature, context):
        # context 해상도가 다르면 feature 해상도에 맞춰서 attention을 계산합니다.
        if context.shape[2:] != feature.shape[2:]:
            context = F.interpolate(context, size=feature.shape[2:], mode="bilinear", align_corners=False)
        attention = self.gate(torch.cat([feature, context], dim=1))
        return feature * (1.0 + self.gamma * attention), attention


class UNetPlusPlusBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        pretrained_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight.copy_(pretrained_conv1.weight.mean(dim=1, keepdim=True))

        # ResNet34 encoder 출력:
        # e0=64x256, e1=64x128, e2=128x64, e3=256x32, e4=512x16
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        # UNet++ dense connection에서 채널 수를 맞추기 위한 1x1 conv입니다.
        self.conv2_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv1_1 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv1_2 = nn.Conv2d(128, 64, kernel_size=1)

        self.dense2_1 = ConvBNAct(128 + 128, 128)
        self.dense2_2 = ConvBNAct(128, 128)

        self.dense1_1 = ConvBNAct(64 + 64, 64)
        self.dense1_1_2 = ConvBNAct(64, 64)
        self.dense1_2 = ConvBNAct(64 + 64 + 64, 64)
        self.dense1_2_2 = ConvBNAct(64, 64)

        self.dec4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4_conv1 = ConvBNAct(256 + 256, 256)
        self.dec4_conv2 = ConvBNAct(256, 256)

        self.dec3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3_conv1 = ConvBNAct(128 + 128 + 128, 128)
        self.dec3_conv2 = ConvBNAct(128, 128)

        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_conv1 = ConvBNAct(64 + 64 + 64 + 64, 64)
        self.dec2_conv2 = ConvBNAct(64, 64)

        self._initialize_decoder_weights()

    def _initialize_decoder_weights(self):
        backbone_modules = (
            list(self.enc0.modules())
            + list(self.enc1.modules())
            + list(self.enc2.modules())
            + list(self.enc3.modules())
            + list(self.enc4.modules())
        )
        backbone_ids = {id(module) for module in backbone_modules}

        for module in self.modules():
            if isinstance(module, nn.Conv2d) and id(module) not in backbone_ids:
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 16 -> 32 scale decoder
        dec4 = self.dec4(e4)
        dec4 = torch.cat((dec4, e3), dim=1)
        dec4 = self.dec4_conv2(self.dec4_conv1(dec4))

        # UNet++ dense node x2_1: e3 정보를 e2 scale로 올려 e2와 결합합니다.
        x2_1 = self.conv2_1(e3)
        x2_1 = F.interpolate(x2_1, size=e2.shape[2:], mode="bilinear", align_corners=False)
        x2_1 = torch.cat((x2_1, e2), dim=1)
        x2_1 = self.dense2_2(self.dense2_1(x2_1))

        # 32 -> 64 scale decoder
        dec3 = self.dec3(dec4)
        dec3 = torch.cat((dec3, x2_1, e2), dim=1)
        dec3 = self.dec3_conv2(self.dec3_conv1(dec3))

        # UNet++ dense nodes x1_1, x1_2: 128 scale decoder가 더 많은 skip 정보를 받게 합니다.
        x1_1 = self.conv1_1(e2)
        x1_1 = F.interpolate(x1_1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        x1_1 = torch.cat((x1_1, e1), dim=1)
        x1_1 = self.dense1_1_2(self.dense1_1(x1_1))

        x1_2 = self.conv1_2(x2_1)
        x1_2 = F.interpolate(x1_2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        x1_2 = torch.cat((x1_2, e1, x1_1), dim=1)
        x1_2 = self.dense1_2_2(self.dense1_2(x1_2))

        # 최종 공통 feature dec2는 128 scale에서 stenosis head로 전달됩니다.
        dec2 = self.dec2(dec3)
        dec2 = torch.cat((dec2, e1, x1_1, x1_2), dim=1)
        dec2 = self.dec2_conv2(self.dec2_conv1(dec2))

        return dec2, dec3, dec4


class BaselineSegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(64, 64),
            ConvBNAct(64, 48),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(48, 48),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(48, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head[-1].bias, -4.0)

    def forward(self, dec2):
        return self.head(dec2)


class StenosisHeatmapFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lat_128 = ConvBNAct(64, 64, kernel_size=1)
        self.lat_64 = ConvBNAct(128, 64, kernel_size=1)
        self.lat_32 = ConvBNAct(256, 64, kernel_size=1)

        self.refine_32 = ConvBNAct(64, 64)
        self.fuse_64 = nn.Sequential(
            ConvBNAct(64 + 64, 64),
            ConvBNAct(64, 64),
        )
        self.fuse_128 = nn.Sequential(
            ConvBNAct(64 + 64, 64),
            ConvBNAct(64, 64),
        )

        self.attention_gate = ResidualAttentionGate(
            feature_channels=64,
            context_channels=64 + 64,
            hidden_channels=32,
        )
        self.out_heatmap = nn.Sequential(
            ConvBNAct(64, 64),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        nn.init.normal_(self.out_heatmap[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.out_heatmap[-1].bias, -4.0)

    def forward(self, dec2, dec3, dec4):
        # dec4/dec3/dec2를 각각 64채널 lateral feature로 맞춥니다.
        f128 = self.lat_128(dec2)
        f64 = self.lat_64(dec3)
        f32 = self.refine_32(self.lat_32(dec4))

        # top-down FPN: 32 scale 정보를 64 scale로 올려 결합합니다.
        f32_up = F.interpolate(f32, size=f64.shape[2:], mode="bilinear", align_corners=False)
        f64 = self.fuse_64(torch.cat([f64, f32_up], dim=1))

        # 64 scale 정보를 128 scale로 올려 최종 localization feature를 만듭니다.
        f64_up = F.interpolate(f64, size=f128.shape[2:], mode="bilinear", align_corners=False)
        f128 = self.fuse_128(torch.cat([f128, f64_up], dim=1))

        f32_to_128 = F.interpolate(f32, size=f128.shape[2:], mode="bilinear", align_corners=False)
        context = torch.cat([f64_up, f32_to_128], dim=1)
        heat_feature, attention = self.attention_gate(f128, context)
        heatmap_logits = self.out_heatmap(heat_feature)
        return heatmap_logits, heat_feature, attention


class HeatmapGuidedSegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_fusion = nn.Sequential(
            ConvBNAct(64 + 64 + 1, 96),
            ConvBNAct(96, 64),
        )
        self.attention_gate = ResidualAttentionGate(
            feature_channels=64,
            context_channels=64 + 1,
            hidden_channels=32,
        )
        self.refine = nn.Sequential(
            ConvBNAct(64, 64),
            ConvBNAct(64, 64),
        )
        self.up_256 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(64, 48),
        )
        self.up_512 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(48, 32),
        )
        self.out_mask = nn.Conv2d(32, 1, kernel_size=1)
        nn.init.normal_(self.out_mask.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.out_mask.bias, -4.0)

    def forward(self, dec2, heat_feature, heatmap_logits):
        # heatmap probability는 segmentation head가 어디를 더 볼지 알려주는 soft prior입니다.
        heat_feature = heat_feature.detach()
        heatmap_prob = torch.sigmoid(heatmap_logits).detach()
        context = torch.cat([heat_feature, heatmap_prob], dim=1)

        fused = self.input_fusion(torch.cat([dec2, context], dim=1))
        fused, _attention = self.attention_gate(fused, context)
        fused = self.refine(fused)
        fused = self.up_256(fused)
        fused = self.up_512(fused)
        return self.out_mask(fused)


class UNetPlusPlusBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = UNetPlusPlusBackbone()
        self.seg_head = BaselineSegmentationHead()

    def forward(self, x):
        dec2, _dec3, _dec4 = self.backbone(x)
        stenosis_logits = self.seg_head(dec2)
        return {"stenosis": stenosis_logits, "heatmap": None}


class UNetPlusPlusHeatmapAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = UNetPlusPlusBackbone()
        self.heatmap_fpn = StenosisHeatmapFPN()
        self.seg_head = HeatmapGuidedSegmentationHead()

    def forward(self, x):
        dec2, dec3, dec4 = self.backbone(x)
        heatmap_logits, heat_feature, _attention = self.heatmap_fpn(dec2, dec3, dec4)
        stenosis_logits = self.seg_head(dec2, heat_feature, heatmap_logits)
        return {"stenosis": stenosis_logits, "heatmap": heatmap_logits}
