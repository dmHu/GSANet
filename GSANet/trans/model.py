import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from configs import Config

class FeatureEncoder(nn.Module):
    """共享的图像特征编码器"""
    def __init__(self, backbone='resnet34', pretrained=True):
        super().__init__()
        base_model = resnet34(pretrained=pretrained)
        
        # 提取多尺度特征
        self.conv1 = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu
        )
        self.conv2 = base_model.layer1  # 1/4
        self.conv3 = base_model.layer2  # 1/8
        self.conv4 = base_model.layer3  # 1/16
        
        # 通道调整
        self.reduce1 = nn.Conv2d(64, Config.FEAT_CHANNELS, 1)
        self.reduce2 = nn.Conv2d(64, Config.FEAT_CHANNELS, 1)
        self.reduce3 = nn.Conv2d(128, Config.FEAT_CHANNELS, 1)
        self.reduce4 = nn.Conv2d(256, Config.FEAT_CHANNELS, 1)
        
        self.conv1x1 = nn.Conv2d(1024, 256, kernel_size=1)
        
    def forward(self, x):
        x1 = self.conv1(x)       # 1/2
        x2 = self.conv2(x1)       # 1/4
        x3 = self.conv3(x2)       # 1/8
        x4 = self.conv4(x3)       # 1/16
        
        # 统一通道数并上采样到1/4分辨率
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        x3 = F.interpolate(self.reduce3(x3), scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.interpolate(self.reduce4(x4), scale_factor=4, mode='bilinear', align_corners=False)
        
        # 融合多尺度特征
        fused = torch.cat([x1, x2, x3, x4], dim=1)
        fused = self.conv1x1(fused)
        return fused

class MaskEncoder(nn.Module):
    """Mask特征编码器"""
    def __init__(self, in_ch=1, out_ch=Config.FEAT_CHANNELS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_ch, 1)
        )
    
    def forward(self, mask):
        return self.conv(mask)

class CrossViewTransformer(nn.Module):
    """跨视角注意力变换模块"""
    def __init__(self, in_ch):
        super().__init__()
        self.query = nn.Conv2d(in_ch, in_ch // Config.ATTENTION_RATIO, 1)
        self.key = nn.Conv2d(in_ch, in_ch // Config.ATTENTION_RATIO, 1)
        self.value = nn.Conv2d(in_ch, in_ch, 1)
        
    def forward(self, src_feat, tgt_feat):
        B, C, H, W = src_feat.shape
        
        # 计算注意力矩阵
        Q = self.query(src_feat).view(B, -1, H*W)  # [B, C', N]
        K = self.key(tgt_feat).view(B, -1, H*W)    # [B, C', M]
        attn = torch.softmax(Q.transpose(1, 2) @ K, dim=-1)  # [B, N, M]
        
        # 特征传递
        V = self.value(tgt_feat).view(B, -1, H*W)  # [B, C, M]
        trans_feat = attn @ V.transpose(1, 2)       # [B, N, C]
        trans_feat = trans_feat.transpose(1, 2).view(B, -1, H, W)
        
        return trans_feat + src_feat  # 残差连接

class Decoder(nn.Module):
    """Mask预测解码器"""
    def __init__(self, in_ch):
        super().__init__()
        self.up1 = self._upsample_block(in_ch, 128)
        self.up2 = self._upsample_block(128, 64)
        self.up3 = self._upsample_block(64, 32)
        self.conv_out = nn.Conv2d(32, 1, 1)
        
        # 可见性预测分支
        self.vis_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def _upsample_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        mask_logits = self.conv_out(x)
        
        # 可见性预测
        visibility = self.vis_head(x)
        return mask_logits, visibility

class CrossViewMaskTransformer(nn.Module):
    """整体模型架构"""
    def __init__(self):
        super().__init__()
        self.img_encoder = FeatureEncoder(Config.ENCODER, Config.PRETRAINED)
        self.mask_encoder = MaskEncoder()
        self.transformer = CrossViewTransformer(Config.FEAT_CHANNELS)
        self.decoder = Decoder(Config.FEAT_CHANNELS)
        
    def forward(self, src_img, tgt_img, src_mask):
        # 提取特征
        src_feat = self.img_encoder(src_img)       # [B, C, H, W]
        tgt_feat = self.img_encoder(tgt_img)       # [B, C, H, W]
        mask_feat = self.mask_encoder(src_mask)    # [B, C, H, W]
        
        # 融合源视角特征
        src_fused = src_feat + mask_feat
        
        # 跨视角变换
        trans_feat = self.transformer(src_fused, tgt_feat)
        
        # 解码预测
        pred_mask, visibility = self.decoder(trans_feat)
        return pred_mask, visibility