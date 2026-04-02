import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridMaskLoss(nn.Module):
    """混合损失：BCE + Dice + 可见性损失"""
    def __init__(self, bce_weight=0.7, dice_weight=0.3, vis_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.vis_weight = vis_weight
        
    def forward(self, preds, targets):
        pred_mask, pred_vis = preds
        tgt_mask, tgt_vis = targets
        
        # Mask损失
        bce_loss = F.binary_cross_entropy_with_logits(pred_mask, tgt_mask)
        
        pred_sigmoid = torch.sigmoid(pred_mask)
        dice_loss = 1 - (2 * torch.sum(pred_sigmoid * tgt_mask) + 1e-8) / \
                    (torch.sum(pred_sigmoid) + torch.sum(tgt_mask) + 1e-8)
        
        # 可见性损失
        vis_loss = F.binary_cross_entropy(pred_vis.squeeze(), tgt_vis)
        
        # 加权组合
        total_loss = (self.bce_weight * bce_loss + 
                     self.dice_weight * dice_loss + 
                     self.vis_weight * vis_loss)
        
        return total_loss, {
            'bce': bce_loss.item(),
            'dice': dice_loss.item(),
            'vis': vis_loss.item(),
            'total': total_loss.item()
        }

def dice_coeff(pred, target):
    """计算Dice系数"""
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    intersection = torch.sum(pred_bin * target)
    return (2. * intersection + 1e-8) / (torch.sum(pred_bin) + torch.sum(target) + 1e-8)