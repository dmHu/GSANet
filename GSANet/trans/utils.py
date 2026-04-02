import os
import torch
import shutil
import numpy as np
from PIL import Image

def setup_logging(log_dir):
    """设置TensorBoard日志"""
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir)

def save_checkpoint(model, optimizer, epoch, save_dir):
    """保存模型检查点"""
    state = {
        'epoch': epoch,
        'state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, os.path.join(save_dir, f"epoch_{epoch}.pth"))
    torch.save(state, os.path.join(save_dir, "last.pth"))

def load_checkpoint(model, optimizer, save_dir, device):
    """加载模型检查点"""
    checkpoint = torch.load(os.path.join(save_dir, "last.pth"), map_location=f'cuda:{device}')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'] + 1

def visualize_results(src_img, tgt_img, src_mask, pred_mask, tgt_mask, save_path):
    """可视化结果对比"""
    # 转换Tensor到PIL
    def tensor_to_pil(tensor):
        return Image.fromarray((tensor.squeeze().cpu().numpy() * 255).astype(np.uint8))
    
    # 创建对比图像
    src_img_pil = tensor_to_pil(src_img[0])
    tgt_img_pil = tensor_to_pil(tgt_img[0])
    src_mask_pil = tensor_to_pil(src_mask[0])
    pred_mask_pil = tensor_to_pil(torch.sigmoid(pred_mask[0]) > 0.5)
    tgt_mask_pil = tensor_to_pil(tgt_mask[0])
    
    # 组合图像
    width, height = src_img_pil.size
    result = Image.new('RGB', (width * 5, height))
    result.paste(src_img_pil, (0, 0))
    result.paste(src_mask_pil, (width, 0))
    result.paste(tgt_img_pil, (width * 2, 0))
    result.paste(pred_mask_pil, (width * 3, 0))
    result.paste(tgt_mask_pil, (width * 4, 0))
    
    result.save(save_path)