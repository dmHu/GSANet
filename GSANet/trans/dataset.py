import os
import random
from glob import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class MultiViewMaskDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode
        self.samples = self._collect_samples()
        
        # 数据增强
        self.transform = self._build_transform()
        self.mask_transform = T.Compose([
            T.Resize(cfg.IMG_SIZE, Image.NEAREST),
            T.ToTensor()
        ])
        
    def _collect_samples(self):
        """收集所有有效样本：至少两个视角有mask的样本"""
        samples = []
        pattern = os.path.join(self.cfg.DATA_ROOT, "NG", "*", "S*")
        
        for sample_dir in glob(pattern):
            # 收集所有视角的mask路径
            mask_paths = {}
            for view in self.cfg.VIEW_NAMES:
                mask_files = glob(os.path.join(sample_dir, f"*{view}*{self.cfg.MASK_SUFFIX}"))
                if mask_files:
                    mask_paths[view] = mask_files[0]
            
            # 至少需要两个视角有mask
            if len(mask_paths) >= 2:
                # 收集对应的图像路径
                img_paths = {}
                for view in mask_paths.keys():
                    img_files = glob(os.path.join(sample_dir, f"*{view}*.jpg"))
                    if img_files:
                        img_paths[view] = img_files[0]
                
                if len(img_paths) == len(mask_paths):
                    samples.append({
                        'dir': sample_dir,
                        'masks': mask_paths,
                        'images': img_paths
                    })
        
        # 划分训练/验证集 (8:2)
        random.shuffle(samples)
        split_idx = int(0.8 * len(samples))
        self.samples = samples[:split_idx] if self.mode == 'train' else samples[split_idx:]
        
        print(f"Loaded {len(self.samples)} {self.mode} samples")
        return samples
    
    def _build_transform(self):
        """构建数据增强变换"""
        transforms = [
            T.Resize(self.cfg.IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if self.mode == 'train':
            transforms.insert(0, T.RandomAffine(
                degrees=self.cfg.AUG_ROTATION,
                scale=self.cfg.AUG_SCALE
            ))
            transforms.insert(0, T.ColorJitter(
                brightness=self.cfg.AUG_BRIGHTNESS,
                contrast=self.cfg.AUG_CONTRAST
            ))
        
        return T.Compose(transforms)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        views = list(sample['images'].keys())
        
        # 随机选择两个视角
        src_view, tgt_view = random.sample(views, 2)
        
        # 加载源视角图像和mask
        src_img = Image.open(sample['images'][src_view]).convert('RGB')
        src_mask = Image.open(sample['masks'][src_view]).convert('L')
        
        # 加载目标视角图像和mask
        tgt_img = Image.open(sample['images'][tgt_view]).convert('RGB')
        tgt_mask = Image.open(sample['masks'][tgt_view]).convert('L')
        
        # 应用变换
        src_img_t = self.transform(src_img)
        tgt_img_t = self.transform(tgt_img)
        src_mask_t = self.mask_transform(src_mask)
        tgt_mask_t = self.mask_transform(tgt_mask)
        
        # 构建可见性标签 (1=可见, 0=不可见)
        visibility = torch.tensor(1.0)  # 当前样本中两个视角都可见
        
        return {
            'src_img': src_img_t,
            'tgt_img': tgt_img_t,
            'src_mask': src_mask_t,
            'tgt_mask': tgt_mask_t,
            'visibility': visibility,
            'src_view': src_view,
            'tgt_view': tgt_view
        }

def build_dataloader(cfg):
    train_set = MultiViewMaskDataset(cfg, 'train')
    val_set = MultiViewMaskDataset(cfg, 'val')
    
    train_loader = DataLoader(
        train_set, batch_size=cfg.BATCH_SIZE, 
        shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set, batch_size=cfg.BATCH_SIZE, 
        shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    
    return train_loader, val_loader