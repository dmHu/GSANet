import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from argparse import Namespace
from configs import Config
from dataset import build_dataloader
from model import CrossViewMaskTransformer
from losses import HybridMaskLoss, dice_coeff
from utils import setup_logging, save_checkpoint, load_checkpoint

def main():
    if len(Config.GPUS) > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        mp.spawn(train_worker, nprocs=len(Config.GPUS))
    else:
        train_worker(0)


def train_worker(rank):
    config_dict = Config.to_dict()
    cfg = Namespace(**config_dict)
    torch.cuda.set_device(rank)

    if len(cfg.GPUS) > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=len(cfg.GPUS),
            rank=rank
        )

    model = CrossViewMaskTransformer().cuda(rank)

    if len(cfg.GPUS) > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.LR_DECAY_STEP, gamma=cfg.LR_DECAY_GAMMA)
    criterion = HybridMaskLoss()

    train_loader, val_loader = build_dataloader(cfg)

    start_epoch = 0
    if os.path.exists(os.path.join(cfg.SAVE_DIR, "last.pth")):
        start_epoch = load_checkpoint(model, optimizer, cfg.SAVE_DIR, rank)

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        model.train()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, rank, epoch, cfg, is_train=True)

        model.eval()
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, None, rank, epoch, cfg, is_train=False)

        scheduler.step()

        if rank == 0 and (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, epoch, cfg.SAVE_DIR)

def run_epoch(model, loader, criterion, optimizer, gpu, epoch, cfg, is_train=True):
    metrics = {'loss': 0.0, 'bce': 0.0, 'dice': 0.0, 'vis': 0.0, 'dice_coeff': 0.0}
    num_batches = len(loader)
    start_time = time.time()

    for i, batch in enumerate(loader):
        src_img = batch['src_img'].cuda(gpu, non_blocking=True)
        tgt_img = batch['tgt_img'].cuda(gpu, non_blocking=True)
        src_mask = batch['src_mask'].cuda(gpu, non_blocking=True)
        tgt_mask = batch['tgt_mask'].cuda(gpu, non_blocking=True)
        visibility = batch['visibility'].cuda(gpu, non_blocking=True)

        pred_mask, pred_vis = model(src_img, tgt_img, src_mask)
        loss, loss_components = criterion((pred_mask, pred_vis), (tgt_mask, visibility))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dice = dice_coeff(pred_mask, tgt_mask)

        metrics['loss'] += loss_components['total']
        metrics['bce'] += loss_components['bce']
        metrics['dice'] += loss_components['dice']
        metrics['vis'] += loss_components['vis']
        metrics['dice_coeff'] += dice.item()

        if i % cfg.LOG_INTERVAL == 0 and gpu == 0:
            print(f"Epoch {epoch} | Batch {i}/{num_batches} | "
                  f"Loss: {loss.item():.4f} | Dice: {dice.item():.4f}")

    for k in metrics:
        metrics[k] /= num_batches

    if gpu == 0:
        mode = "Train" if is_train else "Val"
        print(f"{mode} Epoch {epoch} | Time: {time.time() - start_time:.1f}s | "
              f"Loss: {metrics['loss']:.4f} | Dice: {metrics['dice_coeff']:.4f}")

    return metrics

if __name__ == "__main__":
    config_dict = Config.to_dict()
    os.makedirs(config_dict["LOG_DIR"], exist_ok=True)
    os.makedirs(config_dict["SAVE_DIR"], exist_ok=True)
    main()
