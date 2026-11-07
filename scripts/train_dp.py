#!/usr/bin/env python3
"""
Data Parallel Training Script - Uses nn.DataParallel instead of DDP
This is a fallback for when NCCL fails with multi-GPU setups.
"""
import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import create_dataloaders
from src.losses import create_loss
from src.models import create_model
from src.utils import (
    AverageMeter,
    CheckpointManager,
    MetricLogger,
    SegmentationMetrics,
    setup_logger,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, metrics):
    model.train()
    metrics.reset()
    loss_meter = AverageMeter("Loss")

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            targets = torch.argmax(masks, dim=1) if masks.dim() == 4 else masks
            metrics.update(preds, targets)

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    epoch_metrics = metrics.compute()
    return loss_meter.avg, epoch_metrics


def validate(model, val_loader, criterion, device, epoch, logger, metrics):
    model.eval()
    metrics.reset()
    loss_meter = AverageMeter("Loss")

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks)

            loss_meter.update(loss.item(), images.size(0))

            preds = torch.argmax(outputs, dim=1)
            targets = torch.argmax(masks, dim=1) if masks.dim() == 4 else masks
            metrics.update(preds, targets)

            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    epoch_metrics = metrics.compute()
    return loss_meter.avg, epoch_metrics


def main(args):
    config = load_config(args.config)

    # Use all available GPUs
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0")

    set_seed(config["experiment"]["seed"])

    log_dir = config["paths"]["log_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['experiment']['name']}_dp_{timestamp}"

    logger = setup_logger("train_dp", log_dir, rank=0)

    metric_logger = MetricLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=config["logging"].get("tensorboard", True),
        use_wandb=False,
    )

    logger.info("=" * 60)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Using DataParallel with {num_gpus} GPUs")
    logger.info("=" * 60)

    # Create dataloaders (non-distributed)
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        config, distributed=False, world_size=1, rank=0
    )

    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    model_cfg = config["model"]
    model = create_model(
        architecture=model_cfg["architecture"],
        encoder=model_cfg["encoder"],
        encoder_weights=model_cfg["encoder_weights"],
        num_classes=model_cfg["num_classes"],
        # The losses take logits: an activation here would apply softmax twice and floor the loss
        activation=None,
    )

    # Wrap with DataParallel
    if num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using DataParallel on GPUs: {list(range(num_gpus))}")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model: {model_cfg['architecture']} with {model_cfg['encoder']} encoder")
    logger.info(f"Total parameters: {total_params:.2f}M")

    criterion = create_loss(config, device)
    logger.info(f"Loss function: {criterion.__name__}")

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )

    sched_cfg = config["scheduler"]
    if sched_cfg["type"] == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_cfg["factor"],
            patience=sched_cfg["patience"],
            min_lr=sched_cfg.get("min_lr", 1e-7),
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"], eta_min=sched_cfg.get("min_lr", 1e-7)
        )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config["paths"]["checkpoint_dir"],
        max_checkpoints=config["paths"].get("max_checkpoints", 5),
    )

    metrics = SegmentationMetrics(num_classes=model_cfg["num_classes"], device=device)

    best_val_loss = float("inf")
    best_iou = 0.0
    patience_counter = 0

    for epoch in range(train_cfg["epochs"]):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger, metrics
        )

        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, logger, metrics
        )

        current_lr = optimizer.param_groups[0]["lr"]

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        logger.info(
            f"Epoch {epoch}: Train Loss={train_loss:.5f}, Val Loss={val_loss:.5f}, "
            f"LR={current_lr:.2e}, F1={val_metrics['f1']:.4f}, IoU={val_metrics['iou']:.4f}"
        )

        metric_logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=current_lr,
            metrics=val_metrics,
        )

        is_best = val_metrics["iou"] > best_iou
        if is_best:
            best_iou = val_metrics["iou"]
            patience_counter = 0
        else:
            patience_counter += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Save checkpoint
        model_to_save = model.module if hasattr(model, "module") else model
        checkpoint_manager.save(
            model=model_to_save,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=val_loss,
            metrics={"val_loss": val_loss, "val_iou": val_metrics["iou"]},
            metric_for_best=val_metrics["iou"],
            is_distributed=False,
        )

        if patience_counter >= train_cfg.get("early_stopping_patience", 15):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info("=" * 60)
    logger.info(f"Training complete! Best IoU: {best_iou:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
