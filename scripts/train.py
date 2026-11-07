#!/usr/bin/env python3
"""
Multi-GPU Distributed Training Script for Semantic Segmentation

Usage:
    Single GPU:
        python train.py --config configs/default.yaml

    Multi-GPU (4 GPUs):
        torchrun --nproc_per_node=4 train.py --config configs/default.yaml

    Multi-GPU with specific GPUs:
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
            --config configs/default.yaml
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
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
    cleanup_distributed,
    is_main_process,
    load_checkpoint,
    reduce_tensor,
    setup_distributed,
    setup_logger,
)
from src.utils.config_validator import expand_environment_variables, validate_config
from src.utils.memory_tracker import MemoryTracker
from src.utils.profiler import TrainingProfiler


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file with validation."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Expand environment variables
    config = expand_environment_variables(config)

    # Validate configuration
    validate_config(config)

    return config


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int,
    logger,
    metrics: SegmentationMetrics,
    distributed: bool = False,
    world_size: int = 1,
    scaler: GradScaler = None,
    gradient_accumulation_steps: int = 1,
) -> tuple:
    """
    Train for one epoch with mixed precision and gradient accumulation support.

    Args:
        gradient_accumulation_steps: Number of batches to accumulate before updating
        scaler: GradScaler for mixed precision training

    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.train()
    metrics.reset()
    loss_meter = AverageMeter("Loss")

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", disable=not is_main_process())

    use_amp = scaler is not None
    optimizer.zero_grad()

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Forward pass with mixed precision
        with autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # Update metrics (use unscaled loss for logging)
        with torch.no_grad():
            actual_loss = loss * gradient_accumulation_steps
            if distributed:
                reduced_loss = reduce_tensor(actual_loss, world_size)
            else:
                reduced_loss = actual_loss

            loss_meter.update(reduced_loss.item(), images.size(0))
            # Metrics now handle .detach().cpu() internally for memory efficiency
            metrics.update(outputs, masks)

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    # Handle remaining gradients
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    epoch_metrics = metrics.compute()
    return loss_meter.avg, epoch_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device: torch.device,
    epoch: int,
    metrics: SegmentationMetrics,
    distributed: bool = False,
    world_size: int = 1,
    use_amp: bool = False,
) -> tuple:
    """
    Validate the model with mixed precision support.

    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    metrics.reset()
    loss_meter = AverageMeter("Loss")

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", disable=not is_main_process())

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)

        if distributed:
            reduced_loss = reduce_tensor(loss, world_size)
        else:
            reduced_loss = loss

        loss_meter.update(reduced_loss.item(), images.size(0))
        # Metrics now handle .detach().cpu() internally for memory efficiency
        metrics.update(outputs, masks)

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    epoch_metrics = metrics.compute()
    return loss_meter.avg, epoch_metrics


def emit_event(path, **event):
    """Append one JSON event for the desktop app to follow. No-op without --events."""
    if not path:
        return
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


def main(args):
    """Main training function."""

    # Load configuration
    config = load_config(args.config)

    # Setup distributed training
    distributed = False
    world_size = 1
    rank = 0

    if "WORLD_SIZE" in os.environ:
        distributed = True
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["LOCAL_RANK"])

        # Set environment variables for NCCL without P2P
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_P2P_LEVEL", "LOC")

        # Try to use requested backend, fallback to gloo if NCCL fails
        backend = config["distributed"].get("backend", "nccl")
        try:
            setup_distributed(rank, world_size, backend)
        except Exception as e:
            if backend == "nccl":
                if rank == 0:
                    print(f"NCCL failed: {e}, falling back to Gloo backend")
                backend = "gloo"
                setup_distributed(rank, world_size, backend)
            else:
                raise

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Set seed
    set_seed(config["experiment"]["seed"] + rank)

    # Setup logging (only on main process)
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['experiment']['name']}_{timestamp}"
    experiment_dir = os.path.join(checkpoint_dir, experiment_name)

    # Create experiment directory for all outputs
    if is_main_process(rank):
        os.makedirs(experiment_dir, exist_ok=True)

    # Setup logger - save to both log_dir and experiment_dir
    logger = setup_logger("train", experiment_dir, rank=rank)
    metric_logger = None

    if is_main_process(rank):
        metric_logger = MetricLogger(
            log_dir=experiment_dir,  # Save plots to experiment dir
            experiment_name=experiment_name,
            use_tensorboard=config["logging"].get("tensorboard", True),
            use_wandb=config["logging"].get("wandb", False),
            wandb_project=config["logging"].get("wandb_project"),
        )

        logger.info("=" * 60)
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Output directory: {experiment_dir}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Distributed: {distributed} (world_size={world_size})")
        if distributed:
            logger.info(f"Backend: {backend}")
            logger.info("NCCL P2P disabled (optimized for PCIe topology)")
        logger.info(f"Device: {device}")
        logger.info("=" * 60)

    # Create dataloaders
    train_loader, val_loader, test_loader, train_sampler = create_dataloaders(
        config, distributed, world_size, rank
    )

    if is_main_process(rank):
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
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
    )
    model = model.to(device)

    if is_main_process(rank):
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Model: {model_cfg['architecture']} with {model_cfg['encoder']} encoder")
        logger.info(f"Total parameters: {total_params:.2f}M")

    # Wrap model for distributed training
    if distributed:
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=config["distributed"].get("find_unused_parameters", False),
        )
        # Removed unnecessary barrier - DDP initialization handles sync

    # Create loss function (weights handled inside forward pass)
    criterion = create_loss(config, device)

    if is_main_process(rank):
        logger.info(f"Loss function: {criterion.__name__}")

    # Create optimizer
    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )

    # Setup mixed precision training
    use_amp = train_cfg.get("mixed_precision", True)
    scaler = GradScaler("cuda") if use_amp else None

    # Setup gradient accumulation
    gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)

    if is_main_process(rank):
        logger.info(f"Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")
        logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
        if gradient_accumulation_steps > 1:
            effective_batch = train_cfg["batch_size"] * world_size * gradient_accumulation_steps
            logger.info(f"Effective batch size: {effective_batch}")

    # Create scheduler
    sched_cfg = config["scheduler"]
    if sched_cfg["type"] == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_cfg["factor"],
            patience=sched_cfg["patience"],
            min_lr=sched_cfg.get("min_lr", 1e-7),
        )
    elif sched_cfg["type"] == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"], eta_min=sched_cfg.get("min_lr", 1e-7)
        )
    else:
        scheduler = None

    # Create metrics (compute on CPU by default for GPU memory savings)
    metrics = SegmentationMetrics(
        num_classes=config["data"]["num_classes"],
        device=device,
        compute_on_cpu=train_cfg.get("metrics_on_cpu", True),
    )

    # Create checkpoint manager - use experiment_dir defined earlier
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=experiment_dir,
        max_checkpoints=config["logging"].get("save_top_k", 3),
        mode="min",
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        checkpoint_info = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = checkpoint_info["epoch"] + 1
        best_loss = checkpoint_info["loss"]
        if is_main_process(rank):
            logger.info(f"Resumed from epoch {start_epoch}, loss {best_loss:.5f}")

    # Early stopping
    early_stop_cfg = config["early_stopping"]
    epochs_no_improve = 0

    # Initialize memory tracker and profiler
    memory_tracker = MemoryTracker(
        enabled=train_cfg.get("track_memory", True) and torch.cuda.is_available(), device_id=rank
    )

    profiler = TrainingProfiler(
        enabled=train_cfg.get("enable_profiling", False),
        output_dir=os.path.join(experiment_dir, "profiler"),
        wait_steps=1,
        warmup_steps=2,
        active_steps=3,
    )

    # Training loop
    if is_main_process(rank):
        logger.info("Starting training...")
        if train_cfg.get("enable_profiling", False):
            logger.info("Profiling enabled - will profile first few epochs")

        profiler.start()

    for epoch in range(start_epoch, train_cfg["epochs"]):
        # Set epoch for distributed sampler
        if distributed:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            logger,
            metrics,
            distributed,
            world_size,
            scaler,
            gradient_accumulation_steps,
        )

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, metrics, distributed, world_size, use_amp
        )

        # Update scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Logging (main process only)
        if is_main_process(rank):
            # Get memory stats
            mem_stats_str = memory_tracker.get_formatted_stats()

            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_loss:.5f}, Val Loss={val_loss:.5f}, "
                f"LR={current_lr:.2e}, "
                f"F1={val_metrics['f1']:.4f}, IoU={val_metrics['iou']:.4f}"
            )
            logger.info(f"  {mem_stats_str}")
            emit_event(
                args.events,
                type="epoch",
                epoch=epoch,
                epochs=train_cfg["epochs"],
                train_loss=train_loss,
                val_loss=val_loss,
                lr=current_lr,
                dir=experiment_dir,
                **val_metrics,
            )

            if metric_logger:
                metric_logger.log_epoch(epoch, train_loss, val_loss, current_lr, val_metrics)

            # Save checkpoint
            improved = val_loss < best_loss
            if improved:
                best_loss = val_loss
                epochs_no_improve = 0

                # Save best model
                best_path = os.path.join(experiment_dir, "best_model.pth")
                if distributed:
                    torch.save(model.module.state_dict(), best_path)
                else:
                    torch.save(model.state_dict(), best_path)

                logger.info(f"New best model saved! Val Loss: {best_loss:.5f}")
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement ({epochs_no_improve}/{early_stop_cfg['patience']})")

            # Save checkpoint to manager
            checkpoint_manager.save(
                model, optimizer, scheduler, epoch, val_loss, val_metrics, val_loss, distributed
            )

        # Early stopping check - use all_reduce instead of broadcast for efficiency
        if distributed:
            stop_tensor = torch.tensor([float(epochs_no_improve)], device=device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            epochs_no_improve = int(stop_tensor.item())

        if epochs_no_improve >= early_stop_cfg["patience"]:
            if is_main_process(rank):
                logger.info(f"Early stopping at epoch {epoch}")
            break

        # Profiler step (if enabled)
        if is_main_process(rank):
            profiler.step()

    # Stop profiler
    if is_main_process(rank):
        profiler.stop()

    # Save final training history
    if is_main_process(rank) and metric_logger:
        metric_logger.save_history()
        metric_logger.close()
        logger.info(f"Training complete. Best Val Loss: {best_loss:.5f}")
        logger.info(f"Final GPU Memory Stats: {memory_tracker.get_formatted_stats()}")

    if is_main_process(rank):
        emit_event(args.events, type="done", dir=experiment_dir, best_val_loss=best_loss)

    # Cleanup
    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train semantic segmentation model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--events", type=str, default=None, help="JSONL file for per-epoch progress events"
    )

    args = parser.parse_args()
    main(args)
