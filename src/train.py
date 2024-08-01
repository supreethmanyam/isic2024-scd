import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from dataset import ISICDataset, dev_augment, get_data, val_augment
from models import ISICNet
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import compute_auc, compute_pauc

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train ISIC2024 SCD")
    parser.add_argument(
        "--model_identifier",
        type=str,
        default=None,
        required=True,
        help="Model identifier for the run",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        required=True,
        help="Model name for timm",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="The directory where the data is stored.",
    )
    parser.add_argument(
        "--data_2020_dir",
        type=str,
        default=None,
        help="The directory where the 2020 data is stored.",
    )
    parser.add_argument(
        "--data_2019_dir",
        type=str,
        default=None,
        help="The directory where the 2019 data is stored.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        required=True,
        help="The directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="The directory where the logs will be written.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--fold", type=int, default=None, required=True, help="Fold number."
    )
    parser.add_argument("--out_dim", type=int, default=9, help="Number of classes.")
    parser.add_argument("--image_size", type=int, default=224, help="Image size.")
    parser.add_argument(
        "--train_batch_size", type=int, default=256, help="Batch size for training."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=512, help="Batch size for validation."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        required=True,
        help="Learning rate.",
    )
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs.")
    parser.add_argument(
        "--n_tta", type=int, default=6, help="Number of test time augmentations."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Use a small subset of the data for debugging.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def train_epoch(
    epoch,
    model,
    optimizer,
    criterion,
    dev_dataloader,
    lr_scheduler,
    accelerator,
    log_interval=50,
):
    model.train()
    train_loss = []
    total_steps = len(dev_dataloader)
    for step, (data, target) in enumerate(dev_dataloader):
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        loss_value = accelerator.gather(loss).item()
        train_loss.append(loss_value)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        if step % log_interval == 0:
            logger.info(
                f"Epoch: {epoch} | Step: {step + 1}/{total_steps} |"
                f" Loss: {loss_value:.5f} | Smooth loss: {smooth_loss:.5f}"
            )
    train_loss = np.mean(train_loss)
    return train_loss


def get_trans(img, iteration):
    if iteration >= 6:
        img = img.transpose(2, 3)
    if iteration % 6 == 0:
        return img
    elif iteration % 6 == 1:
        return torch.flip(img, dims=[2])
    elif iteration % 6 == 2:
        return torch.flip(img, dims=[3])
    elif iteration % 6 == 3:
        return torch.rot90(img, 1, dims=[2, 3])
    elif iteration % 6 == 4:
        return torch.rot90(img, 2, dims=[2, 3])
    elif iteration % 6 == 5:
        return torch.rot90(img, 3, dims=[2, 3])


def val_epoch(
    epoch,
    model,
    criterion,
    val_dataloader,
    accelerator,
    out_dim,
    n_tta,
    malignant_idx,
    log_interval=50,
):
    model.eval()
    val_probs = []
    val_targets = []
    val_loss = []
    total_steps = len(val_dataloader)
    with torch.no_grad():
        for step, (data, target) in enumerate(val_dataloader):
            logits = torch.zeros((data.shape[0], out_dim)).to(accelerator.device)
            probs = torch.zeros((data.shape[0], out_dim)).to(accelerator.device)
            for idx in range(n_tta):
                logits_iter = model(get_trans(data, idx))
                logits += logits_iter
                probs += logits_iter.softmax(1)
            logits /= n_tta
            probs /= n_tta

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

            logits, probs, targets = accelerator.gather((logits, probs, target))
            val_probs.append(probs)
            val_targets.append(targets)

            if step % log_interval == 0:
                logger.info(f"Epoch: {epoch} | Step: {step + 1}/{total_steps}")

    val_loss = np.mean(val_loss)
    val_probs = torch.cat(val_probs).cpu().numpy()
    val_targets = torch.cat(val_targets).cpu().numpy()
    if out_dim == 9:
        binary_probs = val_probs[:, malignant_idx].sum(1)
        binary_targets = (
            (val_targets == malignant_idx[0])
            | (val_targets == malignant_idx[1])
            | (val_targets == malignant_idx[2])
        )

        val_auc = compute_auc(binary_targets, binary_probs)
        val_pauc = compute_pauc(binary_targets, binary_probs, min_tpr=0.8)
    else:
        binary_probs = val_probs[:, 1]
        binary_targets = val_targets

        val_auc = compute_auc(binary_targets, binary_probs)
        val_pauc = compute_pauc(binary_targets, binary_probs, min_tpr=0.8)
    return (
        val_loss,
        val_auc,
        val_pauc,
        val_probs,
        val_targets,
        binary_probs,
        binary_targets,
    )


def main(args):
    logging_dir = Path(args.model_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.model_dir, logging_dir=str(logging_dir)
    )
    kwargs = DistributedDataParallelKwargs()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    (
        train_metadata,
        train_images,
        train_metadata_2020,
        train_images_2020,
        train_metadata_2019,
        train_images_2019,
        malignant_idx,
    ) = get_data(
        args.data_dir,
        args.data_2020_dir,
        args.data_2019_dir,
        args.out_dim,
        args.debug,
        args.seed,
    )

    dev_index = train_metadata[train_metadata["fold"] != args.fold].index
    val_index = train_metadata[train_metadata["fold"] == args.fold].index

    dev_metadata = train_metadata.loc[dev_index, :].reset_index(drop=True)
    val_metadata = train_metadata.loc[val_index, :].reset_index(drop=True)

    if "sample_weight" not in dev_metadata.columns:
        dev_metadata["sample_weight"] = 1
    sample_weight = dev_metadata["sample_weight"].values.tolist()

    dev_dataset = ISICDataset(
        dev_metadata, train_images, augment=dev_augment(args.image_size)
    )
    val_dataset = ISICDataset(
        val_metadata, train_images, augment=val_augment(args.image_size)
    )

    if not train_metadata_2020.empty:
        logger.info("Using 2020 data")
        if "sample_weight" not in train_metadata_2020.columns:
            train_metadata_2020["sample_weight"] = 1
        sample_weight += train_metadata_2020["sample_weight"].values.tolist()
        train_dataset_2020 = ISICDataset(
            train_metadata_2020, train_images_2020, augment=dev_augment(args.image_size)
        )
        dev_dataset = torch.utils.data.ConcatDataset([dev_dataset, train_dataset_2020])
    if not train_metadata_2019.empty:
        logger.info("Using 2019 data")
        if "sample_weight" not in train_metadata_2019.columns:
            train_metadata_2019["sample_weight"] = 1
        sample_weight += train_metadata_2019["sample_weight"].values.tolist()
        train_dataset_2019 = ISICDataset(
            train_metadata_2019, train_images_2019, augment=dev_augment(args.image_size)
        )
        dev_dataset = torch.utils.data.ConcatDataset([dev_dataset, train_dataset_2019])

    if np.unique(sample_weight).size > 1:
        logger.info("Using Weighted sampler")
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight), replacement=True)
    logger.info(f"Building a model with {args.out_dim} classes")

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    model = ISICNet(
        model_name=args.model_name, out_dim=args.out_dim, pretrained=True, infer=False
    )
    model = model.to(accelerator.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate / 20)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.num_epochs,
        steps_per_epoch=len(dev_dataloader),
    )

    (
        model,
        optimizer,
        dev_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, dev_dataloader, val_dataloader, lr_scheduler
    )

    best_val_auc = 0
    best_val_pauc = 0
    best_val_loss = 0
    best_epoch = 0
    best_val_probs = None

    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"Fold {args.fold} | Epoch {epoch}")
        start_time = time.time()

        train_loss = train_epoch(
            epoch,
            model,
            optimizer,
            criterion,
            dev_dataloader,
            lr_scheduler,
            accelerator,
        )
        (
            val_loss,
            val_auc,
            val_pauc,
            val_probs,
            val_targets,
            binary_probs,
            binary_targets,
        ) = val_epoch(
            epoch,
            model,
            criterion,
            val_dataloader,
            accelerator,
            args.out_dim,
            args.n_tta,
            malignant_idx,
        )

        if val_pauc > best_val_pauc:
            logger.info(
                f"pAUC: {best_val_pauc:.5f} --> {val_pauc:.5f}, saving model..."
            )
            best_val_pauc = val_pauc
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_probs = binary_probs
            output_dir = f"{args.model_dir}/models/fold_{args.fold}"
            accelerator.save_state(output_dir)
        else:
            logger.info(
                f"pAUC: {best_val_pauc:.5f} --> {val_pauc:.5f}, skipping model save..."
            )
        logger.info(
            f"Fold: {args.fold} | Epoch: {epoch} |"
            f" Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f}"
            f" Val AUC: {val_auc:.5f} | Val pAUC: {val_pauc:.5f}"
        )
        elapsed_time = time.time() - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time % 60)
        logger.info(f"Epoch {epoch} took {elapsed_mins}m {elapsed_secs}s")

    logger.info(
        f"Fold: {args.fold} | "
        f"Best Val pAUC: {best_val_pauc} | Best AUC: {best_val_auc} |"
        f" Best loss: {best_val_loss} |"
        f" Best epoch: {best_epoch}"
    )
    oof_df = pd.DataFrame(
        {
            "isic_id": val_metadata["isic_id"],
            "patient_id": val_metadata["patient_id"],
            "fold": args.fold,
            "label": val_metadata["label"],
            "target": val_metadata["target"],
            f"oof_{args.model_identifier}": best_val_probs,
        }
    )
    oof_df.to_csv(
        f"{args.model_dir}/oof_preds_{args.model_identifier}_fold_{args.fold}.csv",
        index=False,
    )

    fold_metadata = {
        "fold": args.fold,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_val_pauc": best_val_pauc,
        "best_val_loss": float(best_val_loss),
    }
    with open(f"{args.model_dir}/models/fold_{args.fold}/metadata.json", "w") as f:
        json.dump(fold_metadata, f)
    logger.info(f"Finished training fold {args.fold}")


if __name__ == "__main__":
    call_args = parse_args()
    main(call_args)
