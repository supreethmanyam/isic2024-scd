import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from dataset import (
    ISICDatasetMulti,
    all_labels,
    dev_augment,
    get_data_multi,
    val_augment,
)
from engine import train_epoch, val_epoch
from models import ISICNetMulti
from torch.utils.data import DataLoader, RandomSampler
from utils import logger


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train ISIC2024 SCD")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        required=True,
        help="Model name for timm",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        required=True,
        help="Version of the model",
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
        "--fold_column", type=str, default=None, required=True, help="Fold column name."
    )
    parser.add_argument(
        "--fold", type=int, default=None, required=True, help="Fold number."
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
    parser.add_argument("--image_size", type=int, default=64, help="Image size.")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=64,
        help="Batch size for validation.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader."
    )
    parser.add_argument(
        "--init_lr",
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


def main(args):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
        set_seed(args.seed, deterministic=True)

    (
        train_metadata,
        train_images,
        train_metadata_2020,
        train_images_2020,
        train_metadata_2019,
        train_images_2019,
    ) = get_data_multi(args.data_dir, args.data_2020_dir, args.data_2019_dir)

    if args.debug:
        args.num_epochs = 2
        dev_index = (
            train_metadata[train_metadata[args.fold_column] != args.fold]
            .sample(args.train_batch_size * 3, random_state=args.seed)
            .index
        )
        val_index = (
            train_metadata[train_metadata[args.fold_column] == args.fold]
            .sample(args.val_batch_size * 10, random_state=args.seed)
            .index
        )
    else:
        dev_index = train_metadata[train_metadata[args.fold_column] != args.fold].index
        val_index = train_metadata[train_metadata[args.fold_column] == args.fold].index

    mean = None
    std = None

    dev_metadata = train_metadata.loc[dev_index, :].reset_index(drop=True)
    val_metadata = train_metadata.loc[val_index, :].reset_index(drop=True)

    dev_dataset = ISICDatasetMulti(
        dev_metadata,
        train_images,
        augment=dev_augment(args.image_size, mean=mean, std=std),
        infer=False,
    )
    val_dataset = ISICDatasetMulti(
        val_metadata,
        train_images,
        augment=val_augment(args.image_size, mean=mean, std=std),
        infer=False,
    )

    if not train_metadata_2020.empty:
        logger.info("Using 2020 data")
        if args.debug:
            train_metadata_2020 = train_metadata_2020.sample(
                args.train_batch_size * 1
            ).reset_index(drop=True)
        train_dataset_2020 = ISICDatasetMulti(
            train_metadata_2020,
            train_images_2020,
            augment=dev_augment(args.image_size),
            infer=False,
        )
        dev_dataset = torch.utils.data.ConcatDataset([dev_dataset, train_dataset_2020])
    if not train_metadata_2019.empty:
        logger.info("Using 2019 data")
        if args.debug:
            train_metadata_2019 = train_metadata_2019.sample(
                args.train_batch_size * 1
            ).reset_index(drop=True)
        train_dataset_2019 = ISICDatasetMulti(
            train_metadata_2019,
            train_images_2019,
            augment=dev_augment(args.image_size),
            infer=False,
        )
        dev_dataset = torch.utils.data.ConcatDataset([dev_dataset, train_dataset_2019])

    sampler = RandomSampler(dev_dataset)

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

    model = ISICNetMulti(
        model_name=args.model_name,
        pretrained=True,
    )
    model = model.to(accelerator.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        pct_start=1 / args.num_epochs,
        max_lr=args.init_lr * 10,
        div_factor=10,
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

    best_val_loss = 0
    best_val_auc = 0
    best_val_pauc = 0
    best_epoch = 0
    best_multi_val_probs = None
    best_val_probs = None
    train_losses = []
    val_losses = []
    val_paucs = []
    val_aucs = []
    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Fold {args.fold} | Epoch {epoch} | LR {lr:.7f}")
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
            multi_val_probs,
            val_probs,
            val_targets,
        ) = val_epoch(
            epoch,
            model,
            criterion,
            val_dataloader,
            accelerator,
            args.n_tta,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_paucs.append(val_pauc)
        val_aucs.append(val_auc)
        logger.info(
            f"Fold: {args.fold} | Epoch: {epoch} | LR: {lr:.7f} |"
            f" Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f} |"
            f" Val AUC: {val_auc:.5f} | Val pAUC: {val_pauc:.5f}"
        )
        if val_pauc > best_val_pauc:
            logger.info(
                f"pAUC: {best_val_pauc:.5f} --> {val_pauc:.5f}, saving model..."
            )
            best_val_pauc = val_pauc
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_epoch = epoch
            best_multi_val_probs = multi_val_probs
            best_val_probs = val_probs
            output_dir = f"{args.model_dir}/models/fold_{args.fold}"
            accelerator.save_state(output_dir)
        else:
            logger.info(
                f"pAUC: {best_val_pauc:.5f} --> {val_pauc:.5f}, skipping model save..."
            )
        elapsed_time = time.time() - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time % 60)
        logger.info(f"Epoch {epoch} took {elapsed_mins}m {elapsed_secs}s")

    output_dir = f"{args.model_dir}/models/fold_{args.fold}/final"
    accelerator.save_state(output_dir)

    logger.info(
        f"Fold: {args.fold} |"
        f" Best Val pAUC: {best_val_pauc} |"
        f" Best Val AUC: {best_val_auc} |"
        f" Best Val loss: {best_val_loss} |"
        f" Best epoch: {best_epoch}"
    )
    oof_df = pd.DataFrame(
        {
            "isic_id": val_metadata["isic_id"],
            "patient_id": val_metadata["patient_id"],
            "fold": args.fold,
            "target": val_metadata["target"],
            f"oof_{args.model_name}_{args.version}": best_val_probs.flatten(),
        }
    )
    oof_multi_df = pd.DataFrame(
        best_multi_val_probs,
        columns=[
            f"oof_{args.model_name}_{args.version}_{label}" for label in all_labels
        ],
    )
    oof_df = pd.concat([oof_df, oof_multi_df], axis=1)

    oof_df.to_csv(
        f"{args.model_dir}/oof_train_preds_{args.model_name}_{args.version}_fold_{args.fold}.csv",
        index=False,
    )

    fold_metadata = {
        "fold": args.fold,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_val_pauc": best_val_pauc,
        "best_val_loss": float(best_val_loss),
        "train_losses": np.array(train_losses, dtype=float).tolist(),
        "val_losses": np.array(val_losses, dtype=float).tolist(),
        "val_paucs": val_paucs,
        "val_aucs": val_aucs,
    }
    with open(f"{args.model_dir}/models/fold_{args.fold}/metadata.json", "w") as f:
        json.dump(fold_metadata, f)
    logger.info(f"Finished training fold {args.fold}")


if __name__ == "__main__":
    call_args = parse_args()
    main(call_args)
