import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from dataset import (
    ISICDataset,
    dev_augment,
    get_data,
    val_augment,
)
from engine import train_epoch, val_epoch
from models import ISICNet
from libauc.sampler import DualSampler
from libauc.losses.auc import tpAUC_KL_Loss
from libauc.optimizers import SOTAs
from torch.utils.data import DataLoader
from safetensors import safe_open
from utils import logger


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Finetune ISIC2024 SCD")
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
        "--version",
        type=str,
        default=None,
        required=True,
        help="Version of the model",
    )
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        default=None,
        required=True,
        help="The directory where the pretrained model is stored.",
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
    parser.add_argument(
        "--sampling_rate", type=float, help="Sampling rate for fine-tuning."
    )
    parser.add_argument(
        "--tau", type=float, help="Margin for the loss function."
    )
    parser.add_argument(
        "--Lambda", type=float, help="Lambda for the loss function."
    )
    parser.add_argument(
        "--gamma0", type=float, help="Learning rate decay factor."
    )
    parser.add_argument(
        "--gamma1", type=float, help="Learning rate decay factor."
    )
    parser.add_argument(
        "--margin", type=float, help="Margin for the loss function."
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

    train_metadata, train_images = get_data(args.data_dir)

    if args.debug:
        args.num_epochs = 2
        dev_index = (
            train_metadata[train_metadata["fold"] != args.fold]
            .sample(args.train_batch_size * 100, random_state=args.seed)
            .index
        )
        val_index = (
            train_metadata[train_metadata["fold"] == args.fold]
            .sample(args.val_batch_size * 10, random_state=args.seed)
            .index
        )
    else:
        dev_index = train_metadata[train_metadata["fold"] != args.fold].index
        val_index = train_metadata[train_metadata["fold"] == args.fold].index

    dev_metadata = train_metadata.loc[dev_index, :].reset_index(drop=True)
    val_metadata = train_metadata.loc[val_index, :].reset_index(drop=True)

    pos_samples = dev_metadata[dev_metadata["target"] == 1]
    neg_strong_samples = dev_metadata[(dev_metadata["target"] == 0) & (dev_metadata["lesion_id"].notnull())]
    neg_weak_samples = dev_metadata[(dev_metadata["target"] == 0) & (dev_metadata["lesion_id"].isnull())]
    neg_weak_samples = neg_weak_samples.sample(n=pos_samples.shape[0]*100, random_state=args.seed)
    dev_metadata = pd.concat([pos_samples, neg_strong_samples, neg_weak_samples], axis=0).reset_index(drop=True)

    mean = None
    std = None

    dev_dataset = ISICDataset(
        dev_metadata,
        train_images,
        augment=dev_augment(args.image_size, mean=mean, std=std),
        infer=False,
    )
    val_dataset = ISICDataset(
        val_metadata,
        train_images,
        augment=val_augment(args.image_size, mean=mean, std=std),
        infer=False,
    )

    sampler = DualSampler(dataset=dev_dataset,
                          batch_size=args.train_batch_size,
                          sampling_rate=args.sampling_rate,
                          random_seed=args.seed)

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
    model = ISICNet(model_name=args.model_name)
    model = model.to(accelerator.device)

    tensors = {}
    with safe_open(f"{args.pretrained_model_dir}/models/fold_{args.fold}/model.safetensors", framework="pt") as f:
        for key in f.keys():
            if "classifier" not in key:
                tensors[key] = f.get_tensor(key)
    _ = model.load_state_dict(tensors, strict=False)
    logger.info("Pretrained weights loaded successfully")

    criterion = tpAUC_KL_Loss(data_len=len(dev_dataset),
                              tau=args.tau, Lambda=args.Lambda,
                              gammas=(args.gamma0, args.gamma1),
                              margin=args.margin)
    optimizer = SOTAs(model.parameters(), lr=args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        pct_start=1 / args.num_epochs,
        max_lr=args.init_lr * 1,
        div_factor=1,
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
    best_epoch = 0
    best_val_probs = None
    val_paucs = []
    val_aucs = []
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"Fold {args.fold} | Epoch {epoch}")
        start_time = time.time()
        lr = optimizer.param_groups[0]["lr"]
        _ = train_epoch(
            epoch,
            model,
            optimizer,
            criterion,
            dev_dataloader,
            lr_scheduler,
            accelerator,
        )
        (
            val_auc,
            val_pauc,
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
        val_paucs.append(val_pauc)
        val_aucs.append(val_auc)
        logger.info(
            f"Fold: {args.fold} | Epoch: {epoch} | LR: {lr:.7f} |"
            f" Val AUC: {val_auc:.5f} | Val pAUC: {val_pauc:.5f}"
        )
        if val_pauc > best_val_pauc:
            logger.info(
                f"pAUC: {best_val_pauc:.5f} --> {val_pauc:.5f}, saving model..."
            )
            best_val_pauc = val_pauc
            best_val_auc = val_auc
            best_epoch = epoch
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
        f" Best AUC: {best_val_auc} |"
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
    oof_df.to_csv(
        f"{args.model_dir}/oof_preds_{args.model_identifier}_fold_{args.fold}.csv",
        index=False,
    )

    fold_metadata = {
        "fold": args.fold,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_val_pauc": best_val_pauc,
        "val_paucs": val_paucs,
        "val_aucs": val_aucs,
    }
    with open(f"{args.model_dir}/models/fold_{args.fold}/metadata.json", "w") as f:
        json.dump(fold_metadata, f)
    logger.info(f"Finished training fold {args.fold}")


if __name__ == "__main__":
    call_args = parse_args()
    main(call_args)
