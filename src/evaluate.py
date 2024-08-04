import argparse
import json
import logging
import time
import joblib
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
from dataset import ISICDataset, dev_augment, get_data, val_augment, fit_encoder_and_transform
from models import ISICNet
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import compute_auc, compute_pauc

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Evaluate ISIC2024 SCD")
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
    parser.add_argument(
        "--epoch", type=int, default=None, required=True, help="Epoch number"
    )
    parser.add_argument("--out_dim", type=int, default=9, help="Number of classes.")
    parser.add_argument("--image_size", type=int, default=224, help="Image size.")
    parser.add_argument("--use_meta", action="store_true", default=False)
    parser.add_argument("--n_meta_dim", type=str, default="512,128")
    parser.add_argument(
        "--val_batch_size", type=int, default=512, help="Batch size for validation."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader."
    )
    parser.add_argument(
        "--n_tta", type=int, default=6, help="Number of test time augmentations."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


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
    use_meta,
    log_interval=50,
):
    model.eval()
    val_probs = []
    val_targets = []
    val_loss = []
    total_steps = len(val_dataloader)
    with torch.no_grad():
        for step, (data, target) in enumerate(val_dataloader):
            if use_meta:
                image, meta = data
                logits = torch.zeros((image.shape[0], out_dim)).to(accelerator.device)
                probs = torch.zeros((image.shape[0], out_dim)).to(accelerator.device)
                for idx in range(n_tta):
                    logits_iter = model(get_trans(image, idx), meta)
                    logits += logits_iter
                    probs += logits_iter.softmax(1)
            else:
                image = data
                logits = torch.zeros((image.shape[0], out_dim)).to(accelerator.device)
                probs = torch.zeros((image.shape[0], out_dim)).to(accelerator.device)
                for idx in range(n_tta):
                    logits_iter = model(get_trans(image, idx))
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
        _,
        _,
        _,
        _,
        malignant_idx,
    ) = get_data(
        args.data_dir,
        None,
        None,
        args.out_dim,
        False,
        args.seed,
    )

    if args.use_meta:
        logger.info("Using meta features")
        (
            encoder,
            feature_cols,
            train_features,
            train_2020_features,
            train_2019_features
        ) = fit_encoder_and_transform(train_metadata, pd.DataFrame(), pd.DataFrame())
        train_metadata = pd.concat([train_metadata, train_features], axis=1)

        with open(f"{args.model_dir}/encoder.joblib", "wb") as f:
            joblib.dump(encoder, f)
    else:
        feature_cols = None

    val_index = train_metadata[train_metadata["fold"] == args.fold].index
    val_metadata = train_metadata.loc[val_index, :].reset_index(drop=True)
    val_dataset = ISICDataset(
        val_metadata, train_images, feature_cols=feature_cols, augment=val_augment(args.image_size)
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
        model_name=args.model_name,
        out_dim=args.out_dim,
        n_features=len(feature_cols) if feature_cols is not None else 0,
        n_meta_dim=(int(n) for n in args.n_meta_dim.split(",")),
        pretrained=True, infer=False
    )
    model = model.to(accelerator.device)
    criterion = nn.CrossEntropyLoss()
    model, val_dataloader = accelerator.prepare(model, val_dataloader)

    output_dir = f"{args.model_dir}/models/fold_{args.fold}"
    accelerator.load_state(output_dir)

    (
        val_loss,
        val_auc,
        val_pauc,
        val_probs,
        val_targets,
        binary_probs,
        binary_targets,
    ) = val_epoch(
        args.epoch,
        model,
        criterion,
        val_dataloader,
        accelerator,
        args.out_dim,
        args.n_tta,
        malignant_idx,
        args.use_meta,
    )
    logger.info(
        f"Fold: {args.fold} | Epoch: {args.epoch} |"
        f" Val loss: {val_loss:.5f}"
        f" Val AUC: {val_auc:.5f} | Val pAUC: {val_pauc:.5f}"
    )

    oof_df = pd.DataFrame(
        {
            "isic_id": val_metadata["isic_id"],
            "patient_id": val_metadata["patient_id"],
            "fold": args.fold,
            "label": val_metadata["label"],
            "target": val_metadata["target"],
            f"oof_{args.model_identifier}": binary_probs,
        }
    )
    oof_df.to_csv(
        f"{args.model_dir}/oof_preds_{args.model_identifier}_fold_{args.fold}.csv",
        index=False,
    )

    fold_metadata = {
        "fold": args.fold,
        "best_epoch": args.epoch,
        "best_val_auc": val_auc,
        "best_val_pauc": val_pauc,
        "best_val_loss": float(val_loss),
    }
    with open(f"{args.model_dir}/models/fold_{args.fold}/metadata.json", "w") as f:
        json.dump(fold_metadata, f)
    logger.info(f"Finished evaluating fold {args.fold}")


if __name__ == "__main__":
    call_args = parse_args()
    main(call_args)
