import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from dataset import (
    ISICDatasetV1,
    get_data_v1,
    val_augment_v1,
)
from engine import val_epoch
from models import ISICNetV1
from torch.utils.data import DataLoader
from utils import logger
from dataset import all_labels


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Evaluate ISIC2024 SCD")
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
        "--val_batch_size",
        type=int,
        default=64,
        help="Batch size for validation.",
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
        set_seed(args.seed, deterministic=True)

    (
        train_metadata, train_images,
        train_metadata_2020, train_images_2020,
        train_metadata_2019, train_images_2019
    ) = get_data_v1(
        args.data_dir,
        None,
        None,
    )

    fold_column = "fold"
    mean = None
    std = None
    val_index = train_metadata[train_metadata[fold_column] == args.fold].index
    val_metadata = train_metadata.loc[val_index, :].reset_index(drop=True)
    val_dataset = ISICDatasetV1(
        val_metadata,
        train_images,
        augment=val_augment_v1(args.image_size, mean=mean, std=std),
        infer=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    model = ISICNetV1(
        model_name=args.model_name,
        pretrained=True,
    )
    model = model.to(accelerator.device)
    criterion = nn.CrossEntropyLoss()

    (
        model,
        val_dataloader,
    ) = accelerator.prepare(
        model, val_dataloader,
    )
    output_dir = f"{args.model_dir}/models/fold_{args.fold}"
    accelerator.load_state(output_dir)
    start_time = time.time()
    logger.info(f"Fold {args.fold}")
    (
        val_loss,
        val_auc,
        val_pauc,
        multi_val_probs,
        val_probs,
        val_targets,
    ) = val_epoch(
        -1,
        model,
        criterion,
        val_dataloader,
        accelerator,
        args.n_tta,
    )
    logger.info(
        f"Fold: {args.fold} |"
        f" Val loss: {val_loss:.5f} | Val AUC: {val_auc:.5f} | Val pAUC: {val_pauc:.5f}"
    )
    oof_df = pd.DataFrame(
        {
            "isic_id": val_metadata["isic_id"],
            "patient_id": val_metadata["patient_id"],
            "fold": args.fold,
            "target": val_metadata["target"],
            f"oof_{args.model_name}_{args.version}": val_probs.flatten(),
        }
    )
    oof_multi_df = pd.DataFrame(multi_val_probs,
                                columns=[f"oof_{args.model_name}_{args.version}_{label}" for label in all_labels])
    oof_df = pd.concat([oof_df, oof_multi_df], axis=1)
    oof_df.to_csv(
        f"{args.model_dir}/oof_preds_{args.model_name}_{args.version}_fold_{args.fold}.csv",
        index=False,
    )
    logger.info(f"Time taken: {time.time() - start_time:.2f} s")
    logger.info(f"Finished evaluating fold {args.fold}")


if __name__ == "__main__":
    call_args = parse_args()
    main(call_args)
