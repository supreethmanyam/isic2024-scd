import argparse
import json
import logging
from io import BytesIO
from pathlib import Path

import albumentations as A
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import roc_auc_score as compute_auc
from timm import create_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = get_logger(__name__)


def dev_augment(image_size):
    transform = A.Compose(
        [
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.75
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(5, 7)),
                    A.MedianBlur(blur_limit=(5, 7)),
                    A.GaussianBlur(blur_limit=(5, 7)),
                    A.GaussNoise(var_limit=(5.0, 30.0)),
                ],
                p=0.7,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=1.0),
                    A.ElasticTransform(alpha=3),
                ],
                p=0.7,
            ),
            A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85
            ),
            A.Resize(image_size, image_size),
            # A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
            ToTensorV2(),
        ],
        p=1.0,
    )
    return transform


def val_augment(image_size):
    transform = A.Compose([A.Resize(image_size, image_size), ToTensorV2()], p=1.0)
    return transform


class ISICDataset(Dataset):
    def __init__(self, metadata, images, augment, infer=False):
        self.metadata = metadata
        self.images = images
        self.augment = augment
        self.length = len(self.metadata)
        self.infer = infer

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.metadata.iloc[index]

        image = np.array(Image.open(BytesIO(self.images[data["isic_id"]][()])))
        image = self.augment(image=image)["image"]

        record = {"image": image}

        if not self.infer:
            target = data["target"]
            record["target"] = torch.tensor(target).float()

        return record


class ISICNet(nn.Module):
    def __init__(self, model_name, pretrained=True, infer=False):
        super(ISICNet, self).__init__()
        self.infer = infer
        self.model = create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="",
        )
        self.classifier = nn.Linear(self.model.num_features, 1)

        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for i in range(5)])

    def forward(self, batch):
        image = batch["image"]
        image = image.float() / 255

        x = self.model(image)
        bs = len(image)
        pool = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        if self.training:
            logit = 0
            for i in range(len(self.dropouts)):
                logit += self.classifier(self.dropouts[i](pool))
            logit = logit / len(self.dropouts)
        else:
            logit = self.classifier(pool)
        return logit


def compute_pauc(y_true, y_pred, min_tpr: float = 0.80) -> float:
    """
    2024 ISIC Challenge metric: pAUC

    Given a solution file and submission file, this function returns the
    partial area under the receiver operating characteristic (pAUC)
    above a given true positive rate (TPR) = 0.80.
    https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.

    (c) 2024 Nicholas R Kurtansky, MSKCC

    Args:
        min_tpr:
        y_true: ground truth of 1s and 0s
        y_pred: predictions of scores ranging [0, 1]

    Returns:
        Float value range [0, max_fpr]
    """

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(y_true - 1)

    # flip the submissions to their compliments
    v_pred = -1.0 * y_pred

    max_fpr = abs(1 - min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    #     # Equivalent code that uses sklearn's roc_auc_score
    #     v_gt = abs(np.asarray(solution.values)-1)
    #     v_pred = np.array([1.0 - x for x in submission.values])
    #     max_fpr = abs(1-min_tpr)
    #     partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    #     # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    #     # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    #     partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return partial_auc


def make_over_sample(train_index, target, pos_ratio):
    target_series = pd.Series(target, index=train_index)

    # Separate positive and negative indices
    pos_indices = target_series[target_series == 1].index
    neg_indices = target_series[target_series == 0].index

    # Calculate the number of positive samples needed
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    n_total = n_pos + n_neg
    n_desired_pos = int((pos_ratio * n_total) / (1 - pos_ratio))

    # over_sample positive indices
    if n_desired_pos > n_pos:
        pos_indices_oversampled = np.random.choice(
            pos_indices, size=n_desired_pos, replace=True
        )
    else:
        pos_indices_oversampled = pos_indices

    ned_indices_under_sampled = np.unique(
        np.random.choice(neg_indices, size=n_desired_pos * 3, replace=True)
    )

    # Combine with negative indices
    oversampled_indices = np.concatenate(
        [ned_indices_under_sampled, pos_indices_oversampled]
    )
    np.random.shuffle(oversampled_indices)
    return oversampled_indices


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
        "--pos_ratio", type=float, default=0.1, help="Positive ratio for oversampling."
    )
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
        "--tta", action="store_true", help="Use test time augmentation."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Use a small subset of the data for debugging."
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

    train_metadata = pd.read_csv(
        f"{args.data_dir}/train-metadata.csv", low_memory=False
    )
    train_images = h5py.File(f"{args.data_dir}/train-image.hdf5", mode="r")

    folds_df = pd.read_csv(f"{args.data_dir}/folds.csv")
    train_metadata = train_metadata.merge(
        folds_df, on=["isic_id", "patient_id"], how="inner"
    )
    if args.debug:
        train_metadata = train_metadata.sample(
            frac=0.05, random_state=args.seed
        ).reset_index(drop=True)

    y_train = train_metadata["target"]

    dev_index = train_metadata[train_metadata["fold"] != args.fold].index
    val_index = train_metadata[train_metadata["fold"] == args.fold].index

    oversampled_dev_index = make_over_sample(dev_index, y_train, args.pos_ratio)

    dev_metadata = train_metadata.loc[oversampled_dev_index, :].reset_index(drop=True)
    val_metadata = train_metadata.loc[val_index, :].reset_index(drop=True)

    dev_dataset = ISICDataset(
        dev_metadata, train_images, augment=dev_augment(args.image_size)
    )
    val_dataset = ISICDataset(
        val_metadata, train_images, augment=val_augment(args.image_size)
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
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

    model = ISICNet(model_name=args.model_name, pretrained=True, infer=False)
    model = model.to(accelerator.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate / 10)
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

    best_pauc = 0
    best_auc = 0
    best_epoch = 0
    best_val_preds = None

    for epoch in range(args.num_epochs):
        model.train()
        for batch in tqdm(dev_dataloader, total=len(dev_dataloader)):
            optimizer.zero_grad()
            output = model(batch)
            loss = F.binary_cross_entropy_with_logits(
                output, batch["target"].unsqueeze(1)
            )
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, total=len(val_dataloader)):
                image0 = batch["image"].clone().detach()
                val_preds_batch = 0
                counter = 0
                with torch.no_grad():
                    outputs = model(batch)
                preds = torch.sigmoid(outputs)
                val_targets_batch = batch["target"]
                preds, val_targets_batch = accelerator.gather_for_metrics(
                    (preds, val_targets_batch)
                )
                val_preds_batch += preds.data.cpu().numpy().reshape(-1)
                counter += 1
                if args.tta:
                    batch["image"] = torch.flip(image0, dims=[2])
                    with torch.no_grad():
                        outputs = model(batch)
                    preds = torch.sigmoid(outputs)
                    preds = accelerator.gather_for_metrics(preds)
                    val_preds_batch += preds.data.cpu().numpy().reshape(-1)
                    counter += 1

                    batch["image"] = torch.flip(image0, dims=[3])
                    with torch.no_grad():
                        outputs = model(batch)
                    preds = torch.sigmoid(outputs)
                    preds = accelerator.gather_for_metrics(preds)
                    val_preds_batch += preds.data.cpu().numpy().reshape(-1)
                    counter += 1

                    for k in [1, 2, 3]:
                        batch["image"] = torch.rot90(image0, k, dims=[2, 3])
                        with torch.no_grad():
                            outputs = model(batch)
                        preds = torch.sigmoid(outputs)
                        preds = accelerator.gather_for_metrics(preds)
                        val_preds_batch += preds.data.cpu().numpy().reshape(-1)
                        counter += 1
                val_preds_batch = val_preds_batch / counter
                val_preds.append(val_preds_batch)
                val_targets.append(val_targets_batch.data.cpu().numpy().reshape(-1))

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        epoch_auc = compute_auc(val_targets, val_preds)
        epoch_pauc = compute_pauc(val_targets, val_preds, min_tpr=0.8)

        if epoch_pauc > best_pauc:
            best_pauc = epoch_pauc
            best_auc = epoch_auc
            best_epoch = epoch
            best_val_preds = val_preds
        logger.info(
            f"Epoch {epoch} - Epoch pauc: {epoch_pauc} | Best auc: {best_auc} | Best pauc: {best_pauc} | Best "
            f"epoch: {best_epoch}"
        )

        output_dir = f"{args.model_dir}/models/fold_{args.fold}/epoch_{epoch}"
        accelerator.save_state(output_dir)

    logger.info(
        f"Fold: {args.fold} | Best pauc: {best_pauc} | Best auc: {best_auc} | Best epoch: {best_epoch}"
    )
    oof_df = pd.DataFrame(
        {
            "isic_id": val_metadata["isic_id"],
            "patient_id": val_metadata["patient_id"],
            "fold": args.fold,
            "target": val_metadata["target"],
            f"oof_{args.model_identifier}": best_val_preds,
        }
    )
    oof_df.to_csv(
        f"{args.model_dir}/oof_preds_{args.model_identifier}_fold_{args.fold}.csv",
        index=False,
    )

    fold_metadata = {
        "fold": args.fold,
        "best_epoch": best_epoch,
        "best_auc": best_auc,
        "best_pauc": best_pauc,
    }
    with open(f"{args.model_dir}/models/fold_{args.fold}/metadata.json", "w") as f:
        json.dump(fold_metadata, f)
    logger.info(f"Finished training fold {args.fold}")


if __name__ == "__main__":
    call_args = parse_args()
    main(call_args)
