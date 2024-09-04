import os
import json
import time
from typing import List, Dict
from io import BytesIO
from collections import defaultdict

import numpy as np
import pandas as pd

from PIL import Image

from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler

from albumentations.pytorch import ToTensorV2
import albumentations as A

from imblearn.under_sampling import RandomUnderSampler

from timm import create_model
from safetensors import safe_open

from sklearn.metrics import auc, roc_auc_score, roc_curve


feature_mapping_dict = {
    "sex": defaultdict(
        lambda: 0,
        {
            "missing_sex": 0,
            "female": 1,
            "male": 2,
        },
    ),
    "anatom_site_general": defaultdict(
        lambda: 0,
        {
            "missing_anatom_site_general": 0,
            "lower extremity": 1,
            "head/neck": 2,
            "posterior torso": 3,
            "anterior torso": 4,
            "upper extremity": 5,
        },
    ),
    "tbp_tile_type": defaultdict(
        lambda: 0,
        {
            "3D: white": 0,
            "3D: XP": 1,
        },
    ),
    "tbp_lv_location": defaultdict(
        lambda: 0,
        {
            "Unknown": 0,
            "Right Leg - Upper": 1,
            "Head & Neck": 2,
            "Torso Back Top Third": 3,
            "Torso Front Top Half": 4,
            "Right Arm - Upper": 5,
            "Left Leg - Upper": 6,
            "Torso Front Bottom Half": 7,
            "Left Arm - Upper": 8,
            "Right Leg": 9,
            "Torso Back Middle Third": 10,
            "Right Arm - Lower": 11,
            "Right Leg - Lower": 12,
            "Left Leg - Lower": 13,
            "Left Arm - Lower": 14,
            "Left Leg": 15,
            "Torso Back Bottom Third": 16,
            "Left Arm": 17,
            "Right Arm": 18,
            "Torso Front": 19,
            "Torso Back": 20,
        },
    ),
    "tbp_lv_location_simple": defaultdict(
        lambda: 0,
        {
            "Unknown": 0,
            "Right Leg": 1,
            "Head & Neck": 2,
            "Torso Back": 3,
            "Torso Front": 4,
            "Right Arm": 5,
            "Left Leg": 6,
            "Left Arm": 7,
        },
    ),
}

model_factory = {
    "tf_efficientnet_b1_ns": "tf_efficientnet_b1.ns_jft_in1k",
    "mobilevitv2_200": "mobilevitv2_200.cvnets_in22k_ft_in1k"
}


def get_emb_szs(cat_cols):
    emb_szs = {}
    for col in cat_cols:
        emb_szs[col] = (
            len(feature_mapping_dict[col]),
            min(600, round(1.6 * len(feature_mapping_dict[col]) ** 0.56)),
        )
    return emb_szs


def cnn_norm_feature(df, value_col, group_cols, err=1e-5):
    stats = ["mean", "std"]
    tmp = df.groupby(group_cols)[value_col].agg(stats)
    tmp.columns = [f"{value_col}_{stat}" for stat in stats]
    tmp.reset_index(inplace=True)
    df = df.merge(tmp, on=group_cols, how="left")
    feature_name = f"{value_col}_patient_norm"
    df[feature_name] = (
        (df[value_col] - df[f"{value_col}_mean"]) / (df[f"{value_col}_std"] + err)
    ).fillna(0)
    return df, feature_name


def cnn_feature_engineering(df):
    df["age_approx"] = df["age_approx"].fillna(0)
    df["age_approx"] = df["age_approx"] / 90
    df["sex"] = df["sex"].fillna("missing_sex")
    df["sex"] = df["sex"].map(feature_mapping_dict["sex"])
    df["anatom_site_general"] = df["anatom_site_general"].fillna(
        "missing_anatom_site_general"
    )
    df["anatom_site_general"] = df["anatom_site_general"].map(
        feature_mapping_dict["anatom_site_general"]
    )
    df["tbp_tile_type"] = df["tbp_tile_type"].map(feature_mapping_dict["tbp_tile_type"])
    df["tbp_lv_location"] = df["tbp_lv_location"].map(
        feature_mapping_dict["tbp_lv_location"]
    )
    df["tbp_lv_location_simple"] = df["tbp_lv_location_simple"].map(
        feature_mapping_dict["tbp_lv_location_simple"]
    )

    cat_cols = [
        "sex",
        "anatom_site_general",
        "tbp_tile_type",
        "tbp_lv_location",
        "tbp_lv_location_simple",
    ]

    df["num_images"] = df["patient_id"].map(df.groupby("patient_id")["isic_id"].count())
    df["num_images"] = np.log1p(df["num_images"])

    cols_to_norm = [
        "age_approx",
        "clin_size_long_diam_mm",
        "tbp_lv_A",
        "tbp_lv_Aext",
        "tbp_lv_B",
        "tbp_lv_Bext",
        "tbp_lv_C",
        "tbp_lv_Cext",
        "tbp_lv_H",
        "tbp_lv_Hext",
        "tbp_lv_L",
        "tbp_lv_Lext",
        "tbp_lv_areaMM2",
        "tbp_lv_area_perim_ratio",
        "tbp_lv_color_std_mean",
        "tbp_lv_deltaA",
        "tbp_lv_deltaB",
        "tbp_lv_deltaL",
        "tbp_lv_deltaLB",
        "tbp_lv_deltaLBnorm",
        "tbp_lv_eccentricity",
        "tbp_lv_minorAxisMM",
        "tbp_lv_nevi_confidence",
        "tbp_lv_norm_border",
        "tbp_lv_norm_color",
        "tbp_lv_perimeterMM",
        "tbp_lv_radial_color_std_max",
        "tbp_lv_stdL",
        "tbp_lv_stdLExt",
        "tbp_lv_symm_2axis",
        "tbp_lv_symm_2axis_angle",
        "tbp_lv_x",
        "tbp_lv_y",
        "tbp_lv_z",
    ]
    cont_cols = cols_to_norm[:]
    for col in cols_to_norm:
        df, feature_name = cnn_norm_feature(df, col, ["patient_id"])
        cont_cols += [feature_name]

    df["num_images"] = np.log1p(
        df["patient_id"].map(df.groupby("patient_id")["isic_id"].count())
    )
    cont_cols += ["num_images"]
    assert df[cont_cols].isnull().sum().sum() == 0
    return df, cat_cols, cont_cols


def dev_augment(image_size, mean=None, std=None):
    if mean is not None and std is not None:
        normalize = A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0)
    else:
        normalize = A.Normalize(max_pixel_value=255.0, p=1.0)
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
            A.CoarseDropout(
                max_height=int(image_size * 0.375),
                max_width=int(image_size * 0.375),
                max_holes=1,
                min_holes=1,
                p=0.7,
            ),
            normalize,
            ToTensorV2(),
        ],
        p=1.0,
    )
    return transform


def val_augment(image_size, mean=None, std=None):
    if mean is not None and std is not None:
        normalize = A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0)
    else:
        normalize = A.Normalize(max_pixel_value=255.0, p=1.0)
    transform = A.Compose(
        [A.Resize(image_size, image_size), normalize, ToTensorV2()], p=1.0
    )
    return transform


def test_augment_binary(image_size, mean=None, std=None):
    if mean is not None and std is not None:
        normalize = A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0)
    else:
        normalize = A.Normalize(max_pixel_value=255.0, p=1.0)
    transform = A.Compose(
        [A.Resize(image_size, image_size), normalize, ToTensorV2()], p=1.0
    )
    return transform


class ISICDatasetBinary(Dataset):
    def __init__(
        self,
        metadata,
        images,
        augment,
        use_meta=False,
        cat_cols: List = None,
        cont_cols: List = None,
        infer=False,
    ):
        self.metadata = metadata
        self.images = images
        self.augment = augment
        self.use_meta = use_meta
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.length = len(self.metadata)
        self.infer = infer

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        image = np.array(Image.open(BytesIO(self.images[row["isic_id"]][()])))
        if self.augment is not None:
            image = self.augment(image=image)["image"].float()

        if self.use_meta:
            x_cat = torch.tensor([row[col] for col in self.cat_cols], dtype=torch.long)
            x_cont = torch.tensor(
                [row[col] for col in self.cont_cols], dtype=torch.float
            )
        else:
            x_cat = torch.tensor(0)
            x_cont = torch.tensor(0)

        if self.infer:
            return image, x_cat, x_cont
        else:
            target = torch.tensor(row["label"])
            return image, x_cat, x_cont, target


class ISICNetBinary(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=True,
        use_meta=False,
        cat_cols: List = None,
        cont_cols: List = None,
        emb_szs: Dict = None,
    ):
        super(ISICNetBinary, self).__init__()
        model_name = model_factory.get(model_name, model_name)
        self.model = create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="",
        )
        in_dim = self.model.num_features
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.use_meta = use_meta
        if use_meta:
            self.linear = nn.Linear(in_dim, 256)

            self.embeddings = nn.ModuleList(
                [nn.Embedding(emb_szs[col][0], emb_szs[col][1]) for col in cat_cols]
            )
            self.embedding_dropout = nn.Dropout(0.1)
            n_emb = sum([emb_szs[col][1] for col in cat_cols])
            n_cont = len(cont_cols)
            self.bn_cont = nn.BatchNorm1d(n_cont)
            self.meta = nn.Sequential(
                nn.Linear(n_emb + n_cont, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Dropout(0.1),
            )
            self.classifier = nn.Linear(256 + 64, 1)
        else:
            self.linear = nn.Linear(in_dim, 1)

    def forward(self, images, x_cat=None, x_cont=None):
        x = self.model(images)
        bs = len(images)
        pool = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        if self.training:
            x_image = 0
            for i in range(len(self.dropouts)):
                x_image += self.linear(self.dropouts[i](pool))
            x_image = x_image / len(self.dropouts)
        else:
            x_image = self.linear(pool)

        if self.use_meta:
            x_cat = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x_cat = torch.cat(x_cat, 1)
            x_cat = self.embedding_dropout(x_cat)
            x_cont = self.bn_cont(x_cont)
            x_meta = self.meta(torch.cat([x_cat, x_cont], 1))
            x = torch.cat([x_image, x_meta], 1)
            logits = self.classifier(x)
        else:
            logits = x_image
        return logits


def train_epoch(
    epoch,
    model,
    optimizer,
    criterion,
    dev_dataloader,
    lr_scheduler,
    accelerator,
    use_meta,
    log_interval=100,
):
    model.train()
    train_loss = []
    total_steps = len(dev_dataloader)
    for step, (images, x_cat, x_cont, targets) in enumerate(dev_dataloader):
        optimizer.zero_grad()
        if use_meta:
            logits = model(images, x_cat, x_cont)
        else:
            logits = model(images)
        probs = torch.sigmoid(logits)
        targets = targets.float().unsqueeze(1)
        loss = criterion(probs, targets)
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0)
        optimizer.step()
        lr_scheduler.step()

        loss_value = accelerator.gather(loss).item()
        train_loss.append(loss_value)
        smooth_loss = sum(train_loss[-500:]) / min(len(train_loss), 500)
        if (step == 0) or ((step + 1) % log_interval == 0):
            print(
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
    n_tta,
    use_meta,
    log_interval=10,
):
    model.eval()
    val_probs = []
    val_targets = []
    val_loss = []
    total_steps = len(val_dataloader)
    with torch.no_grad():
        for step, (images, x_cat, x_cont, targets) in enumerate(val_dataloader):
            logits = 0
            probs = 0
            for i in range(n_tta):
                if use_meta:
                    logits_iter = model(get_trans(images, i), x_cat, x_cont)
                else:
                    logits_iter = model(get_trans(images, i))
                logits += logits_iter
                probs += torch.sigmoid(logits_iter)
            logits /= n_tta
            probs /= n_tta

            targets = targets.float().unsqueeze(1)
            loss = criterion(probs, targets)
            val_loss.append(loss.detach().cpu().numpy())

            probs, targets = accelerator.gather((probs, targets))
            val_probs.append(probs)
            val_targets.append(targets)

            if (step == 0) or ((step + 1) % log_interval == 0):
                print(f"Epoch: {epoch} | Step: {step + 1}/{total_steps}")

    val_loss = np.mean(val_loss)
    val_probs = torch.cat(val_probs).cpu().numpy()
    val_targets = torch.cat(val_targets).cpu().numpy()
    val_auc = compute_auc(val_targets, val_probs)
    val_pauc = compute_pauc(val_targets, val_probs, min_tpr=0.8)
    return (
        val_loss,
        val_auc,
        val_pauc,
        val_probs,
        val_targets,
    )


def predict_binary(
    model, test_dataloader, accelerator, n_tta, use_meta, log_interval=10
):
    model.eval()
    test_probs = []
    total_steps = len(test_dataloader)
    with torch.no_grad():
        for step, (images, x_cat, x_cont) in enumerate(test_dataloader):
            logits = 0
            probs = 0
            for i in range(n_tta):
                if use_meta:
                    logits_iter = model(get_trans(images, i), x_cat, x_cont)
                else:
                    logits_iter = model(get_trans(images, i))
                logits += logits_iter
                probs += torch.sigmoid(logits_iter)
            logits /= n_tta
            probs /= n_tta

            probs = accelerator.gather(probs)
            test_probs.append(probs)

            if (step == 0) or ((step + 1) % log_interval == 0):
                print(f"Step: {step + 1}/{total_steps}")

    test_probs = torch.cat(test_probs).cpu().numpy()
    return test_probs


def compute_auc(y_true, y_pred) -> float:
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true: ground truth of 1s and 0s
        y_pred: predictions of scores ranging [0, 1]

    Returns:
        Float value range [0, 1]
    """
    return roc_auc_score(y_true, y_pred)


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


def main(args, train_metadata, train_images, test_psuedo_metadata, test_metadata, test_images):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.model_dir
    )
    kwargs = DistributedDataParallelKwargs()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.seed is not None:
        set_seed(args.seed, deterministic=True)

    train_metadata, cat_cols, cont_cols = cnn_feature_engineering(train_metadata)
    emb_szs = get_emb_szs(cat_cols)
    test_psuedo_metadata, _, _ = cnn_feature_engineering(test_psuedo_metadata)
    test_metadata, _, _ = cnn_feature_engineering(test_metadata)

    dev_index = train_metadata[train_metadata[args.fold_column] != args.fold].index
    val_index = train_metadata[train_metadata[args.fold_column] == args.fold].index

    dev_df = train_metadata.loc[dev_index, :].reset_index(drop=True)
    pos_samples = dev_df[dev_df["target"] == 1]
    num_pos = len(pos_samples)
    num_neg = int(num_pos * (1.0 / args.sampling_rate))
    num_psuedo_samples = test_psuedo_metadata.shape[0]

    mean = None
    std = None

    test_psuedo_dataset = ISICDatasetBinary(
        test_psuedo_metadata,
        test_images,
        augment=dev_augment(args.image_size, mean=mean, std=std),
        use_meta=args.use_meta,
        cat_cols=cat_cols,
        cont_cols=cont_cols,
        infer=False,
    )

    val_metadata = train_metadata.loc[val_index, :].reset_index(drop=True)
    val_dataset = ISICDatasetBinary(
        val_metadata,
        train_images,
        augment=val_augment(args.image_size, mean=mean, std=std),
        use_meta=args.use_meta,
        cat_cols=cat_cols,
        cont_cols=cont_cols,
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

    test_metadata = test_metadata.reset_index(drop=True)
    test_dataset = ISICDatasetBinary(
        test_metadata,
        test_images,
        augment=test_augment_binary(args.image_size, mean=mean, std=std),
        use_meta=args.use_meta,
        cat_cols=cat_cols,
        cont_cols=cont_cols,
        infer=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    print(f"Loading pretrained weights from {args.pretrained_weights_dir}")
    model = ISICNetBinary(
        model_name=args.model_name,
        pretrained=False,
        use_meta=args.use_meta,
        cat_cols=cat_cols,
        cont_cols=cont_cols,
        emb_szs=emb_szs,
    )
    tensors = {}
    with safe_open(f"{args.pretrained_weights_dir}/models/fold_{args.fold}/model.safetensors", framework="pt") as f:
        for key in f.keys():
            if "model" in key:
                tensors[key] = f.get_tensor(key)
    _ = model.load_state_dict(tensors, strict=False)
    print("Pretrained weights loaded successfully")

    model = model.to(accelerator.device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        pct_start=1 / args.num_epochs,
        max_lr=args.init_lr * 10,
        div_factor=10,
        epochs=args.num_epochs,
        steps_per_epoch=(num_pos + num_neg + num_psuedo_samples) // args.train_batch_size,
    )

    (
        model,
        optimizer,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, val_dataloader, lr_scheduler
    )

    best_val_loss = 0
    best_val_auc = 0
    best_val_pauc = 0
    best_epoch = 0
    best_val_probs = None
    train_losses = []
    val_losses = []
    val_paucs = []
    val_aucs = []
    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()
        rus = RandomUnderSampler(
            sampling_strategy={0: num_neg, 1: num_pos},
            random_state=args.seed + (epoch * 100),
        )
        dev_metadata, _ = rus.fit_resample(dev_df, dev_df["target"])
        dev_dataset = ISICDatasetBinary(
            dev_metadata,
            train_images,
            augment=dev_augment(args.image_size, mean=mean, std=std),
            use_meta=args.use_meta,
            cat_cols=cat_cols,
            cont_cols=cont_cols,
            infer=False,
        )
        dev_dataset = torch.utils.data.ConcatDataset([dev_dataset, test_psuedo_dataset])
        sampler = RandomSampler(dev_dataset)
        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=args.train_batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        dev_dataloader = accelerator.prepare(dev_dataloader)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Fold {args.fold} | Epoch {epoch} | LR {lr:.7f}")
        train_loss = train_epoch(
            epoch,
            model,
            optimizer,
            criterion,
            dev_dataloader,
            lr_scheduler,
            accelerator,
            args.use_meta,
        )
        (
            val_loss,
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
            args.use_meta,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_paucs.append(val_pauc)
        val_aucs.append(val_auc)
        print(
            f"Fold: {args.fold} | Epoch: {epoch} | LR: {lr:.7f} |"
            f" Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f} |"
            f" Val AUC: {val_auc:.5f} | Val pAUC: {val_pauc:.5f}"
        )
        if val_pauc > best_val_pauc:
            print(
                f"pAUC: {best_val_pauc:.5f} --> {val_pauc:.5f}, saving model..."
            )
            best_val_pauc = val_pauc
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_probs = val_probs
            output_dir = f"{args.model_dir}/models/fold_{args.fold}"
            accelerator.save_state(output_dir)
        else:
            print(
                f"pAUC: {best_val_pauc:.5f} --> {val_pauc:.5f}, skipping model save..."
            )
        elapsed_time = time.time() - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time % 60)
        print(f"Epoch {epoch} took {elapsed_mins}m {elapsed_secs}s")

    best_output_dir = f"{args.model_dir}/models/fold_{args.fold}"
    accelerator.load_state(best_output_dir)

    test_probs = predict_binary(
        model, test_dataloader, accelerator, args.n_tta, args.use_meta
    )

    oof_train_preds_df = pd.DataFrame(
        {
            "isic_id": val_metadata["isic_id"],
            "patient_id": val_metadata["patient_id"],
            "fold": args.fold,
            "target": val_metadata["target"],
            f"oof_{args.model_name}_{args.version}": best_val_probs.flatten(),
        }
    )
    oof_train_preds_df.to_csv(
        f"{args.model_dir}/oof_train_preds_{args.model_name}_{args.version}_fold_{args.fold}.csv",
        index=False,
    )

    oof_test_preds_df = pd.DataFrame(
        {
            "isic_id": test_metadata["isic_id"],
            "patient_id": test_metadata["patient_id"],
            f"oof_{args.model_name}_{args.version}_fold_{args.fold}": test_probs.flatten(),
        }
    )
    oof_test_preds_df.to_csv(
        f"{args.model_dir}/oof_test_preds_{args.model_name}_{args.version}_fold_{args.fold}.csv",
        index=False,
    )

    print(
        f"Fold: {args.fold} |"
        f" Best Val pAUC: {best_val_pauc} |"
        f" Best Val AUC: {best_val_auc} |"
        f" Best Val loss: {best_val_loss} |"
        f" Best epoch: {best_epoch}"
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
    print(f"Finished training fold {args.fold}")
    return oof_train_preds_df, oof_test_preds_df
