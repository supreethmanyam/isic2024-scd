from io import BytesIO
from typing import List
import albumentations as A
import h5py
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from utils import logger
from collections import defaultdict


feature_mapping_dict = {
    "sex": defaultdict(lambda: 0, {
        "missing_sex": 0,
        "female": 1,
        "male": 2,
    }),
    "anatom_site_general": defaultdict(lambda: 0, {
        "missing_anatom_site_general": 0,
        "lower extremity": 1,
        "head/neck": 2,
        "posterior torso": 3,
        "anterior torso": 4,
        "upper extremity": 5,
    }),
    "tbp_tile_type": defaultdict(lambda: 0, {
        "3D: white": 0,
        "3D: XP": 1,
    }),
    "tbp_lv_location": defaultdict(lambda: 0, {
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
        "Torso Back": 20
    }),
    "tbp_lv_location_simple": defaultdict(lambda: 0, {
        "Unknown": 0,
        "Right Leg": 1,
        "Head & Neck": 2,
        "Torso Back": 3,
        "Torso Front": 4,
        "Right Arm": 5,
        "Left Leg": 6,
        "Left Arm": 7,
    }),
}


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


def test_augment(image_size, mean=None, std=None):
    if mean is not None and std is not None:
        normalize = A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0)
    else:
        normalize = A.Normalize(max_pixel_value=255.0, p=1.0)
    transform = A.Compose(
        [A.Resize(image_size, image_size), normalize, ToTensorV2()], p=1.0
    )
    return transform


class ISICDataset(Dataset):
    def __init__(self, metadata, images, augment,
                 use_meta=False, cat_cols: List = None, cont_cols: List = None,
                 infer=False):
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
            x_cont = torch.tensor([row[col] for col in self.cont_cols], dtype=torch.float)
        else:
            x_cat = torch.tensor(0)
            x_cont = torch.tensor(0)

        if self.infer:
            return image, x_cat, x_cont
        else:
            target = torch.tensor(row["target"])
            return image, x_cat, x_cont, target


def preprocess(df):
    df["age_approx"] = df["age_approx"].fillna(0)
    df["age_approx"] = df["age_approx"] / 90
    df["sex"] = df["sex"].fillna("missing_sex")
    df["sex"] = df["sex"].map(feature_mapping_dict["sex"])
    df["anatom_site_general"] = df["anatom_site_general"].fillna("missing_anatom_site_general")
    df["anatom_site_general"] = df["anatom_site_general"].map(feature_mapping_dict["anatom_site_general"])
    df["tbp_tile_type"] = df["tbp_tile_type"].map(feature_mapping_dict["tbp_tile_type"])
    df["tbp_lv_location"] = df["tbp_lv_location"].map(feature_mapping_dict["tbp_lv_location"])
    df["tbp_lv_location_simple"] = df["tbp_lv_location_simple"].map(feature_mapping_dict["tbp_lv_location_simple"])
    return df


def get_emb_szs(cat_cols):
    emb_szs = {}
    for col in cat_cols:
        emb_szs[col] = (len(feature_mapping_dict[col]), min(600, round(1.6 * len(feature_mapping_dict[col]) ** 0.56)))
    return emb_szs


def norm_feature(df, value_col, group_cols, err=1e-5):
    stats = ["mean", "std"]
    tmp = df.groupby(group_cols)[value_col].agg(stats)
    tmp.columns = [f"{value_col}_{stat}" for stat in stats]
    tmp.reset_index(inplace=True)
    df = df.merge(tmp, on=group_cols, how="left")
    feature_name = f"{value_col}_patient_norm"
    df[feature_name] = ((df[value_col] - df[f"{value_col}_mean"]) / (df[f"{value_col}_std"] + err))
    return df, feature_name


def feature_engineering(df):
    cat_cols = ["sex", "anatom_site_general",
                "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"]
    cont_cols = ["age_approx",
                 "clin_size_long_diam_mm",
                 "tbp_lv_A", "tbp_lv_Aext",
                 "tbp_lv_B", "tbp_lv_Bext",
                 "tbp_lv_C", "tbp_lv_Cext",
                 "tbp_lv_H", "tbp_lv_Hext",
                 "tbp_lv_L", "tbp_lv_Lext",
                 "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio",
                 "tbp_lv_color_std_mean",
                 # "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLB", "tbp_lv_deltaLBnorm",
                 "tbp_lv_eccentricity",
                 "tbp_lv_minorAxisMM", "tbp_lv_nevi_confidence", "tbp_lv_norm_border",
                 "tbp_lv_norm_color", "tbp_lv_perimeterMM",
                 "tbp_lv_radial_color_std_max", "tbp_lv_stdL", "tbp_lv_stdLExt",
                 "tbp_lv_symm_2axis", "tbp_lv_symm_2axis_angle",
                 # "tbp_lv_x", "tbp_lv_y", "tbp_lv_z"
                 ]

    # for col in cont_cols:
    #     df, feature_name = norm_feature(df, col, ["patient_id"])
    #     cont_cols.append(feature_name)
    df["num_images"] = df["patient_id"].map(df.groupby("patient_id")["isic_id"].count())
    cont_cols.append("num_images")

    for col in cont_cols:
        df[col] = np.log(df[col] + 30)
        df[col] = df[col].fillna(0)
    return df, cat_cols, cont_cols


def get_data(data_dir):
    train_metadata = pd.read_csv(f"{data_dir}/train-metadata.csv", low_memory=False)
    train_images = h5py.File(f"{data_dir}/train-image.hdf5", mode="r")

    folds_df = pd.read_csv(f"{data_dir}/folds.csv")
    train_metadata = train_metadata.merge(
        folds_df, on=["isic_id", "patient_id"], how="inner"
    )

    logger.info(f"Preprocessing metadata...")
    train_metadata = preprocess(train_metadata)

    logger.info(f"Feature engineering...")
    train_metadata, cat_cols, cont_cols = feature_engineering(train_metadata)

    emb_szs = get_emb_szs(cat_cols)
    return train_metadata, train_images, cat_cols, cont_cols, emb_szs
