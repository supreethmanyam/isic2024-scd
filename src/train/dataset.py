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
from collections import defaultdict


feature_mapping_dict_v1 = {
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
    })
}

binary_target_mapping_dict = {
    "2020": {"benign": 0, "malignant": 1},
    "2019": {
        "nevus": 0,
        "melanoma": 1,
        "seborrheic keratosis": 0,
        "pigmented benign keratosis": 0,
        "dermatofibroma": 0,
        "squamous cell carcinoma": 1,
        "basal cell carcinoma": 1,
        "vascular lesion": 0,
        "actinic keratosis": 0,
        "solar lentigo": 0,
    },
}
multi_target_mapping_dict = {
    "2024": {
        "Hidradenoma": "unknown",
        "Lichen planus like keratosis": "BKL",
        "Pigmented benign keratosis": "BKL",
        "Seborrheic keratosis": "BKL",
        "Solar lentigo": "BKL",
        "Nevus": "NV",
        "Angiofibroma": "unknown",
        "Dermatofibroma": "DF",
        "Fibroepithelial polyp": "unknown",
        "Scar": "unknown",
        "Hemangioma": "unknown",
        "Trichilemmal or isthmic-catagen or pilar cyst": "unknown",
        "Lentigo NOS": "BKL",
        "Verruca": "unknown",
        "Solar or actinic keratosis": "AKIEC",
        "Atypical intraepithelial melanocytic proliferation": "unknown",
        "Atypical melanocytic neoplasm": "unknown",
        "Basal cell carcinoma": "BCC",
        "Squamous cell carcinoma in situ": "SCC",
        "Squamous cell carcinoma, Invasive": "SCC",
        "Squamous cell carcinoma, NOS": "SCC",
        "Melanoma in situ": "MEL",
        "Melanoma Invasive": "MEL",
        "Melanoma metastasis": "MEL",
        "Melanoma, NOS": "MEL",
    },
    "2020": {
        "nevus": "NV",
        "melanoma": "MEL",
        "seborrheic keratosis": "BKL",
        "lentigo NOS": "BKL",
        "lichenoid keratosis": "BKL",
        "other": "unknown",
        "solar lentigo": "BKL",
        "scar": "unknown",
        "cafe-au-lait macule": "unknown",
        "atypical melanocytic proliferation": "unknown",
        "pigmented benign keratosis": "BKL",
    },
    "2019": {
        "nevus": "NV",
        "melanoma": "MEL",
        "seborrheic keratosis": "BKL",
        "pigmented benign keratosis": "BKL",
        "dermatofibroma": "DF",
        "squamous cell carcinoma": "SCC",
        "basal cell carcinoma": "BCC",
        "vascular lesion": "VASC",
        "actinic keratosis": "AKIEC",
        "solar lentigo": "BKL",
    },
}
all_labels = np.unique(
    list(multi_target_mapping_dict["2024"].values())
    + list(multi_target_mapping_dict["2020"].values())
    + list(multi_target_mapping_dict["2019"].values())
)
label2idx = {label: idx for idx, label in enumerate(all_labels)}
malignant_labels = ["BCC", "MEL", "SCC"]
malignant_idx = [label2idx[label] for label in malignant_labels]


def dev_augment_v1(image_size, mean=None, std=None):
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


def val_augment_v1(image_size, mean=None, std=None):
    if mean is not None and std is not None:
        normalize = A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0)
    else:
        normalize = A.Normalize(max_pixel_value=255.0, p=1.0)
    transform = A.Compose(
        [A.Resize(image_size, image_size), normalize, ToTensorV2()], p=1.0
    )
    return transform


def test_augment_v1(image_size, mean=None, std=None):
    if mean is not None and std is not None:
        normalize = A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0)
    else:
        normalize = A.Normalize(max_pixel_value=255.0, p=1.0)
    transform = A.Compose(
        [A.Resize(image_size, image_size), normalize, ToTensorV2()], p=1.0
    )
    return transform


class ISICDatasetV1(Dataset):
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
            target = torch.tensor(row["label"])
            return image, x_cat, x_cont, target


def preprocess_v1(df):
    df["age_approx"] = df["age_approx"].fillna(0)
    df["age_approx"] = df["age_approx"] / 90
    df["sex"] = df["sex"].fillna("missing_sex")
    df["sex"] = df["sex"].map(feature_mapping_dict_v1["sex"])
    df["anatom_site_general"] = df["anatom_site_general"].fillna("missing_anatom_site_general")
    df["anatom_site_general"] = df["anatom_site_general"].map(feature_mapping_dict_v1["anatom_site_general"])
    return df


def get_emb_szs_v1(cat_cols):
    emb_szs = {}
    for col in cat_cols:
        emb_szs[col] = (len(feature_mapping_dict_v1[col]),
                        min(600, round(1.6 * len(feature_mapping_dict_v1[col]) ** 0.56)))
    return emb_szs


def feature_engineering_v1(df):
    cat_cols = ["sex", "anatom_site_general"]
    cont_cols = ["age_approx"]
    return df, cat_cols, cont_cols


def get_data_v1(data_dir, data_2020_dir, data_2019_dir):
    train_metadata = pd.read_csv(f"{data_dir}/train-metadata.csv", low_memory=False)
    train_images = h5py.File(f"{data_dir}/train-image.hdf5", mode="r")

    folds_df = pd.read_csv(f"{data_dir}/folds.csv")
    train_metadata = train_metadata.merge(
        folds_df, on=["isic_id", "patient_id"], how="inner"
    )

    train_metadata = preprocess_v1(train_metadata)
    train_metadata, cat_cols, cont_cols = feature_engineering_v1(train_metadata)
    emb_szs = get_emb_szs_v1(cat_cols)

    train_metadata["label"] = train_metadata["iddx_3"].fillna("unknown")
    train_metadata["label"] = train_metadata["label"].replace(
        multi_target_mapping_dict["2024"]
    )
    train_metadata["label"] = train_metadata["label"].map(label2idx)

    if data_2020_dir is not None:
        train_metadata_2020 = pd.read_csv(f"{data_2020_dir}/train-metadata.csv", low_memory=False)
        train_images_2020 = h5py.File(f"{data_2020_dir}/train-image.hdf5", mode="r")

        train_metadata_2020 = preprocess_v1(train_metadata_2020)
        train_metadata_2020, _, _ = feature_engineering_v1(train_metadata_2020)

        train_metadata_2020["label"] = train_metadata_2020["diagnosis"].fillna(
            "unknown"
        )
        train_metadata_2020["label"] = train_metadata_2020["label"].replace(
            multi_target_mapping_dict["2020"]
        )
        train_metadata_2020["label"] = train_metadata_2020["label"].map(label2idx)
        train_metadata_2020["target"] = train_metadata_2020["benign_malignant"].map(
            binary_target_mapping_dict["2020"]
        ).astype(int)
    else:
        train_metadata_2020 = pd.DataFrame()
        train_images_2020 = None

    if data_2019_dir is not None:
        train_metadata_2019 = pd.read_csv(f"{data_2019_dir}/train-metadata.csv", low_memory=False)
        train_images_2019 = h5py.File(f"{data_2019_dir}/train-image.hdf5", mode="r")

        train_metadata_2019 = preprocess_v1(train_metadata_2019)
        train_metadata_2019, _, _ = feature_engineering_v1(train_metadata_2019)

        train_metadata_2019["label"] = train_metadata_2019["diagnosis"].replace(
            multi_target_mapping_dict["2019"]
        )
        train_metadata_2019["label"] = train_metadata_2019["label"].map(label2idx)
        train_metadata_2019["target"] = train_metadata_2019["diagnosis"].map(
            binary_target_mapping_dict["2019"]
        ).astype(int)
    else:
        train_metadata_2019 = pd.DataFrame()
        train_images_2019 = None

    return (train_metadata, train_images,
            train_metadata_2020, train_images_2020,
            train_metadata_2019, train_images_2019,
            cat_cols, cont_cols, emb_szs)
