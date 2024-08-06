from io import BytesIO

import albumentations as A
import h5py
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

label_mapping = {
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
    "2018": {
        "nevus": "NV",
        "melanoma": "MEL",
        "pigmented benign keratosis": "BKL",
        "basal cell carcinoma": "BCC",
        "squamous cell carcinoma": "SCC",
        "vascular lesion": "VASC",
        "actinic keratosis": "AKIEC",
        "dermatofibroma": "DF",
    },
}


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
            A.CoarseDropout(
                max_height=int(image_size * 0.375),
                max_width=int(image_size * 0.375),
                max_holes=1,
                min_holes=1,
                p=0.7,
            ),
            A.Normalize(),
            ToTensorV2(),
        ],
        p=1.0,
    )
    return transform


def val_augment(image_size):
    transform = A.Compose(
        [A.Resize(image_size, image_size), A.Normalize(), ToTensorV2()], p=1.0
    )
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
        row = self.metadata.iloc[index]
        image = np.array(Image.open(BytesIO(self.images[row["isic_id"]][()])))
        if self.augment is not None:
            image = self.augment(image=image)["image"].float()
        if self.infer:
            return image
        else:
            label = torch.tensor(row["label"]).float()
            return image, label


def get_data(data_dir, data_2020_dir, data_2019_dir, data_2018_dir, only_malignant=True):
    # all_labels = np.unique(
    #     list(label_mapping["2024"].values())
    #     + list(label_mapping["2020"].values())
    #     + list(label_mapping["2019"].values())
    #     + list(label_mapping["2018"].values())
    # )
    # label2idx = {label: idx for idx, label in enumerate(all_labels)}
    malignant_labels = ["BCC", "MEL", "SCC"]

    train_metadata = pd.read_csv(f"{data_dir}/train-metadata.csv", low_memory=False)
    train_images = h5py.File(f"{data_dir}/train-image.hdf5", mode="r")

    folds_df = pd.read_csv(f"{data_dir}/folds.csv")
    train_metadata = train_metadata.merge(
        folds_df, on=["isic_id", "patient_id"], how="inner"
    )

    train_metadata["label"] = train_metadata["target"]

    if data_2020_dir is not None:
        train_metadata_2020 = pd.read_csv(
            f"{data_2020_dir}/train-metadata.csv", low_memory=False
        )
        train_images_2020 = h5py.File(f"{data_2020_dir}/train-image.hdf5", mode="r")
        train_metadata_2020["label"] = train_metadata_2020["diagnosis"].fillna(
            "unknown"
        )
        train_metadata_2020["label"] = train_metadata_2020["label"].replace(
            label_mapping["2020"]
        )
        train_metadata_2020["label"] = np.where(
            train_metadata_2020["label"].isin(malignant_labels), 1, 0
        )

        if only_malignant:
            train_metadata_2020 = train_metadata_2020[
                train_metadata_2020["label"] == 1
            ].reset_index(drop=True)
    else:
        train_metadata_2020 = pd.DataFrame()
        train_images_2020 = None

    if data_2019_dir is not None:
        train_metadata_2019 = pd.read_csv(
            f"{data_2019_dir}/train-metadata.csv", low_memory=False
        )
        train_images_2019 = h5py.File(f"{data_2019_dir}/train-image.hdf5", mode="r")
        train_metadata_2019["label"] = train_metadata_2019["diagnosis"].replace(
            label_mapping["2019"]
        )
        train_metadata_2019["label"] = np.where(
            train_metadata_2019["label"].isin(malignant_labels), 1, 0
        )
        if only_malignant:
            train_metadata_2019 = train_metadata_2019[
                train_metadata_2019["label"] == 1
            ].reset_index(drop=True)
    else:
        train_metadata_2019 = pd.DataFrame()
        train_images_2019 = None

    if data_2018_dir is not None:
        train_metadata_2018 = pd.read_csv(
            f"{data_2018_dir}/train-metadata.csv", low_memory=False
        )
        train_images_2018 = h5py.File(f"{data_2018_dir}/train-image.hdf5", mode="r")
        train_metadata_2018["patient_id"] = "unknown"
        train_metadata_2018["label"] = train_metadata_2018["diagnosis"].replace(
            label_mapping["2018"]
        )
        train_metadata_2018["label"] = np.where(
            train_metadata_2018["label"].isin(malignant_labels), 1, 0
        )
        if only_malignant:
            train_metadata_2018 = train_metadata_2018[
                train_metadata_2018["label"] == 1
            ].reset_index(drop=True)
    else:
        train_metadata_2018 = pd.DataFrame()
        train_images_2018 = None

    return (
        train_metadata,
        train_images,
        train_metadata_2020,
        train_images_2020,
        train_metadata_2019,
        train_images_2019,
        train_metadata_2018,
        train_images_2018,
    )
