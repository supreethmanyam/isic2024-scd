from io import BytesIO

import albumentations as A
import h5py
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

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
        "Lichen planus like keratosis": "unknown",
        "Pigmented benign keratosis": "unknown",
        "Seborrheic keratosis": "unknown",
        "Solar lentigo": "unknown",
        "Nevus": "unknown",
        "Angiofibroma": "unknown",
        "Dermatofibroma": "unknown",
        "Fibroepithelial polyp": "unknown",
        "Scar": "unknown",
        "Hemangioma": "unknown",
        "Trichilemmal or isthmic-catagen or pilar cyst": "unknown",
        "Lentigo NOS": "unknown",
        "Verruca": "unknown",
        "Solar or actinic keratosis": "unknown",
        "Atypical intraepithelial melanocytic proliferation": "unknown",
        "Atypical melanocytic neoplasm": "unknown",
        "Basal cell carcinoma": "BCC",
        "Squamous cell carcinoma in situ": "SCC",
        "Squamous cell carcinoma, Invasive": "SCC",
        "Squamous cell carcinoma, NOS": "SCC",
        "Melanoma in situ": "MEL",
        "Melanoma Invasive": "MEL",
        "Melanoma metastasis": "MEL",
        "Melanoma, NOS": "MEL"
    },
    "2020": {
        "nevus": "unknown",
        "melanoma": "MEL",
        "seborrheic keratosis": "unknown",
        "lentigo NOS": "unknown",
        "lichenoid keratosis": "unknown",
        "other": "unknown",
        "solar lentigo": "unknown",
        "scar": "unknown",
        "cafe-au-lait macule": "unknown",
        "atypical melanocytic proliferation": "unknown",
        "pigmented benign keratosis": "unknown"
    },
    "2019": {
        "nevus": "unknown",
        "melanoma": "MEL",
        "seborrheic keratosis": "unknown",
        "pigmented benign keratosis": "unknown",
        "dermatofibroma": "unknown",
        "squamous cell carcinoma": "SCC",
        "basal cell carcinoma": "BCC",
        "vascular lesion": "unknown",
        "actinic keratosis": "unknown",
        "solar lentigo": "unknown",
    },
}
all_labels = np.unique(list(multi_target_mapping_dict["2024"].values()) +
                       list(multi_target_mapping_dict["2020"].values()) +
                       list(multi_target_mapping_dict["2019"].values()))
label2idx = {label: idx for idx, label in enumerate(all_labels)}
malignant_labels = ["BCC", "MEL", "SCC"]
malignant_idx = [label2idx[label] for label in malignant_labels]


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


def test_augment(image_size):
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
            target = torch.tensor(row["label"])
            return image, target


def get_data(data_dir, data_2020_dir, data_2019_dir, target_mode, only_malignant=True):
    train_metadata = pd.read_csv(f"{data_dir}/train-metadata.csv", low_memory=False)
    train_images = h5py.File(f"{data_dir}/train-image.hdf5", mode="r")

    folds_df = pd.read_csv(f"{data_dir}/folds.csv")
    train_metadata = train_metadata.merge(
        folds_df, on=["isic_id", "patient_id"], how="inner"
    )

    if target_mode == "binary":
        train_metadata["label"] = train_metadata["target"]
    elif target_mode == "multi":
        train_metadata["label"] = train_metadata["iddx_3"].fillna("unknown")
        train_metadata["label"] = train_metadata["label"].replace(multi_target_mapping_dict["2024"])
        train_metadata["label"] = train_metadata["label"].map(label2idx)
    else:
        raise ValueError(f"Invalid target_mode: {target_mode}")

    if data_2020_dir is not None:
        train_metadata_2020 = pd.read_csv(
            f"{data_2020_dir}/train-metadata.csv", low_memory=False
        )
        train_images_2020 = h5py.File(f"{data_2020_dir}/train-image.hdf5", mode="r")
        if target_mode == "binary":
            train_metadata_2020["label"] = train_metadata_2020["benign_malignant"].map(
                binary_target_mapping_dict["2020"]
            )
            if only_malignant:
                train_metadata_2020 = train_metadata_2020[
                    train_metadata_2020["label"] == 1
                ].reset_index(drop=True)
        elif target_mode == "multi":
            train_metadata_2020["label"] = train_metadata_2020["diagnosis"].fillna("unknown")
            train_metadata_2020['label'] = train_metadata_2020["label"].replace(multi_target_mapping_dict["2020"])
            train_metadata_2020['label'] = train_metadata_2020['label'].map(label2idx)
            if only_malignant:
                train_metadata_2020 = train_metadata_2020[
                    train_metadata_2020["label"].isin(malignant_idx)
                ].reset_index(drop=True)
        else:
            raise ValueError(f"Invalid target_mode: {target_mode}")
    else:
        train_metadata_2020 = pd.DataFrame()
        train_images_2020 = None

    if data_2019_dir is not None:
        train_metadata_2019 = pd.read_csv(
            f"{data_2019_dir}/train-metadata.csv", low_memory=False
        )
        train_images_2019 = h5py.File(f"{data_2019_dir}/train-image.hdf5", mode="r")
        if target_mode == "binary":
            train_metadata_2019["label"] = train_metadata_2019["diagnosis"].map(
                binary_target_mapping_dict["2019"]
            )
            if only_malignant:
                train_metadata_2019 = train_metadata_2019[
                    train_metadata_2019["label"] == 1
                ].reset_index(drop=True)
        elif target_mode == "multi":
            train_metadata_2019['label'] = train_metadata_2019["diagnosis"].replace(multi_target_mapping_dict["2019"])
            train_metadata_2019['label'] = train_metadata_2019['label'].map(label2idx)
            if only_malignant:
                train_metadata_2019 = train_metadata_2019[
                    train_metadata_2019["label"].isin(malignant_idx)
                ].reset_index(drop=True)
        else:
            raise ValueError(f"Invalid target_mode: {target_mode}")
    else:
        train_metadata_2019 = pd.DataFrame()
        train_images_2019 = None

    return (
        train_metadata,
        train_images,
        train_metadata_2020,
        train_images_2020,
        train_metadata_2019,
        train_images_2019,
    )
