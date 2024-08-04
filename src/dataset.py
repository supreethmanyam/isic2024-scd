from io import BytesIO

import albumentations as A
import h5py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
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
    def __init__(self, metadata, images, augment, feature_cols, infer=False):
        self.metadata = metadata
        self.images = images
        self.augment = augment
        self.feature_cols = feature_cols
        self.length = len(self.metadata)
        self.infer = infer

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        row = self.metadata.iloc[index]

        image = np.array(Image.open(BytesIO(self.images[row["isic_id"]][()])))
        image = self.augment(image=image)["image"]

        if self.feature_cols is not None:
            data = (image.float().div(255), torch.tensor(row[self.feature_cols].values.tolist()).float())
        else:
            data = image.float().div(255)

        if not self.infer:
            label = torch.tensor(row["label"]).long()
            return data, label

        return data


def feature_engineering(metadata):
    metadata["sex"] = metadata["sex"].map({"male": 1, "female": 0})
    metadata["sex"] = metadata["sex"].fillna(-1)

    metadata["age_approx"] = metadata["age_approx"].fillna(0)
    metadata["age_approx"] = metadata["age_approx"] / 90

    metadata["patient_id"] = metadata["patient_id"].fillna("unknown")
    metadata["num_images"] = metadata["patient_id"].map(
        metadata.groupby("patient_id")["isic_id"].count()
    )
    metadata.loc[metadata["patient_id"] == "unknown", "num_images"] = 1
    metadata["num_images"] = np.log1p(metadata["num_images"])

    return metadata


def fit_encoder_and_transform(train, train_2020, train_2019):
    numerical_features = ["sex", "age_approx", "num_images"]
    ohe_categorical_features = ["anatom_site_general"]
    mixed_encoded_preprocessor = ColumnTransformer(
        [
            ("numerical", "passthrough", numerical_features),
            (
                "ohe_categorical",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                ohe_categorical_features
            )
        ],
        verbose_feature_names_out=False,
    )
    mixed_encoded_preprocessor.set_output(transform="pandas")

    mixed_encoded_preprocessor.fit(train)
    train_features = mixed_encoded_preprocessor.transform(train)
    feature_cols = [f"feature_{col}" for col in train_features.columns]
    train_features.columns = feature_cols
    if not train_2020.empty:
        train_2020_features = mixed_encoded_preprocessor.transform(train_2020)
        train_2020_features.columns = feature_cols
    else:
        train_2020_features = None
    if not train_2019.empty:
        train_2019_features = mixed_encoded_preprocessor.transform(train_2019)
        train_2019_features.columns = feature_cols
    else:
        train_2019_features = None
    return mixed_encoded_preprocessor, feature_cols, train_features, train_2020_features, train_2019_features


def get_data(data_dir, data_2020_dir, data_2019_dir, out_dim, debug, seed):
    all_labels = np.unique(
        list(label_mapping["2024"].values())
        + list(label_mapping["2020"].values())
        + list(label_mapping["2019"].values())
    )
    label2idx = {label: idx for idx, label in enumerate(all_labels)}
    malignant_labels = ["BCC", "MEL", "SCC"]
    malignant_idx = [label2idx[label] for label in malignant_labels]

    train_metadata = pd.read_csv(f"{data_dir}/train-metadata.csv", low_memory=False)
    train_images = h5py.File(f"{data_dir}/train-image.hdf5", mode="r")

    folds_df = pd.read_csv(f"{data_dir}/folds.csv")
    train_metadata = train_metadata.merge(
        folds_df, on=["isic_id", "patient_id"], how="inner"
    )

    train_metadata = feature_engineering(train_metadata)
    if out_dim == 2:
        train_metadata["label"] = train_metadata["target"]
    elif out_dim == 9:
        train_metadata["label"] = train_metadata["iddx_3"].fillna("unknown")
        train_metadata["label"] = train_metadata["label"].replace(label_mapping["2024"])
        train_metadata["label"] = train_metadata["label"].map(label2idx)
    else:
        raise ValueError(f"Invalid out_dim: {out_dim}")

    if debug:
        train_metadata = train_metadata.sample(
            frac=0.05, random_state=seed
        ).reset_index(drop=True)

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
        train_metadata_2020 = feature_engineering(train_metadata_2020)
        if out_dim == 2:
            train_metadata_2020["label"] = np.where(
                train_metadata_2020["label"].isin(malignant_labels), 1, 0
            )
        elif out_dim == 9:
            train_metadata_2020["label"] = train_metadata_2020["label"].map(label2idx)
        else:
            raise ValueError(f"Invalid out_dim: {out_dim}")

        if debug:
            train_metadata_2020 = train_metadata_2020.sample(
                frac=0.05, random_state=seed
            ).reset_index(drop=True)
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
        train_metadata_2019 = feature_engineering(train_metadata_2019)
        if out_dim == 2:
            train_metadata_2019["label"] = np.where(
                train_metadata_2019["label"].isin(malignant_labels), 1, 0
            )
        elif out_dim == 9:
            train_metadata_2019["label"] = train_metadata_2019["label"].map(label2idx)
        else:
            raise ValueError(f"Invalid out_dim: {out_dim}")
        if debug:
            train_metadata_2019 = train_metadata_2019.sample(
                frac=0.05, random_state=seed
            ).reset_index(drop=True)
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
        malignant_idx,
    )
