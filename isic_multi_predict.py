# %% [code]
import argparse
import logging
from accelerate.logging import get_logger
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from timm import create_model
from accelerate import Accelerator


logger = get_logger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

id_column = "isic_id"
target_column = "target"
group_column = "patient_id"
fold_column = "fold"

INPUT_PATH = Path("/kaggle/input/isic-2024-challenge/")
FOLDS_PATH = Path("/kaggle/input/isic-scd-folds")

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


def get_data():
    train_metadata = pd.read_csv(INPUT_PATH / "train-metadata.csv", low_memory=False)
    train_images = h5py.File(INPUT_PATH / "train-image.hdf5", mode="r")
    test_metadata = pd.read_csv(INPUT_PATH / "test-metadata.csv", low_memory=False)
    test_images = h5py.File(INPUT_PATH / "test-image.hdf5", mode="r")

    folds_df = pd.read_csv(FOLDS_PATH / "folds.csv")
    train_metadata = train_metadata.merge(
        folds_df, on=["isic_id", "patient_id"], how="inner"
    )
    return train_metadata, train_images, test_metadata, test_images


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


class ISICNetV1(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=True,
    ):
        super(ISICNetV1, self).__init__()
        self.model = create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="",
        )
        in_dim = self.model.num_features
        self.classifier = nn.Linear(in_dim, len(all_labels))
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

    def forward(self, images):
        x = self.model(images)
        bs = len(images)
        pool = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        if self.training:
            logits = 0
            for i in range(len(self.dropouts)):
                logits += self.classifier(self.dropouts[i](pool))
            logits = logits / len(self.dropouts)
        else:
            logits = self.classifier(pool)
        return logits


def predict_v1(model, test_dataloader, accelerator, n_tta, log_interval=10):
    model.eval()
    multi_test_probs = []
    total_steps = len(test_dataloader)
    out_dim = len(all_labels)
    with torch.no_grad():
        for step, images in enumerate(test_dataloader):
            logits = torch.zeros((images.shape[0], out_dim)).to(accelerator.device)
            probs = torch.zeros((images.shape[0], out_dim)).to(accelerator.device)
            for i in range(n_tta):
                logits_iter = model(get_trans(images, i))
                logits += logits_iter
                probs += logits_iter.softmax(1)
            logits /= n_tta
            probs /= n_tta

            probs = accelerator.gather(probs)
            multi_test_probs.append(probs)

            if (step == 0) or ((step + 1) % log_interval == 0):
                print(f"Step: {step + 1}/{total_steps}")

    multi_test_probs = torch.cat(multi_test_probs).cpu().numpy()
    test_probs = multi_test_probs[:, malignant_idx].sum(1)
    return multi_test_probs, test_probs


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Predict ISIC2024 SCD")
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
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="The directory where the data is stored.",
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
    (
        train_metadata, train_images, 
        test_metadata, test_images
    ) = get_data_v1()

    fold_column = "fold"
    mean = None
    std = None
    
    test_dataset = ISICDatasetV1(
        test_metadata, 
        test_images, 
        augment=test_augment_v1(args.image_size, mean=mean, std=std),
        infer=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
    )
    all_folds = np.unique(train_metadata[fold_column])
    for fold in all_folds:
        start_time = time.time()
        logger.info(f"Fold {args.fold}")
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
        )

        model = ISICNetV1(
            model_name=args.model_name,
            pretrained=True,
        )
        model = model.to(accelerator.device)

        (
            model,
            test_dataloader,
        ) = accelerator.prepare(
            model, test_dataloader,
        )
        model_filepath = f"{args.model_dir}/models/fold_{args.fold}"
        accelerator.load_state(model_filepath)
        (
            multi_test_probs_fold,
            test_probs_fold
        ) = predict_v1(
            model,
            test_dataloader,
            accelerator,
            args.n_tta,
        )
        if fold == 1:
            multi_test_probs = multi_test_probs_fold
            test_probs = test_probs_fold
        else:
            multi_test_probs += multi_test_probs_fold
            test_probs += test_probs_fold
    multi_test_probs /= len(all_folds)
    test_probs /= len(all_folds)
    oof_df = pd.DataFrame(
        {
            "isic_id": test_metadata["isic_id"],
            f"oof_{args.model_name}_{args.version}": test_probs.flatten(),
        }
    )
    oof_multi_df = pd.DataFrame(multi_test_probs,
                                columns=[f"oof_{args.model_name}_{args.version}_{label}" for label in all_labels])
    oof_df = pd.concat([oof_df, oof_multi_df], axis=1)
    oof_df.to_csv(
        f"oof_test_preds_{args.model_name}_{args.version}.csv",
        index=False,
    )
    runtime = time.time() - start_time
    logger.info(f"Time taken: {runtime:.2f} s")
    logger.info(f"Finished predicting")