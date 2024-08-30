# %% [code]
import time
import json
from pprint import pprint
from io import BytesIO

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from albumentations.pytorch import ToTensorV2
from PIL import Image
from timm import create_model
from torch.utils.data import DataLoader, Dataset


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


def test_augment_multi(image_size, mean=None, std=None):
    if mean is not None and std is not None:
        normalize = A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0)
    else:
        normalize = A.Normalize(max_pixel_value=255.0, p=1.0)
    transform = A.Compose(
        [A.Resize(image_size, image_size), normalize, ToTensorV2()], p=1.0
    )
    return transform


class ISICDatasetMulti(Dataset):
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


class ISICNetMulti(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=True,
    ):
        super(ISICNetMulti, self).__init__()
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


def predict_multi(model, test_dataloader, accelerator, n_tta, log_interval=10):
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


def run(
    test_metadata, test_images, model_name, version, model_dir, folds_to_run
):
    print(f"Predicting for {model_name}_{version}")
    start_time = time.time()
    with open(f"{model_dir}/{model_name}_{version}_run_metadata.json", "r") as f:
        run_metadata = json.load(f)
    pprint(run_metadata["params"])
    mixed_precision = run_metadata["params"]["mixed_precision"]
    image_size = run_metadata["params"]["image_size"]
    batch_size = run_metadata["params"]["val_batch_size"]
    n_tta = run_metadata["params"]["n_tta"]

    mean = None
    std = None

    test_dataset = ISICDatasetMulti(
        test_metadata,
        test_images,
        augment=test_augment_multi(image_size, mean=mean, std=std),
        infer=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
    )

    multi_test_probs = 0
    test_probs = 0
    for fold in folds_to_run:
        print(f"\nFold {fold}")
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
        )

        model = ISICNetMulti(
            model_name=model_name,
            pretrained=False,
        )
        model = model.to(accelerator.device)

        (
            model,
            test_dataloader,
        ) = accelerator.prepare(
            model,
            test_dataloader,
        )
        model_filepath = f"{model_dir}/models/fold_{fold}"
        accelerator.load_state(model_filepath)

        (multi_test_probs_fold, test_probs_fold) = predict_multi(
            model,
            test_dataloader,
            accelerator,
            n_tta,
        )
        if fold == 1:
            multi_test_probs = multi_test_probs_fold
            test_probs = test_probs_fold
        else:
            multi_test_probs += multi_test_probs_fold
            test_probs += test_probs_fold
    multi_test_probs /= len(folds_to_run)
    test_probs /= len(folds_to_run)
    oof_df = pd.DataFrame(
        {
            "isic_id": test_metadata["isic_id"],
            f"oof_{model_name}_{version}": test_probs.flatten(),
        }
    )
    oof_multi_df = pd.DataFrame(
        multi_test_probs,
        columns=[f"oof_{model_name}_{version}_{label}" for label in all_labels],
    )
    oof_df = pd.concat([oof_df, oof_multi_df], axis=1)
    runtime = time.time() - start_time
    print(f"Time taken: {runtime:.2f} s")
    print(f"Predictions generated for {model_name}_{version}")
    return oof_df, runtime
