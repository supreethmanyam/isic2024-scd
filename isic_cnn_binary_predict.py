# %% [code]
import time
import json
from typing import List, Dict
from pprint import pprint
from collections import defaultdict
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


def get_emb_szs(cat_cols):
    emb_szs = {}
    for col in cat_cols:
        emb_szs[col] = (
            len(feature_mapping_dict[col]),
            min(600, round(1.6 * len(feature_mapping_dict[col]) ** 0.56)),
        )
    return emb_szs


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
            target = torch.tensor(row["target"])
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


def run(
    test_metadata, test_images, model_name, version, model_dir, folds_to_run, cat_cols, cont_cols
):
    print(f"Predicting for {model_name}_{version}")
    start_time = time.time()
    with open(f"{model_dir}/{model_name}_{version}_run_metadata.json", "r") as f:
        run_metadata = json.load(f)
    pprint(run_metadata["params"])
    mixed_precision = run_metadata["params"]["mixed_precision"]
    image_size = run_metadata["params"]["image_size"]
    batch_size = run_metadata["params"]["val_batch_size"]
    use_meta = run_metadata["params"]["use_meta"]
    n_tta = run_metadata["params"]["n_tta"]

    emb_szs = get_emb_szs(cat_cols)

    mean = None
    std = None

    test_dataset = ISICDatasetBinary(
        test_metadata,
        test_images,
        augment=test_augment_binary(image_size, mean=mean, std=std),
        use_meta=use_meta,
        cat_cols=cat_cols,
        cont_cols=cont_cols,
        infer=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
    )

    test_probs = 0
    for fold in folds_to_run:
        print(f"\nFold {fold}")
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
        )

        model = ISICNetBinary(
            model_name=model_name,
            pretrained=False,
            use_meta=use_meta,
            cat_cols=cat_cols,
            cont_cols=cont_cols,
            emb_szs=emb_szs,
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

        test_probs_fold = predict_binary(
            model,
            test_dataloader,
            accelerator,
            n_tta,
            use_meta,
        )
        if fold == 1:
            test_probs = test_probs_fold
        else:
            test_probs += test_probs_fold
    test_probs /= len(folds_to_run)
    oof_df = pd.DataFrame(
        {
            "isic_id": test_metadata["isic_id"],
            f"oof_{model_name}_{version}": test_probs.flatten(),
        }
    )
    runtime = time.time() - start_time
    print(f"Time taken: {runtime:.2f} s")
    print(f"Predictions generated for {model_name}_{version}")
    return oof_df, runtime
