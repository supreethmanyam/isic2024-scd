import os
import subprocess
import pathlib
import zipfile
import shutil
from typing import Tuple
from dataclasses import dataclass
from modal import Image, App, Secret, Volume, gpu


BASE_PATH = pathlib.Path("/kaggle/input/")
INPUT_PATH = BASE_PATH / "isic-2024-challenge"
REQUIRED_INPUT_FILES = [
    "train-image.hdf5",
    "train-metadata.csv",
    "test-image.hdf5",
    "test-metadata.csv",
    "sample_submission.csv",
    "folds.csv"
]


image = (
    Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("tree")
    .pip_install(
        "kaggle==1.6.17",
        "tqdm==4.66.4",
        "pillow==10.4.0",
        "h5py==3.11.0",
        "pandas==2.2.2",
        "torch==2.4.0",
        "albumentations==1.4.11",
        "timm==1.0.7",
        "accelerate==0.33.0"
    )
)
app = App(
    "isic2024-scd",
    image=image,
    secrets=[Secret.from_name("kaggle-api-token")],
)
input_volume = Volume.from_name("isic2024-scd-input", create_if_missing=True)
artifacts_volume = Volume.from_name("isic2024-scd-artifacts", create_if_missing=True)


def extract(fzip, dest, desc="Extracting", allowed_extensions: Tuple[str] = None):
    from tqdm.auto import tqdm
    from tqdm.utils import CallbackIOWrapper

    dest = pathlib.Path(dest).expanduser()
    with zipfile.ZipFile(fzip) as zipf, tqdm(
        desc=desc,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        total=sum(getattr(i, "file_size", 0) for i in zipf.infolist()),
    ) as pbar:
        for i in zipf.infolist():
            if allowed_extensions and not i.filename.endswith(allowed_extensions):
                continue
            if not getattr(i, "file_size", 0):  # directory
                zipf.extract(i, os.fspath(dest))
            else:
                full_path = dest / i.filename
                full_path.parent.mkdir(exist_ok=True, parents=True)
                with zipf.open(i) as fi, open(full_path, "wb") as fo:
                    shutil.copyfileobj(CallbackIOWrapper(pbar.update, fi), fo)


def download_data():
    # Download competition dataset
    subprocess.run(
        f"kaggle competitions download -c isic-2024-challenge --path {BASE_PATH}",
        shell=True,
        check=True,
    )
    # Extract competition dataset
    filepath = BASE_PATH / "isic-2024-challenge.zip"
    print(f"Extracting .zip into {INPUT_PATH}...")
    extract(filepath, INPUT_PATH, allowed_extensions=(".hdf5", ".csv"))
    print(f"Extracted {filepath} to {INPUT_PATH}")

    # Download folds dataset
    subprocess.run(
        f"kaggle kernels output supreethmanyam/isic-scd-folds --path {INPUT_PATH}",
        shell=True,
        check=True,
    )

    subprocess.run(f"tree -L 3 {INPUT_PATH}", shell=True, check=True)
    input_volume.commit()
    print("Dataset is downloaded ✅")


def get_data(reload: bool = False):
    existing_filenames = set(p.name for p in INPUT_PATH.iterdir())
    if reload:
        input_volume.reload()
    if len(existing_filenames) > 0 and set(REQUIRED_INPUT_FILES) <= existing_filenames:
        print("Dataset is loaded ✅")
    else:
        download_data()


@dataclass
class Config:
    model_name: str = "resnet18_v1"
    arch = "resnet18"
    mixed_precision: bool = "fp16"
    image_size: int = 128
    learning_rate: float = 5e-4
    batch_size: int = 256
    num_workers: int = 4
    num_epochs: int = 2
    tta: bool = True

    pos_ratio: float = 0.1

    seed: int = 2022


@app.function(
    volumes={"/kaggle/input/": input_volume, "/kaggle/working/": artifacts_volume},
    gpu=gpu.A10G(count=2),
    timeout=60 * 60 * 8,  # 8 hours
)
def train():
    import json
    from pprint import pprint
    from tqdm.auto import tqdm

    import pandas as pd
    import numpy as np
    from PIL import Image
    import h5py
    from io import BytesIO

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch.optim.lr_scheduler import OneCycleLR

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    from timm import create_model

    from accelerate import Accelerator

    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score as compute_auc

    kaggle_api_token_data = os.environ["KAGGLE_API_TOKEN"]
    kaggle_token_filepath = pathlib.Path.home() / ".config" / "kaggle" / "kaggle.json"
    kaggle_token_filepath.parent.mkdir(exist_ok=True, parents=True)
    kaggle_token_filepath.write_text(kaggle_api_token_data)
    subprocess.run(f"chmod 600 {kaggle_token_filepath}", shell=True, check=True)

    print("Downloading data if volume doesn't exist...")
    INPUT_PATH.mkdir(parents=True, exist_ok=True)
    get_data()

    cfg = Config()

    MODELS_OUTPUT_PATH = pathlib.Path("/kaggle/working") / cfg.model_name
    MODELS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    train_metadata = pd.read_csv(INPUT_PATH / "train-metadata.csv", low_memory=False)
    train_images = h5py.File(INPUT_PATH / "train-image.hdf5", mode="r")

    folds_df = pd.read_csv(INPUT_PATH / "folds.csv")
    train_metadata = train_metadata.merge(folds_df, on=["isic_id", "patient_id"], how="inner")
    train_metadata = train_metadata.sample(frac=0.05, random_state=cfg.seed).reset_index(drop=True)
    print(f"Train data size: {train_metadata.shape}")
    print(torch.cuda.device_count())


    def set_seed(seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        

    def compute_pauc(y_true, y_pred, min_tpr: float=0.80) -> float:
        '''
        2024 ISIC Challenge metric: pAUC
        
        Given a solution file and submission file, this function returns the
        the partial area under the receiver operating characteristic (pAUC) 
        above a given true positive rate (TPR) = 0.80.
        https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.
        
        (c) 2024 Nicholas R Kurtansky, MSKCC

        Args:
            y_true: ground truth of 1s and 0s
            y_pred: predictions of scores ranging [0, 1]

        Returns:
            Float value range [0, max_fpr]
        '''

        # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
        v_gt = abs(y_true-1)
        
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
        
        return(partial_auc)


    def dev_augment(image_size):
        transform = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            A.OneOf([
                A.MotionBlur(blur_limit=(5, 7)),
                A.MedianBlur(blur_limit=(5, 7)),
                A.GaussianBlur(blur_limit=(5, 7)),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=0.7),

            A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.Resize(image_size, image_size),
    #         A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
            ToTensorV2()
        ], p=1.)
        return transform


    def val_augment(image_size):
        transform = A.Compose([
            A.Resize(image_size, image_size),
    #         A.Normalize(
    #             mean=[0., 0., 0.],
    #             std=[1, 1, 1],
    #             max_pixel_value=255.0,
    #             p=1.0
    #         ),
            ToTensorV2()
        ], p=1.)
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
            
            image = np.array(Image.open(BytesIO(self.images[data[id_column]][()])))
            image = self.augment(image=image)["image"]
            
            record = {
                "image": image
            }
            
            if not self.infer:
                target = data[target_column]
                record["target"] = torch.tensor(target).float()
            
            return record


    class ISICNet(nn.Module):
        def __init__(self, arch, pretrained=False, infer=False):
            super(ISICNet, self).__init__()
            self.infer = infer
            self.model = create_model(model_name=arch, pretrained=pretrained, in_chans=3,  num_classes=0, global_pool='')
            self.classifier = nn.Linear(self.model.num_features, 1)
            
            self.dropouts = nn.ModuleList([nn.Dropout(0.5) for i in range(5)])
            
        def forward(self, batch):
            image = batch["image"]
            image = image.float() / 255
            
            x = self.model(image)
            bs = len(image)
            pool = F.adaptive_avg_pool2d(x, 1).reshape(bs,-1)
            
            if self.training:
                logit = 0
                for i in range(len(self.dropouts)):
                    logit += self.classifier(self.dropouts[i](pool))
                logit = logit/len(self.dropouts)
            else:
                logit = self.classifier(pool)
            return logit


    def make_over_sample(train_index, target, pos_ratio):
        target_series = pd.Series(y_train, index=train_index)
        
        # Separate positive and negative indices
        pos_indices = target_series[target_series == 1].index
        neg_indices = target_series[target_series == 0].index
        
        # Calculate the number of positive samples needed
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        n_total = n_pos + n_neg
        n_desired_pos = int((pos_ratio * n_total) / (1 - pos_ratio))
        
        # Oversample positive indices
        if n_desired_pos > n_pos:
            pos_indices_oversampled = np.random.choice(pos_indices, size=n_desired_pos, replace=True)
        else:
            pos_indices_oversampled = pos_indices
        
        ned_indices_undersampled = np.unique(np.random.choice(neg_indices, size=n_desired_pos * 3, replace=True))
        
        # Combine with negative indices
        oversampled_indices = np.concatenate([ned_indices_undersampled, pos_indices_oversampled])
        np.random.shuffle(oversampled_indices)
        return oversampled_indices

    id_column = "isic_id"
    target_column = "target"
    group_column = "patient_id"
    
    train_ids = train_metadata[id_column]
    groups= train_metadata[group_column]
    folds = train_metadata["fold"]
    y_train = train_metadata[target_column]

    best_num_epochs = {}
    val_auc_scores = {}
    val_pauc_scores = {}
    all_folds = np.unique(train_metadata["fold"])
    oof_predictions = np.zeros(train_metadata.shape[0])

    for fold in all_folds:
        accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
        
        set_seed(cfg.seed)
        
        print(f"Running fold: {fold}")
        dev_index = folds[folds != fold].index
        val_index = folds[folds == fold].index
        
        oversampled_dev_index = make_over_sample(dev_index, y_train, cfg.pos_ratio)
        
        dev_metadata = train_metadata.loc[oversampled_dev_index, :].reset_index(drop=True)
        val_metadata = train_metadata.loc[val_index, :].reset_index(drop=True)
        
        dev_dataset = ISICDataset(dev_metadata, train_images, augment=dev_augment(image_size=cfg.image_size))
        val_dataset = ISICDataset(val_metadata, train_images, augment=val_augment(image_size=cfg.image_size))

        dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers, drop_last=False, pin_memory=True)
        
        net = ISICNet(arch=cfg.arch, pretrained=True)
        net = net.to(accelerator.device)
        print(f"Arch: {cfg.arch}, Training on: {accelerator.device}")
        print(f"Number of devices: {accelerator.num_processes}")
            
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate / 5)
        lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=cfg.learning_rate, epochs=cfg.num_epochs, steps_per_epoch=len(dev_dataloader))

        net, optimizer, dev_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            net, optimizer, dev_dataloader, val_dataloader, lr_scheduler
        )
        
        starting_epoch = 0
        best_pauc_score = -np.Inf
        best_auc_score = -np.Inf
        best_epoch = None
        best_val_preds = None

        for epoch in range(starting_epoch, cfg.num_epochs):
            net.train()
            for step, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = net(batch)
                loss = F.binary_cross_entropy_with_logits(outputs, batch["target"].unsqueeze(1))
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()

            net.eval()
            val_preds = []
            val_y = []
            for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                image0 = batch['image'].clone().detach()
                val_preds_batch = 0
                counter = 0
                with torch.no_grad():
                    outputs = net(batch)
                preds = torch.sigmoid(outputs)
                val_y_batch = batch["target"]
                preds, val_y_batch = accelerator.gather_for_metrics((preds, val_y_batch))
                val_preds_batch += preds.data.cpu().numpy().reshape(-1)
                counter += 1
                if cfg.tta:
                    batch["image"] = torch.flip(image0,dims=[2])
                    with torch.no_grad():
                        outputs = net(batch)
                    preds = torch.sigmoid(outputs)
                    preds = accelerator.gather_for_metrics((preds))
                    val_preds_batch += preds.data.cpu().numpy().reshape(-1)
                    counter += 1
                    
                    batch["image"] = torch.flip(image0,dims=[3])
                    with torch.no_grad():
                        outputs = net(batch)
                    preds = torch.sigmoid(outputs)
                    preds = accelerator.gather_for_metrics((preds))
                    val_preds_batch += preds.data.cpu().numpy().reshape(-1)
                    counter += 1
                    
                    for k in [1, 2, 3]:
                        batch["image"] = torch.rot90(image0,k, dims=[2, 3])
                        with torch.no_grad():
                            outputs = net(batch)
                        preds = torch.sigmoid(outputs)
                        preds = accelerator.gather_for_metrics((preds))
                        val_preds_batch += preds.data.cpu().numpy().reshape(-1)
                        counter += 1
                val_preds_batch = val_preds_batch / counter   
                val_preds.append(val_preds_batch)    
                val_y.append(val_y_batch.data.cpu().numpy().reshape(-1))
                
            val_preds = np.concatenate(val_preds)
            val_y = np.concatenate(val_y)
            epoch_auc = compute_auc(val_y, val_preds) 
            epoch_pauc = compute_pauc(val_y, val_preds, min_tpr=0.80)
            
            if epoch_pauc >= best_pauc_score:
                best_auc_score = epoch_auc
                best_pauc_score = epoch_pauc
                best_epoch = epoch
                best_val_preds = val_preds
            print(f"Epoch pauc: {epoch_pauc} | Best auc: {best_auc_score} | Best pauc: {best_pauc_score} | Best epoch: {best_epoch}")
            
            output_dir = MODELS_OUTPUT_PATH / f"models/fold_{fold}/epoch_{epoch}"
            accelerator.save_state(output_dir)
        
        best_num_epochs[f"fold_{fold}"] = best_epoch
        val_auc_scores[f"fold_{fold}"] = best_auc_score
        val_pauc_scores[f"fold_{fold}"] = best_pauc_score
        
        oof_predictions[val_index] = best_val_preds
        print("\n")
        break
    
    print("Val AUC scores:")
    pprint(val_auc_scores)
    print("Val PAUC scores:")
    pprint(val_pauc_scores)
    
    oof_preds_df = pd.DataFrame({
        id_column: train_ids,
        group_column: groups,
        "fold": folds,
        target_column: y_train,
        f"oof_{cfg.model_name}": oof_predictions
    })
    oof_preds_df.to_csv(MODELS_OUTPUT_PATH / f"oof_preds_{cfg.model_name}.csv", index=False)

    cv_auc_oof = compute_auc(oof_preds_df[target_column], oof_preds_df[f"oof_{cfg.model_name}"])
    cv_pauc_oof = compute_pauc(oof_preds_df[target_column], oof_preds_df[f"oof_{cfg.model_name}"], min_tpr=0.8)

    cv_auc_avg = np.mean(list(val_auc_scores.values()))
    cv_pauc_avg = np.mean(list(val_pauc_scores.values()))

    cv_auc_std = np.std(list(val_auc_scores.values()))
    cv_pauc_std = np.std(list(val_pauc_scores.values()))

    print(f"CV AUC OOF: {cv_auc_oof}")
    print(f"CV PAUC OOF: {cv_pauc_oof}")
    print(f"CV AUC AVG: {cv_auc_avg}")
    print(f"CV PAUC AVG: {cv_pauc_avg}")
    print(f"CV AUC STD: {cv_auc_std}")
    print(f"CV PAUC STD: {cv_pauc_std}")

    print(f"Best number of epochs: {best_num_epochs}")

    metadata = {
        "params": cfg.__dict__,
        "best_num_epochs": best_num_epochs,
        "val_auc_scores": val_auc_scores,
        "val_pauc_scores": val_pauc_scores,
        "cv_auc_oof": cv_auc_oof,
        "cv_pauc_oof": cv_pauc_oof,
        "cv_auc_avg": cv_auc_avg,
        "cv_pauc_avg": cv_pauc_avg
    }

    with open(MODELS_OUTPUT_PATH / f"{cfg.model_name}_run_metadata.json", "w") as f:
        json.dump(metadata, f)

    print("Training is done ✅")
    print("Uploading weights to Kaggle...")
    subprocess.run(f"kaggle datasets init -p {MODELS_OUTPUT_PATH}", shell=True, check=True)
    with open(MODELS_OUTPUT_PATH / "dataset-metadata.json", "r") as f:
        metadata = json.load(f)
        model_name, version = cfg.model_name.split("_")
        title_part = f"{model_name.upper()}_{version}"
        metadata["title"] = f"ISIC_SCD_{title_part}_TRAIN"
        metadata["id"] = f"supreethmanyam/{metadata['title'].lower().replace('_', '-')}"
    with open(MODELS_OUTPUT_PATH / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f)

    search_results = subprocess.run(f"kaggle datasets list -m -v -s {metadata['title']}", shell=True, check=True, capture_output=True)
    search_results_df = pd.read_csv(BytesIO(search_results.stdout))
    if search_results_df.empty:
        print("Creating new dataset...")
        subprocess.run(f"kaggle datasets create -p {MODELS_OUTPUT_PATH} --dir-mode tar", shell=True, check=True)
    else:
        print("Updating existing dataset...")
        version_string = f"CV AUC: {cv_auc_avg:.4f}, CV pAUC: {cv_pauc_avg:.4f}"
        subprocess.run(f"kaggle datasets version -p {MODELS_OUTPUT_PATH} -m '{version_string}' --dir-mode tar", shell=True, check=True)
    print("Weights uploaded to Kaggle ✅")
    artifacts_volume.commit()
