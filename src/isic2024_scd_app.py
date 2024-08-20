import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from modal import App, Image, Mount, Secret, Volume

GPU_CONFIG = os.environ.get("GPU_CONFIG", "a10g:1")
app = App(name="isic2024-scd", secrets=[Secret.from_name("kaggle-api-token")])
image = (
    Image.debian_slim(python_version="3.10")
    .poetry_install_from_file(
        poetry_pyproject_toml="../pyproject.toml",
        poetry_lockfile="../poetry.lock",
        only=["main"],
    )
    .run_commands(
        "apt-get update",
        # Required to install libs such as libGL.so.1
        "apt-get install ffmpeg libsm6 libxext6 tree --yes",
    )
)

weights_volume = Volume.from_name("isic2024-scd-weights", create_if_missing=True)
WEIGHTS_DIR = Path("/kaggle/working")

input_volume = Volume.from_name("isic2024-scd-input", create_if_missing=True)
INPUT_DIR = Path("/kaggle/input")


def setup_kaggle():
    import subprocess

    kaggle_api_token_data = os.environ["KAGGLE_API_TOKEN"]
    kaggle_token_filepath = Path.home() / ".config" / "kaggle" / "kaggle.json"
    kaggle_token_filepath.parent.mkdir(exist_ok=True, parents=True)
    kaggle_token_filepath.write_text(kaggle_api_token_data)
    subprocess.run(f"chmod 600 {kaggle_token_filepath}", shell=True, check=True)


def extract(fzip, dest, desc="Extracting", allowed_extensions: Tuple[str] = None):
    from tqdm.auto import tqdm
    from tqdm.utils import CallbackIOWrapper

    dest = Path(dest).expanduser()
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


def download_competition_data(path: Path, recreate: bool = False):
    import subprocess

    required_files = [
        "train-image.hdf5",
        "train-metadata.csv",
        "test-image.hdf5",
        "test-metadata.csv",
        "sample_submission.csv",
        "folds.csv",
    ]
    existing_files = set(file.name for file in path.iterdir())
    if (
        not recreate
        and len(existing_files) > 0
        and set(required_files) <= existing_files
    ):
        print("Competition dataset is loaded ✅")
    else:
        # Download competition dataset
        subprocess.run(
            f"kaggle competitions download -c isic-2024-challenge --path {path}",
            shell=True,
            check=True,
        )
        # Extract competition dataset
        filepath = path / "isic-2024-challenge.zip"
        print(f"Extracting .zip into {path}...")
        extract(filepath, path, allowed_extensions=(".hdf5", ".csv"))
        print(f"Extracted {filepath} to {path}")

        # Download folds dataset
        subprocess.run(
            f"kaggle kernels output supreethmanyam/isic-scd-folds --path {path}",
            shell=True,
            check=True,
        )

        subprocess.run(f"tree -L 3 {path}", shell=True, check=True)
        input_volume.commit()
        print("Downloaded competition dataset from kaggle ✅")


external_data_mapping = {
    "2020": {"id": 70, "path": INPUT_DIR / "isic-2020-challenge"},
    "2019": {"id": 65, "path": INPUT_DIR / "isic-2019-challenge"},
}


def center_crop_and_resize(img, resize_to):
    from PIL import Image as PILImage

    size = min(img.size)
    offset0 = (img.size[0] - size) // 2
    offset1 = (img.size[1] - size) // 2
    cropped_img = img.crop((offset0, offset1, offset0 + size, offset1 + size))

    resized_img = cropped_img.resize(
        (resize_to, resize_to), PILImage.Resampling.LANCZOS
    )
    return resized_img


def process_file(file):
    import numpy as np
    from PIL import Image as PILImage

    extensions = ["jpg", "png", "bmp", "jpeg"]
    filename = os.path.splitext(os.path.basename(file))[0]
    ext = os.path.splitext(file)[1][1:].lower()
    if ext not in extensions:
        print("%s does not have a supported extension. Skipping!!" % file)
        return
    else:
        tmp = PILImage.open(file)
        tmp = center_crop_and_resize(tmp, resize_to=256)
        tmp.save("temp.jpg", "jpeg", quality=100)
        fin = open("temp.jpg", "rb")
        binary_data = fin.read()
        binary_data_np = np.asarray(binary_data)
        fin.close()
        return filename, binary_data_np


def prepare_external_data(images_dir: Path, data_dir: Path):
    import os
    from glob import glob

    import h5py
    import pandas as pd
    from tqdm import tqdm

    direc = str(images_dir)
    flist = glob(os.path.join(direc, "*"))

    f = h5py.File(data_dir / "train-image.hdf5", "w")
    for file in tqdm(flist):
        result = process_file(file)
        if result is not None:
            image_filename, image_data = result
            f.create_dataset(image_filename, data=image_data)
    f.close()

    # Read the metadata file
    metadata = pd.read_csv(images_dir / "metadata.csv", low_memory=False)
    metadata.to_csv(data_dir / "train-metadata.csv", index=False)


def download_external_data(year: str, recreate: bool = False):
    import subprocess

    path = external_data_mapping[year]["path"]
    if recreate and path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    images_path = path / "images"

    required_files = [
        "train-image.hdf5",
        "train-metadata.csv",
    ]
    existing_files = set(file.name for file in path.iterdir())
    if len(existing_files) > 0 and set(required_files) <= existing_files:
        print(f"External dataset {year} is loaded ✅")
    else:
        # Download external dataset
        print(f"Downloading external dataset {year}...")
        subprocess.run(
            f"isic image download -c {external_data_mapping[year]['id']} {images_path}",
            shell=True,
            check=True,
        )
        prepare_external_data(images_path, path)
        shutil.rmtree(images_path)
        input_volume.commit()
        print(f"Downloaded external dataset {year} from kaggle ✅")


@app.function(
    image=image,
    volumes={str(INPUT_DIR): input_volume},
    timeout=60 * 60 * 6,  # 6 hours
)
def download_data(ext: str = "", recreate: bool = False):
    setup_kaggle()
    data_dir = INPUT_DIR / "isic-2024-challenge"
    data_dir.mkdir(parents=True, exist_ok=True)
    download_competition_data(data_dir)
    if "2020" in ext:
        download_external_data("2020", recreate)
    if "2019" in ext:
        download_external_data("2019", recreate)
    input_volume.commit()


@app.function(
    image=image,
    volumes={str(INPUT_DIR): input_volume},
    timeout=60 * 60,  # 1 hour
)
def upload_external_data(year: str):
    import json
    import subprocess
    from io import BytesIO

    import pandas as pd

    setup_kaggle()
    data_dir = external_data_mapping[year]["path"]
    subprocess.run(f"kaggle datasets init -p {data_dir}", shell=True, check=True)
    with open(data_dir / "dataset-metadata.json", "r") as f:
        metadata = json.load(f)
        title_part = f"ISIC_{year}_CHALLENGE"
        metadata["title"] = f"{title_part}"
        metadata["id"] = f"supreethmanyam/{metadata['title'].lower().replace('_', '-')}"
    with open(data_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f)

    search_results = subprocess.run(
        f"kaggle datasets list -m -v -s {metadata['title']}",
        shell=True,
        check=True,
        capture_output=True,
    )
    search_results_df = pd.read_csv(BytesIO(search_results.stdout))
    if search_results_df.empty:
        print("Creating new dataset...")
        subprocess.run(
            f"kaggle datasets create -p {data_dir} --dir-mode tar",
            shell=True,
            check=True,
        )
    else:
        print("Updating existing dataset...")
        subprocess.run(
            f"kaggle datasets version -p {data_dir} --dir-mode tar",
            shell=True,
            check=True,
        )
    print(f"External data for {year} uploaded to Kaggle ✅")


def mount_folder(folder_name: str):
    script_local_path = Path(__file__).parent / folder_name
    script_remote_path = Path(f"/root/{folder_name}")
    return Mount.from_local_dir(
        local_path=script_local_path, remote_path=script_remote_path
    )


@dataclass
class PreTrainConfig:
    mode: str = "pretrain"
    mixed_precision: bool = "fp16"
    image_size: int = 128
    train_batch_size: int = 64
    val_batch_size: int = 512
    num_workers: int = 8
    init_lr: float = 3e-5
    num_epochs: int = 20
    n_tta: int = 8
    down_sampling: bool = True
    use_meta: bool = True
    seed: int = 2022

    debug: bool = False


@dataclass
class FinetuneConfig:
    mixed_precision: bool = "fp16"
    image_size: int = 64
    train_batch_size: int = 64
    val_batch_size: int = 512
    num_workers: int = 8
    init_lr: float = 8e-5
    num_epochs: int = 20
    n_tta: int = 8
    fold_method: str = "gkf"
    seed: int = 2022

    sampling_rate: float = 0.1
    tau: float = 0.1
    Lambda: float = 1.0
    gamma0: float = 0.5
    gamma1: float = 0.5
    margin: float = 1.0

    debug: bool = False


@app.function(
    image=image,
    mounts=[mount_folder("pretrain")],
    volumes={str(INPUT_DIR): input_volume, str(WEIGHTS_DIR): weights_volume},
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 24,  # 24 hours
    cpu=PreTrainConfig.num_workers,
)
def pretrain(model_name: str, version: str, fold: int):
    import json
    import subprocess
    from pprint import pprint

    from accelerate.utils import write_basic_config

    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    setup_kaggle()
    config = PreTrainConfig()
    print(f"Running with config:")
    pprint(config.__dict__)
    metadata = {"params": config.__dict__}
    model_identifier = f"{model_name}_{version}_pretrain"
    model_dir = Path(WEIGHTS_DIR) / model_identifier
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / f"{model_name}_{version}_run_metadata.json", "w") as f:
        json.dump(metadata, f)

    data_dir = INPUT_DIR / "isic-2024-challenge"

    write_basic_config(mixed_precision=config.mixed_precision)
    print("Launching training script")
    commands = (
        [
            "accelerate",
            "launch",
            "pretrain/run.py",
            f"--model_name={model_name}",
            f"--version={version}",
            f"--model_dir={model_dir}",
            f"--data_dir={data_dir}",
        ]
        + [
            f"--fold={fold}",
        ]
        + [
            f"--mixed_precision={config.mixed_precision}",
            f"--image_size={config.image_size}",
            f"--train_batch_size={config.train_batch_size}",
            f"--val_batch_size={config.val_batch_size}",
            f"--num_workers={config.num_workers}",
            f"--init_lr={config.init_lr}",
            f"--num_epochs={config.num_epochs}",
            f"--n_tta={config.n_tta}",
            f"--seed={config.seed}",
        ]
        + (["--use_meta"] if config.use_meta else [])
        + (["--down_sampling"] if config.down_sampling else [])
        + (["--debug"] if config.debug else [])
    )
    print(subprocess.list2cmdline(commands))
    _exec_subprocess(commands)
    weights_volume.commit()


@app.function(
    image=image,
    mounts=[mount_folder("finetune")],
    volumes={str(INPUT_DIR): input_volume, str(WEIGHTS_DIR): weights_volume},
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 24,  # 24 hours
    cpu=FinetuneConfig.num_workers,
)
def finetune(model_name: str, version: str, fold: int):
    import json
    import subprocess
    from pprint import pprint

    from accelerate.utils import write_basic_config

    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    setup_kaggle()
    config = FinetuneConfig()
    print(f"Running with config:")
    pprint(config.__dict__)
    metadata = {"params": config.__dict__}
    model_identifier = f"{model_name}_{version}_finetune"
    model_dir = Path(WEIGHTS_DIR) / model_identifier
    model_dir.mkdir(parents=True, exist_ok=True)
    pretrained_model_dir = Path(WEIGHTS_DIR) / f"{model_name}_{version}_pretrain"
    with open(model_dir / f"{model_name}_{version}_run_metadata.json", "w") as f:
        json.dump(metadata, f)

    data_dir = INPUT_DIR / "isic-2024-challenge"

    write_basic_config(mixed_precision=config.mixed_precision)
    print("Launching training script")
    commands = (
        [
            "accelerate",
            "launch",
            "finetune/run.py",
            f"--model_name={model_name}",
            f"--version={version}",
            f"--pretrained_model_dir={pretrained_model_dir}",
            f"--model_dir={model_dir}",
            f"--data_dir={data_dir}",
            f"--fold_method={config.fold_method}",
        ]
        + [
            f"--fold={fold}",
        ]
        + [
            f"--mixed_precision={config.mixed_precision}",
            f"--image_size={config.image_size}",
            f"--train_batch_size={config.train_batch_size}",
            f"--val_batch_size={config.val_batch_size}",
            f"--num_workers={config.num_workers}",
            f"--init_lr={config.init_lr}",
            f"--sampling_rate={config.sampling_rate}",
            f"--tau={config.tau}",
            f"--Lambda={config.Lambda}",
            f"--gamma0={config.gamma0}",
            f"--gamma1={config.gamma1}",
            f"--margin={config.margin}",
            f"--num_epochs={config.num_epochs}",
            f"--n_tta={config.n_tta}",
            f"--seed={config.seed}",
        ]
        + (["--debug"] if config.debug else [])
    )
    print(subprocess.list2cmdline(commands))
    _exec_subprocess(commands)
    weights_volume.commit()


@app.function(
    image=image,
    mounts=[mount_folder("pretrain"), mount_folder("finetune")],
    volumes={str(WEIGHTS_DIR): weights_volume},
    timeout=60 * 60,  # 1 hour
)
def upload_weights(model_name: str, version: str, mode: str | None = None):
    import json
    import subprocess
    from glob import glob
    from io import BytesIO
    from pprint import pprint

    import numpy as np
    import pandas as pd
    from pretrain.utils import compute_auc, compute_pauc

    setup_kaggle()

    if mode not in ["pretrain", "finetune"]:
        raise ValueError("Value of mode must be one of ['pretrain', 'finetune']")
    model_identifier = f"{model_name}_{version}_{mode}"
    model_dir = Path(WEIGHTS_DIR) / model_identifier

    oof_preds_fold_filepaths = glob(
        str(model_dir / f"oof_preds_{model_name}_{version}_fold_*.csv")
    )
    oof_preds_df = pd.concat(
        [pd.read_csv(filepath) for filepath in oof_preds_fold_filepaths],
        ignore_index=True,
    )
    oof_preds_df.to_csv(model_dir / f"oof_preds_{model_name}_{version}.csv", index=False)

    all_folds = np.unique(oof_preds_df["fold"])
    val_auc_scores = {}
    val_pauc_scores = {}
    best_num_epochs = {}
    val_epoch_paucs = {}
    val_epoch_aucs = {}
    for fold in all_folds:
        val_index = oof_preds_df[oof_preds_df["fold"] == fold].index

        val_auc_scores[f"fold_{fold}"] = compute_auc(
            oof_preds_df.loc[val_index, "target"],
            oof_preds_df.loc[val_index, f"oof_{model_name}_{version}"],
        )
        val_pauc_scores[f"fold_{fold}"] = compute_pauc(
            oof_preds_df.loc[val_index, "target"],
            oof_preds_df.loc[val_index, f"oof_{model_name}_{version}"],
            min_tpr=0.8,
        )

        with open(model_dir / f"models/fold_{fold}/metadata.json", "r") as f:
            fold_metadata = json.load(f)
            best_num_epochs[f"fold_{fold}"] = fold_metadata["best_epoch"]
            val_epoch_paucs[f"fold_{fold}"] = fold_metadata["val_paucs"]
            val_epoch_aucs[f"fold_{fold}"] = fold_metadata["val_aucs"]
        assert np.allclose(
            val_pauc_scores[f"fold_{fold}"], fold_metadata["best_val_pauc"]
        )
    print("Val AUC scores:")
    pprint(val_auc_scores)
    print("Val PAUC scores:")
    pprint(val_pauc_scores)

    cv_auc_oof = compute_auc(
        oof_preds_df["target"], oof_preds_df[f"oof_{model_name}_{version}"]
    )
    cv_pauc_oof = compute_pauc(
        oof_preds_df["target"], oof_preds_df[f"oof_{model_name}_{version}"], min_tpr=0.8
    )

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

    metrics_metadata = {
        "best_num_epochs": best_num_epochs,
        "val_auc_scores": val_auc_scores,
        "val_pauc_scores": val_pauc_scores,
        "cv_auc_oof": cv_auc_oof,
        "cv_pauc_oof": cv_pauc_oof,
        "cv_auc_avg": cv_auc_avg,
        "cv_pauc_avg": cv_pauc_avg,
        "cv_auc_std": cv_auc_std,
        "cv_pauc_std": cv_pauc_std,
        "val_epoch_paucs": val_epoch_paucs,
        "val_epoch_aucs": val_epoch_aucs,
    }

    with open(model_dir / f"{model_name}_{version}_run_metadata.json", "r") as f:
        metadata = json.load(f)
        metadata = {**metadata, **metrics_metadata}
    with open(model_dir / f"{model_name}_{version}_run_metadata.json", "w") as f:
        json.dump(metadata, f)

    shutil.copy(f"{mode}/models.py", model_dir)
    shutil.copy(f"{mode}/dataset.py", model_dir)
    shutil.copy(f"{mode}/engine.py", model_dir)
    shutil.copy(f"{mode}/utils.py", model_dir)

    subprocess.run(f"kaggle datasets init -p {model_dir}", shell=True, check=True)
    with open(model_dir / "dataset-metadata.json", "r") as f:
        metadata = json.load(f)
        title_part = f"{model_name.upper()}_{version}"
        metadata["title"] = f"ISIC_SCD_{title_part}_{mode.upper()}"
        metadata["id"] = f"supreethmanyam/{metadata['title'].lower().replace('_', '-')}"
    with open(model_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f)

    search_results = subprocess.run(
        f"kaggle datasets list -m -v -s {metadata['title']}",
        shell=True,
        check=True,
        capture_output=True,
    )
    search_results_df = pd.read_csv(BytesIO(search_results.stdout))
    if search_results_df.empty:
        print("Creating new dataset...")
        subprocess.run(
            f"kaggle datasets create -p {model_dir} --dir-mode tar",
            shell=True,
            check=True,
        )
    else:
        print("Updating existing dataset...")
        version_string = f"CV AUC: {cv_auc_avg:.4f}, CV pAUC: {cv_pauc_avg:.4f}"
        subprocess.run(
            f"kaggle datasets version -p {model_dir} -m '{version_string}' --dir-mode tar",
            shell=True,
            check=True,
        )
    print(f"Weights for {model_name}_{version} uploaded to Kaggle ✅")
