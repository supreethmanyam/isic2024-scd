import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import all_labels
from timm import create_model
from medvit import MedViT_base, MedViT_small
from isic2024_scd_app import INPUT_DIR


medvit_model_dict = {
    "medvit_base": {"model": MedViT_base, "weights_filepath": INPUT_DIR / "model-weights/MedViT_base_im1k.pth"},
    "medvit_small": {"model": MedViT_small, "weights_filepath": INPUT_DIR / "model-weights/MedViT_small_im1k.pth"},
}


class ISICNet(nn.Module):
    def __init__(
        self,
        model_name,
        target_mode,
        pretrained=True,
    ):
        super(ISICNet, self).__init__()
        if "medvit" in model_name.lower():
            if model_name not in medvit_model_dict:
                raise ValueError(f"Invalid model name: {model_name}")
            self.model = medvit_model_dict[model_name]["model"]()
            weights = torch.load(medvit_model_dict[model_name]["weights_filepath"], weights_only=True)
            self.model.load_state_dict(weights["model"])
            self.model.proj_head = nn.Identity()
            self.model.avgpool = nn.Identity()
            in_dim = 1024
        else:
            self.model = create_model(
                model_name=model_name,
                pretrained=pretrained,
                in_chans=3,
                num_classes=0,
                global_pool="",
            )
            in_dim = self.model.num_features
        if target_mode == "binary":
            self.classifier = nn.Linear(in_dim, 1)
        elif target_mode == "multi":
            self.classifier = nn.Linear(in_dim, len(all_labels))
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.model_name = model_name

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
