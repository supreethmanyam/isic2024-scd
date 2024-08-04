import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


class ISICNet(nn.Module):
    def __init__(self, model_name, out_dim, n_features=0, n_meta_dim=(512, 128), pretrained=True, infer=False):
        super(ISICNet, self).__init__()
        self.infer = infer
        self.model = create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="",
        )
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

        in_dim = self.model.num_features
        self.n_features = n_features
        n_meta_dim = list(n_meta_dim)
        if n_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                nn.SiLU(),
            )
            in_dim += n_meta_dim[1]
        self.classifier = nn.Linear(in_dim, out_dim)

    def forward(self, image, meta=None):
        x = self.model(image)
        bs = len(image)
        pool = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        if self.n_features > 0:
            meta = self.meta(meta)
            pool = torch.cat((pool, meta), dim=1)

        logit = 0
        for i in range(len(self.dropouts)):
            logit += self.classifier(self.dropouts[i](pool))
        logit = logit / len(self.dropouts)
        return logit
