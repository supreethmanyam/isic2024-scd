from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


model_factory = {
    "tf_efficientnet_b1_ns": "tf_efficientnet_b1.ns_jft_in1k"
}


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
        model_name = model_factory.get(model_name, model_name)
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
