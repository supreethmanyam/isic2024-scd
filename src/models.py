import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


class ISICNet(nn.Module):
    def __init__(self, model_name, out_dim, pretrained=True, infer=False):
        super(ISICNet, self).__init__()
        self.infer = infer
        self.model = create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="",
        )
        self.classifier = nn.Linear(self.model.num_features, out_dim)

        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

    def forward(self, data):
        image = data
        x = self.model(image)
        bs = len(image)
        pool = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        if self.training:
            logit = 0
            for i in range(len(self.dropouts)):
                logit += self.classifier(self.dropouts[i](pool))
            logit = logit / len(self.dropouts)
        else:
            logit = self.classifier(pool)
        return logit
