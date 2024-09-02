import torch.nn as nn
import torch.nn.functional as F
from dataset import all_labels
from timm import create_model


model_factory = {
    "tf_efficientnet_b1_ns": "tf_efficientnet_b1.ns_jft_in1k",
    "mobilevitv2_200": "mobilevitv2_200.cvnets_in22k_ft_in1k"
}


class ISICNetMulti(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=True,
    ):
        super(ISICNetMulti, self).__init__()
        model_name = model_factory.get(model_name, model_name)
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
