from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Classifier(nn.Module):
    """Binary classifier based on an ImageNet-pretrained ResNet18."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_count = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_count, 1),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return one binary logit per image."""
        features = self.backbone(images)
        return self.classifier(features).squeeze(1)

    def freeze_backbone(self) -> None:
        """Freeze ResNet feature-extractor parameters."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Enable fine-tuning of all model parameters."""
        for parameter in self.parameters():
            parameter.requires_grad = True

    def trainable_parameters(self):
        """Return parameters currently enabled for optimization."""
        return (parameter for parameter in self.parameters() if parameter.requires_grad)
