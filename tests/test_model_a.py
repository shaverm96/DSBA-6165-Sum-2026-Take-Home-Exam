import pytest
from pathlib import Path

import pandas as pd
from PIL import Image


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.models.classifier import ResNet18Classifier
from src.data.dataset import ImageClassificationDataset
from src.training.train_classifier import (
    _binary_metrics,
    build_classification_loaders,
    train_classifier,
)


def test_classification_dataset_skips_invalid_images(tmp_path):
    valid_image = tmp_path / "valid.jpg"
    invalid_image = tmp_path / "invalid.jpg"
    split_csv = tmp_path / "split.csv"
    Image.new("RGB", (8, 8), color="white").save(valid_image)
    invalid_image.write_bytes(b"not an image")
    pd.DataFrame(
        [
            {"image_path": valid_image, "label": "cats"},
            {"image_path": invalid_image, "label": "dogs"},
        ]
    ).to_csv(split_csv, index=False)

    dataset = ImageClassificationDataset(
        split_csv=split_csv,
        class_names=["cats", "dogs"],
        validate_images=True,
    )

    assert len(dataset) == 1
    assert dataset.invalid_image_paths == [str(invalid_image)]


def test_resnet18_classifier_returns_one_logit_per_image():
    model = ResNet18Classifier(pretrained=False)
    logits = model(torch.randn(2, 3, 128, 128))
    assert logits.shape == (2,)


def test_binary_metrics_reports_perfect_predictions():
    logits = torch.tensor([-10.0, 10.0, -10.0, 10.0])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    metrics = _binary_metrics(logits, labels)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_classification_loaders_use_original_128_pixel_inputs():
    split_dir = Path("data/splits")
    if not (split_dir / "train_split.csv").exists():
        pytest.skip("Committed split files are not available")

    train_loader, validation_loader = build_classification_loaders(
        split_dir=split_dir,
        class_names=["cats", "dogs"],
        image_size=128,
        batch_size=2,
        num_workers=0,
    )
    images, labels, _ = next(iter(train_loader))
    assert images.shape == (2, 3, 128, 128)
    assert labels.shape == (2,)
    assert len(validation_loader.dataset) > 0


def test_classifier_training_writes_resume_checkpoints(tmp_path):
    examples = [
        (torch.randn(3, 128, 128), torch.tensor(float(index % 2)), f"image-{index}.jpg")
        for index in range(4)
    ]
    loader = torch.utils.data.DataLoader(examples, batch_size=2, shuffle=False)
    model = ResNet18Classifier(pretrained=False)

    history = train_classifier(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-4,
        weight_decay=0.0,
        patience=2,
        checkpoint_dir=tmp_path,
    )

    assert len(history) == 1
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "history.json").exists()
