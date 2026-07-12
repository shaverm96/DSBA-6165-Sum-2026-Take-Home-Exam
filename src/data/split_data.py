from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class SplitPaths:
    train_csv: Path
    val_csv: Path
    test_csv: Path


def collect_image_records(dataset_root: Path, class_names: Sequence[str]) -> pd.DataFrame:
    """Collect image paths and labels from images nested anywhere under a dataset root."""
    records: List[Dict[str, str]] = []
    class_name_set = set(class_names)

    for image_path in sorted(dataset_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label = next((part for part in image_path.parts if part in class_name_set), None)
        if label is None:
            continue

        records.append(
            {
                "image_path": str(image_path.resolve()),
                "label": label,
            }
        )
    if not records:
        raise ValueError(f"No images found under {dataset_root}")
    return pd.DataFrame(records)


def create_stratified_splits(
    records: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train, validation, and test splits.

    The test split is taken from the full dataset. The validation split is then
    carved out of the remaining training portion.
    """
    train_val_records, test_records = train_test_split(
        records,
        test_size=test_size,
        stratify=records["label"],
        random_state=random_state,
    )

    train_records, val_records = train_test_split(
        train_val_records,
        test_size=val_size,
        stratify=train_val_records["label"],
        random_state=random_state,
    )
    return (
        train_records.reset_index(drop=True),
        val_records.reset_index(drop=True),
        test_records.reset_index(drop=True),
    )


def save_split_csvs(
    train_records: pd.DataFrame,
    val_records: pd.DataFrame,
    test_records: pd.DataFrame,
    output_dir: Path,
) -> SplitPaths:
    """Save split assignments as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / "train_split.csv"
    val_csv = output_dir / "val_split.csv"
    test_csv = output_dir / "test_split.csv"
    train_records.to_csv(train_csv, index=False)
    val_records.to_csv(val_csv, index=False)
    test_records.to_csv(test_csv, index=False)
    return SplitPaths(train_csv=train_csv, val_csv=val_csv, test_csv=test_csv)


def build_split_csvs(
    dataset_root: Path,
    output_dir: Path,
    class_names: Sequence[str],
    test_size: float,
    val_size: float,
    random_state: int,
) -> SplitPaths:
    """Collect image records, create splits, and save them to CSV files."""
    records = collect_image_records(dataset_root, class_names)
    train_records, val_records, test_records = create_stratified_splits(
        records=records,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    return save_split_csvs(train_records, val_records, test_records, output_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for split generation."""
    parser = argparse.ArgumentParser(description="Create stratified image splits.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root folder containing the raw dataset")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where split CSVs will be written")
    parser.add_argument("--classes", nargs="+", default=["cats", "dogs"], help="Class folder names to include")
    parser.add_argument("--test-size", type=float, default=0.30, help="Proportion of the full dataset used for testing")
    parser.add_argument("--val-size", type=float, default=0.15, help="Proportion of the full dataset used for validation")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for stratified splitting")
    return parser.parse_args()


def main() -> None:
    """Generate split CSV files from the raw dataset root."""
    args = parse_args()
    split_paths = build_split_csvs(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        class_names=args.classes,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )
    print(f"Saved train split to {split_paths.train_csv}")
    print(f"Saved validation split to {split_paths.val_csv}")
    print(f"Saved test split to {split_paths.test_csv}")


if __name__ == "__main__":
    main()
