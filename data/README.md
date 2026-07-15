# Data directory

The reproducible split assignments are tracked in `data/splits/`.

The raw Dogs vs. Cats images are intentionally not committed to GitHub because the dataset is approximately 1.1 GB and contains nearly 25,000 image files. Download or copy the dataset into the local repository before running the notebooks.

Expected local dataset layout:

```text
data/dogs-vs-cats-classification/
├── train/
│   ├── cats/
│   └── dogs/
├── validation/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```
