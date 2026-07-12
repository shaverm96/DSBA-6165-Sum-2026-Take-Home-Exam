# Data directory

The reproducible split assignments are tracked in `data/splits/`.

The raw Dogs vs. Cats images are intentionally not committed to GitHub because the dataset is approximately 1.1 GB and contains nearly 25,000 image files. The notebook downloads the dataset from Kaggle on the first Colab run and caches it in Google Drive for subsequent sessions.

Expected local or Colab dataset layout:

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
