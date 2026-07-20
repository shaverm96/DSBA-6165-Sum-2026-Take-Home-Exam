# DSBA 6165 Applied AI Midterm Exam

- **Student:** Matthew Shaver (801083206)
- **Project:** Dogs vs. Cats image classification with a MobileNetV2 baseline and an SRGAN-enhanced classifier

This README is the project documentation for the midterm submission. It records the steps I used to build, train, and compare the models so that another student can reproduce a similar experiment from a fresh clone of this repository.

I kept the design intentionally focused: first establish a strong classifier baseline, then test whether a classifier adapted to SRGAN-generated images can recover some of the performance lost when starting from very low-resolution images.

## Project question

Can a classifier trained on SRGAN-generated 128 x 128 images classify cats and dogs better than the original baseline classifier when both receive the same low-resolution source image?

The project evaluates two primary pipelines on the same held-out test records:

| Pipeline | Input path | Purpose |
| --- | --- | --- |
| Model A | Original image -> resize to 128 x 128 -> MobileNetV2 | Baseline performance on the original image domain. |
| Model B | Original image -> downsample to 32 x 32 -> SRGAN -> 128 x 128 -> MobileNetV2 | Performance after super-resolution, with the classifier adapted to generated images. |

The comparison notebook also evaluates the cross-domain combinations (Model A on SRGAN images and Model B on original images). These are diagnostics rather than the main comparison because they show how much the input domain affects each classifier.

## Repository contents

```text
data/
  README.md                         Dataset placement instructions
  splits/                           Reproducible train, validation, and test CSV files
models/
  model_a_mobilenetv2_best.pt       Best Model A checkpoint
  model_b_srgan_mobilenetv2_best.pt Best Model B checkpoint
  srgan/last.pt                     Most recent SRGAN training checkpoint
notebooks/
  01_data_exploration.ipynb         Data audit and split creation
  02_model_a_baseline.ipynb         Model A training and test evaluation
  03_srgan.ipynb                    SRGAN training and Model B training/evaluation
  04_model_comparison.ipynb         Paired Model A versus Model B evaluation
outputs/                            Locally generated figures, metrics, and predictions
requirements.txt                    Python package versions
```

The raw image dataset is intentionally not included because it is large. Generated files under `outputs/` are also ignored by Git and are recreated by running the notebooks.

## Software and hardware requirements

I ran this project with Python 3.11, PyTorch 2.1.2 with CUDA 12.1 support, and an NVIDIA GPU. The training notebooks explicitly require CUDA; they stop with an error if a GPU-enabled PyTorch installation is not available.

From PowerShell on Windows, create the environment and install the documented dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter notebook
```

The key packages are NumPy, pandas, matplotlib, scikit-learn, Pillow, Jupyter Notebook, PyTorch 2.1.2+cu121, and torchvision 0.16.2+cu121. If the computer uses a different CUDA version, install the compatible PyTorch and torchvision wheels before opening the notebooks.

## Data setup

1. Obtain the authorized Dogs vs. Cats image dataset locally. Do not commit the raw images to GitHub.
2. Extract or copy the images into `data/dogs-vs-cats-classification/` so that the class folder names include `cats` and `dogs`. The exact nested folders are flexible because the data notebook searches recursively, but the expected layout is:

   ```text
   data/dogs-vs-cats-classification/
   |-- train/
   |   |-- cats/
   |   `-- dogs/
   |-- validation/
   |   |-- cats/
   |   `-- dogs/
   `-- test/
       |-- cats/
       `-- dogs/
   ```

3. Run [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb). It verifies class coverage, inspects image metadata and samples, checks image readability, and creates the split files in `data/splits/`.

The split procedure uses seed 42 and stratifies on the cat/dog label. It first holds out 30% of images for testing, then assigns 15% of the remaining images to validation. In my local run, this produced 14,831 training images, 2,618 validation images, and 7,479 test images.

> Important reproducibility note: the split CSV files store absolute paths from the machine that generated them. After cloning the repository, run Notebook 01 again to regenerate the CSVs with paths that are valid on the new machine. The fixed seed keeps the split procedure reproducible when the same source images are used.

## Reproduction order

Run the notebooks in the following order. Each notebook is self-contained, so running the cells from top to bottom is important.

1. **[01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)**
   - Audits the raw data and creates `train_split.csv`, `val_split.csv`, and `test_split.csv`.
   - Do this first after placing the dataset locally.

2. **[02_model_a_baseline.ipynb](notebooks/02_model_a_baseline.ipynb)**
   - Trains Model A, a MobileNetV2 binary classifier initialized with ImageNet weights.
   - Images are resized to 128 x 128, normalized to the range [-1, 1], and randomly flipped horizontally during training.
   - The MobileNetV2 feature extractor is frozen; the final classification layer is trained for the cats-versus-dogs task using binary cross-entropy with logits.
   - The best validation-loss checkpoint is saved as `models/model_a_mobilenetv2_best.pt`.

3. **[03_srgan.ipynb](notebooks/03_srgan.ipynb)**
   - Trains an SRGAN generator to map 32 x 32 images to 128 x 128 images. The generator uses residual blocks and pixel-shuffle upsampling; the discriminator provides the adversarial signal. The generator objective combines L1 content loss with a small adversarial-loss weight.
   - The SRGAN checkpoint contains the generator, discriminator, both optimizer states, training history, and the completed epoch number.
   - By default, `RESUME_SRGAN_FROM_LAST = True`. When `models/srgan/last.pt` exists, the SRGAN training cell restores that complete state and continues from the saved epoch instead of starting over. In the supplied run, `last.pt` is at epoch 150 of 150, so the SRGAN section reports that training is already complete and does not repeat the 150 epochs.
   - The notebook then creates Model B. It loads the saved SRGAN in evaluation mode, initializes a second MobileNetV2 from Model A's checkpoint, freezes the SRGAN, and fine-tunes Model B's classifier head on SRGAN-generated training images. The best Model B checkpoint is saved as `models/model_b_srgan_mobilenetv2_best.pt`.

4. **[04_model_comparison.ipynb](notebooks/04_model_comparison.ipynb)**
   - Loads the saved Model A, SRGAN, and Model B checkpoints without training them.
   - Evaluates paired test images and produces classification metrics, ROC/precision-recall plots, class-level metrics, bootstrap intervals, confidence diagnostics, and prediction CSV files.

## Current experiment results

The following values are from `outputs/model_comparison/model_comparison_metrics.json`, using the held-out test set and a dog-probability threshold of 0.50.

| Evaluation | Accuracy | F1 | ROC AUC | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Model A on original images | 0.9315 | 0.9318 | 0.9842 | Primary baseline. |
| Model A on SRGAN images | 0.8047 | 0.8095 | 0.8917 | Shows the domain shift caused by SRGAN images. |
| Model B on original images | 0.8787 | 0.8791 | 0.9533 | Cross-domain diagnostic; Model B was not trained for this input type. |
| Model B on SRGAN images | 0.8866 | 0.8859 | 0.9572 | Primary SRGAN-enhanced pipeline. |

For this first experiment, Model B substantially improved on Model A when both classified SRGAN images (88.66% versus 80.47% accuracy). However, Model B did not exceed Model A on the original 128 x 128 images (88.66% versus 93.15%). My interpretation is that training the classifier on SRGAN outputs helps it adapt to that generated-image domain, but the 32 x 32 downsampling step still removes useful information. I treat this as evidence of the trade-off, not as proof that super-resolution improves image classification in general.

## Reproducibility choices and limitations

- The notebooks set Python, NumPy, and PyTorch random seeds to 42. GPU operations can still have small run-to-run differences, so the goal is similar results rather than guaranteed bit-for-bit identical results.
- The test split is created once and not used for early stopping or model selection. Early stopping for Models A and B uses validation loss.
- Model B is intentionally a staged pipeline: the trained SRGAN is frozen while the Model A classifier head is adapted. I did not perform an end-to-end joint optimization of the generator and classifier.
- I used a small set of manually chosen settings (for example, 32 x 32 low-resolution inputs, 128 x 128 outputs, and 150 SRGAN epochs). I did not run a formal hyperparameter search, so a future version should validate alternative image sizes, generator architectures, and fine-tuning depths.
- The raw dataset is not included in the repository. A reproducer needs the same class labels and a local copy of the images before running Notebook 01.

## Generated results and checkpoints

After a complete run, the most useful files to inspect are:

| Location | Contents |
| --- | --- |
| `models/model_a_mobilenetv2_best.pt` | Best Model A weights selected by validation loss. |
| `models/srgan/last.pt` | Latest resumable SRGAN state. |
| `models/model_b_srgan_mobilenetv2_best.pt` | Best Model B weights selected by validation loss. |
| `outputs/model_a/` | Model A history, test metrics, predictions, and plots. |
| `outputs/srgan/` | SRGAN loss history and generated-image examples. |
| `outputs/model_b/` | Model B history, test metrics, predictions, and plots. |
| `outputs/model_comparison/` | Final paired comparison tables, figures, confidence diagnostics, and predictions. |

## Submission note

This repository and README are the intended submission materials. 