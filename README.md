# Melanoma Classification with ResNet34

A binary skin-lesion classifier that distinguishes **benign** from **malignant** melanoma images using transfer learning on a pretrained **ResNet34** backbone, built with **fastai** (PyTorch).

## Overview

- **Dataset:** [Melanoma Cancer Image Dataset](https://www.kaggle.com/) (Kaggle) — ~10,605 images total (8,484 train / 2,121 validation / 1,000 test)
- **Classes:** `benign`, `malignant`
- **Model:** ResNet34, pretrained on ImageNet, fine-tuned via fastai's `vision_learner`
- **Framework:** fastai / PyTorch

## Pipeline

1. **Data loading** — `ImageDataLoaders.from_folder` with `Resize(460)` item transform and `aug_transforms(size=224)` + ImageNet normalization as batch transforms.
2. **Model** — ResNet34 via `vision_learner`, initialized with a frozen pretrained backbone.
3. **Training** — two-stage fine-tuning:
   - **Frozen backbone:** learning-rate search (`lr_find`) → `fit_one_cycle(3, ...)`
   - **Unfrozen (full model):** fresh `lr_find` → `fit_one_cycle(10, ...)`
4. **Evaluation** — classification report and confusion matrix on a held-out test set.
5. **Cross-validation** — 3-fold CV with accuracy and ROC-AUC tracking, plus ROC curve visualization.

## Results

| Metric | Benign | Malignant | Overall |
|---|---|---|---|
| Precision | 0.97 | 0.99 | — |
| Recall | 0.99 | 0.97 | — |
| F1-score | 0.98 | 0.98 | — |
| Accuracy | — | — | **98%** |

## Requirements

```
fastai
torch
torchvision
scikit-learn
opencv-python
pandas
numpy
seaborn
matplotlib
Pillow
```

## Usage

1. Download the melanoma dataset and place it so the directory structure matches:
   ```
   melanoma_cancer_dataset/
   ├── train/
   │   ├── benign/
   │   └── malignant/
   └── test/
       ├── benign/
       └── malignant/
   ```
2. Update the `dataset_dir` / `train` / `test_data` paths in the notebook to point to your local dataset location.
3. Run the notebook cells in order: data loading → model setup → training (frozen, then unfrozen) → evaluation → cross-validation.

## Notes

- This is a research/educational project, not a diagnostic tool. Predictions should not be used for actual medical decision-making.
- The K-fold cross-validation section continues fine-tuning the same trained model across folds rather than retraining from scratch each time, so fold-wise metrics should be interpreted as a sensitivity check rather than a strict independent cross-validation estimate.

## License

Add your preferred license here (e.g. MIT).
