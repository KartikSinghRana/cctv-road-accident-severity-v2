# CCTV road-accident severity

A three-class EfficientNetV2-S image classifier for road-accident severity in
CCTV footage, with a reproducible R evaluation report and a Streamlit
dashboard for model inspection, image prediction, Grad-CAM, and saliency.

> **Research/demo project only.** This model has not been validated for
> real-world safety decisions. Its held-out split uses approximate
> filename-block groups rather than verified source-video identities.

## What is included

- `road_accident_cctv_severity.ipynb` — dataset preparation, grouped
  train/validation/test splitting, training, and test prediction export.
- `artifacts/best_model.pth` — trained EfficientNetV2-S checkpoint for the
  Streamlit image-prediction tab.
- `streamlit_app.py` and `model_inference.py` — evaluation dashboard,
  single-image predictions, class scores, Grad-CAM, and input-gradient
  saliency.
- `evaluate_model.R` — base-R evaluation with split-manifest checks,
  bootstrap intervals, class metrics, calibration, and plots.
- `artifacts/` — saved test predictions, training history, plots, and R
  evaluation outputs.
- `data/split_manifest.csv` — portable split assignments with relative image
  paths; absolute local machine paths are not included.

The downloaded dataset archive and extracted images are local inputs and are
intentionally excluded from this repository.

## Quick start: dashboard

Use Python 3.10 or later. From this folder, install the packages and launch
the app:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

The trained checkpoint is included under `artifacts/`, so the **Predict &
explain** tab works without running training first. The app accepts one JPG,
JPEG, PNG, WEBP, or BMP image up to 10 MB. It displays the predicted class,
class scores, a class-selectable Grad-CAM overlay, and an input-gradient
saliency overlay. Uploaded images are processed in memory and are not written
to project files by the inference code.

The classifier is a CNN and does not have native self-attention layers. The
gradient saliency view is attention-style, not a true attention map; Grad-CAM
and saliency are approximate diagnostic visualizations, not proof of model
reasoning.

## Train on the dataset

The project uses the [Road Accidents from CCTV Footages Dataset on Kaggle](https://www.kaggle.com/datasets/suryaprabhakaran2005/road-accidents-from-cctv-footages-dataset).
Obtain the archive under the dataset's terms and place it in this folder as
`archive.zip`, or configure the Kaggle CLI credentials outside this
repository and follow the notebook's download cell. Run the cells in
`road_accident_cctv_severity.ipynb` with this folder as the working
directory.

Install a PyTorch build suitable for your CPU/GPU before installing the
remaining requirements if the default pip build is not appropriate for your
hardware.

### Three-class mapping

| Model class | Source folder | Published images |
|---|---|---:|
| `No_Accident` | `NonAccident/NonAccident` | 15,420 |
| `Minor_Accident` | `SeverityScore/Severity Score Dataset with Labels/1` | 1,874 |
| `Severe_Accident` | `SeverityScore/Severity Score Dataset with Labels/2` and `/3` | 4,317 |

The source labels “Substantial Impact” and “Critical Impact” are combined as
`Severe_Accident`. The generic `Accident/Accident` folder is excluded because
it overlaps the severity subsets and is not a separate severity class.

The notebook writes split assignments to `data/split_manifest.csv` using
relative paths only. If `data/source_video_groups.csv` is provided, it must
contain `relative_path` and `group_id` columns with actual per-image source
video IDs. Otherwise, consecutive numeric filenames are grouped in blocks of
100 as a leakage-reduction proxy. This does **not** guarantee
video-independent evaluation; report the split mode and interpret results
accordingly.

## R evaluation

After running the notebook through test prediction export, run the base-R
report from this folder:

```powershell
Rscript evaluate_model.R
```

The script checks that the saved predictions match the held-out rows,
labels, and assignments in the split manifest. It generates accuracy and
stratified-bootstrap 95% confidence intervals, macro/weighted F1, per-class
precision/recall/specificity, one-vs-rest AUC, Brier score, calibration
summaries, a confusion matrix, and a four-panel plot under
`artifacts/r_evaluation/`. It uses base R and does not retrain or tune the
CNN.

The bootstrap intervals quantify image-sample uncertainty for this split.
They do not account for the approximate filename-block grouping or establish
video-independent generalization.

## Current held-out results

The committed evaluation corresponds to the checkpoint and saved test
predictions in `artifacts/`:

- Test accuracy: **91.38%** (stratified image-bootstrap 95% CI:
  **90.14%–92.57%**)
- Macro F1: **0.8984**
- Weighted F1: **0.9162**
- Macro one-vs-rest ROC AUC: **0.9590**

These are results for the current approximate grouped split, not a verified
video-independent benchmark.

## Data and model terms

The Kaggle dataset listing reports an MIT license, while the listing for its
source CCTV video dataset reports an unknown license. Review both datasets'
terms and attribution requirements before redistribution or commercial use.
The checkpoint in this repository was trained on those images; assess the
applicable terms before redistributing or deploying it.

No separate license is provided here for this project. A public repository
does not grant permission to redistribute the source dataset.
