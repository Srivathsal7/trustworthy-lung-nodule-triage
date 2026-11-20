# Trustworthy AI for Pulmonary Nodule Diagnosis & Clinical Triage

An end-to-end **Human-in-the-Loop Computer-Aided Diagnosis (CADx)** system for lung nodule malignancy classification on thoracic CT scans. Rather than producing overconfident single-point predictions, this pipeline pairs a **2.5D ResNet-18** backbone with **30-pass Monte Carlo (MC) Dropout**, **Probability Calibration**, and **Layer-4 Grad-CAM** explainability to route uncertain or borderline cases to a human radiologist.

## Live Application Access

**Live Web App:** <https://trustworthy-lung-nodule-triage.streamlit.app>

### How to Use the Diagnostic Interface

1. **Launch the App:** Open the link above in any desktop or mobile browser. *(If the app has been idle for more than 12 hours, click **"Yes, get this app back up!"** and allow 30–60 seconds for the cloud container to initialize.)*
2. **Choose an Input Source (Left Sidebar):**
   * **Browse LIDC Dataset:** Select a cohort partition (`test`, `val`, or `train`) and pick any patient nodule from the dropdown menu.
   * **Upload CT / Patch:** Upload an external `.npz` 2.5D patch, `.dcm` DICOM slice, or grayscale image (`.png`, `.jpg`).
3. **Inspect Triage & Visual Audit Results:**
   * **KPI Summary Cards:** View the predicted diagnosis, MC-calibrated confidence score, predictive Shannon entropy ($H$), and ground-truth label.
   * **Automated Clinical Triage Gate:** Predictions with $H < 0.30$ receive automated diagnostic sign-off; predictions with $H \ge 0.30$ trigger an alert recommending secondary radiologist review.
   * **Four-Panel Grad-CAM Visualizer:** Compare the central axial CT slice (`[-1000, 400]` HU), radiologist consensus mask, focal Layer-4 Grad-CAM heatmap, and saliency overlay.

## Clinical Motivation & Six Trustworthy AI Pillars

Standard deep neural networks suffer from **softmax overconfidence**—often assigning $>95\%$ certainty to ambiguous, ground-glass, or out-of-distribution lung nodules. This project addresses clinical safety through **Six Trustworthy AI Pillars**:

1. **Epistemic Uncertainty Quantification (MC Dropout):** Runs **30 stochastic forward passes** at inference with active Dropout layers (while keeping `BatchNorm2d` frozen) to measure model disagreement via **Predictive Shannon Entropy (**$H$**)**.
2. **Probability Calibration:** Verified with a low **6.15% Expected Calibration Error (ECE)** so displayed confidence scores reflect true empirical likelihood.
3. **Human-in-the-Loop Triage Gate (**$H = 0.30$**):** Automatically signs off on clear lesions ($H < 0.30$) while escalating borderline lesions ($H \ge 0.30$) for secondary radiologist review.
4. **Visual Explainability (Layer-4 Grad-CAM):** Generates noise-filtered saliency heatmaps confirming that predictions are driven by nodule morphology rather than background parenchymal artifacts.
5. **Ambiguity Stress-Testing:** Evaluated on held-out radiologically *Indeterminate* nodules (LIDC consensus rating = 3), demonstrating **1.43x higher predictive uncertainty** on edge cases.
6. **Consensus & Physics Alignment:** Preserves quantitative Hounsfield Unit (`[-1000, 400]` HU) lung-window scaling alongside multi-radiologist ground-truth segmentation masks.

## Quantitative Benchmarks (LIDC-IDRI Test Cohort, $N = 182$)

| Metric | Achieved Score | Clinical Significance |
| :--- | :--- | :--- |
| **Test ROC-AUC** | **0.931** | High discrimination between benign and malignant lesions |
| **Test Accuracy** | **89.56%** | 163 / 182 held-out test nodules classified accurately |
| **Balanced Accuracy** | **88.45%** | Strong sensitivity across imbalanced benign/malignant classes |
| **Test F1 Score** | **0.819** | Balanced precision-recall on positive malignant detections |
| **Expected Calibration Error (ECE)** | **6.15%** | Well-calibrated confidence estimates |
| **Mean Entropy (Clear Nodules)** | **0.1893** | High consensus across 30 MC Dropout passes |
| **Mean Entropy (Indeterminate)** | **0.2713** | **1.43x uncertainty spike** on borderline score-3 lesions |

## Repository Structure

```text
trustworthy-lung-nodule-triage/
├── data/
│   └── processed_large/
│       └── large_manifest.csv             # Metadata manifest indexing 2,317 extracted nodules & cohort splits
├── outputs/
│   ├── best_model_large.pth               # Optimal trained ResNet-18 checkpoint (Val AUC = 0.8234)
│   └── low_entropy_presentation_cases.csv # Top 20 lowest-entropy benign & malignant test cases
│
├── setup_pylidc.py                        # Configures local pylidc paths & LIDC-IDRI DICOM archive connection
├── preprocess_large.py                    # Queries 1,018 CT scans via pylidc & extracts (3, 64, 64) HU patches
├── debug_extraction.py                    # Verifies patch pixel ranges, HU windowing, and mask alignment
├── dataset_large.py                       # Patient-stratified PyTorch Dataset with synchronized spatial flips
├── model_large.py                         # Modified ResNet-18 (NoduleResNet) with MC Dropout & Grad-CAM hooks
├── train_large.py                         # 30-epoch training loop with class-balanced weights [0.74, 1.55]
├── evaluate_large.py                      # Offline test cohort evaluation (ROC-AUC, ECE, Confusion Matrix)
├── explainability_large.py                # Generates batch Layer-4 Grad-CAM visualizations for test nodules
├── generate_external_samples.py           # Creates external/synthetic .npz & image slices for upload testing
├── inspect_eval_entropy.py                # Analyzes entropy distributions across clear vs. indeterminate cohorts
├── test_correct_mc.py                     # Validates BatchNorm-frozen single-sample MC Dropout inference
├── export_low_entropy_cases.py            # Scans test cohort and exports clearest cases for demonstration
├── app.py                                 # Cross-platform Streamlit Clinical Triage & Grad-CAM Dashboard
├── requirements.txt                       # Cloud and local Python dependencies
└── README.md                              # Project documentation
```

## Local Installation & Setup

### 1. Clone the Repository and Activate Environment

```powershell
git clone https://github.com/Srivathsal7/trustworthy-lung-nodule-triage.git
cd trustworthy-lung-nodule-triage
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies

> **Windows 11 Smart App Control / WDAC Note:** If Windows Application Control blocks unsigned CUDA DLLs (`c10_cuda.dll`) inside `venv`, install the signed CPU-only PyTorch wheel. Single-patch 30-pass MC Dropout inference runs in $<150\text{ ms}$ on CPU.

```powershell
# Install dependencies via requirements.txt
python -m pip install -r requirements.txt

# Optional: Additional scripts for LIDC raw DICOM preprocessing
python -m pip install scipy scikit-learn pylidc
```

## End-to-End Execution Pipeline

### Step 1: LIDC-IDRI Setup & Preprocessing

Configure `pylidc`, extract 2.5D `(3, 64, 64)` patches windowed to `[-1000, 400]` HU, and verify tensor integrity:

```powershell
python setup_pylidc.py
python preprocess_large.py
python debug_extraction.py
python dataset_large.py
```

### Step 2: Model Training & Quantitative Evaluation

Verify the architecture, train `NoduleResNet` for 30 epochs, and run calibration/uncertainty benchmarks:

```powershell
python model_large.py
python train_large.py
python evaluate_large.py
```

### Step 3: Uncertainty Diagnostics & Explainability Verification

Validate single-sample MC Dropout behavior, generate Grad-CAM maps, and export low-entropy test cases:

```powershell
python test_correct_mc.py
python inspect_eval_entropy.py
python explainability_large.py
python export_low_entropy_cases.py
python generate_external_samples.py
```

### Step 4: Launch the Interactive Clinical Dashboard Locally

Run Streamlit via Python's module flag (`-m`) to avoid unsigned `.exe` wrapper restrictions:

```powershell
python -m streamlit run app.py
```

## Key Implementation Note: Single-Sample MC Dropout

When running Monte Carlo Dropout on a single patient scan (`batch_size = 1`), calling `model.train()` globally causes PyTorch `nn.BatchNorm2d` layers to compute batch statistics over a single image—zeroing out feature maps and collapsing predictions.

In `app.py` and `test_correct_mc.py`, we keep `BatchNorm2d` frozen in `eval()` mode while selectively activating `Dropout` layers across all 30 stochastic passes:

```python
model.eval()
for m in model.modules():
    if isinstance(m, (nn.Dropout, nn.Dropout2d)):
        m.train()

with torch.no_grad():
    stoch_preds = [F.softmax(model(tensor_in), dim=1).cpu().numpy() for _ in range(30)]
model.eval()
```

## Authors

* **Akkala Srivathsal** (`2024BCS-005`)
* **Gone Abhinav** (`2024BCS-024`)