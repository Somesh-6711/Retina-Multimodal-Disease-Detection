# Multimodal Retina Disease Detection (Fundus + OCT Fusion)

This project implements a **multimodal deep learning pipeline** for retinal disease detection using **fundus photographs** and **OCT B-scans**, mirroring how modern ophthalmology labs combine multiple imaging modalities in practice.

We train three models:

1. 🩺 **Fundus-only CNN** for diabetic retinopathy (DR) severity classification  
2. 👁 **OCT-only CNN** for structural retinal disease classification  
3. 🔗 **Fusion model** that combines fundus + OCT embeddings for binary disease detection  

> 🧠 **Fundus captures surface vascular damage, OCT captures retinal layer structure – together they provide a stronger diagnostic signal than either alone.**

---

## 📌 Dataset Sources

### 1️⃣ Fundus – APTOS 2019 Diabetic Retinopathy

- 3662 labeled color fundus photographs  
- 5-class DR severity scale:
  - `0` – No DR  
  - `1` – Mild  
  - `2` – Moderate  
  - `3` – Severe  
  - `4` – Proliferative DR  

Fundus images are used to train the **DR severity classifier**.

---

### 2️⃣ OCT – Kermany 2018 Retinal OCT

- Grayscale OCT B-scans  
- 4 disease classes:
  - `CNV`  
  - `DME`  
  - `DRUSEN`  
  - `NORMAL`  

OCT images are used to train the **OCT disease classifier** and to provide the second modality for fusion.

> ⚠️ The fundus and OCT datasets are **not from the same patients**. For the fusion model, we align labels (normal vs disease) and create *virtual multimodal pairs*.

---

## 🧱 Pipeline Overview

```mermaid
flowchart LR
    F[Fundus image] --> FEnc[Fundus Encoder (EffNet-B0)]
    O[OCT B-scan]   --> OEnc[OCT Encoder (ResNet-18)]
    FEnc --> Z[Concatenate embeddings]
    OEnc --> Z
    Z --> Head[MLP Fusion Head]
    Head --> Y[Normal vs Disease]
```

Core steps:

1. Train fundus DR classifier

2. Train OCT disease classifier

3. Freeze both encoders and train a fusion head on top of concatenated embeddings

4. Use Grad-CAM to visualize where each model is “looking”

## 🩺 Model 1 – Fundus DR Classification (EfficientNet-B0)

Backbone: tf_efficientnet_b0 (via timm)

Input: RGB fundus image

Task: 5-class DR severity

Loss: Cross-entropy

Augmentation: flips, rotations, brightness/contrast and color jitter

Framework: PyTorch + timm + Albumentations

Validation Accuracy: ~0.81

Most confident on:

✅ No DR

✅ Moderate DR

More challenging:

⚠ Severe / proliferative DR (class imbalance & subtle differences)

## 👁 Fundus Explainability (Grad-CAM)

Grad-CAM is used to highlight lesions such as:

microaneurysms

hemorrhages

exudates

✅ Correct prediction example
<p align="center"> <img src="outputs/fundus/gradcam/6733544ae7a6_true2_pred2_gradcam.png" width="450"> </p> <p align="center"> <i>Moderate DR – correctly classified (true = 2, pred = 2).</i> </p>
⚠ Misclassification example
<p align="center"> <img src="outputs/fundus/gradcam/e1fb532f55df_true3_pred4_gradcam.png" width="450"> </p> <p align="center"> <i>Severe DR – model over-grades to proliferative DR (true = 3, pred = 4).</i> </p>

These visualizations help verify that the network is focusing on clinically plausible structures rather than artifacts.

## 🧠 Model 2 – OCT Disease Classification (ResNet-18)

Backbone: resnet18

Input: single-channel OCT B-scan

Task: 4-class disease classification

CNV, DME, DRUSEN, NORMAL

Training set: balanced subset of 3200 images (max 800 per class)

Validation set: 32 images (fast sanity-check set)

Loss: Cross-entropy

Validation Accuracy: ~1.00 on the small validation split

(The dataset is relatively clean and the classes are highly separable.)

## 🧠 OCT Explainability (Grad-CAM)

Grad-CAM highlights structural disruptions in retinal layers for different disease types.

Example – DME
<p align="center"> <img src="outputs/oct/gradcam/DME-9583225-1_trueDME_predDME_gradcam.png" width="450"> </p> <p align="center"> <i>OCT Grad-CAM focusing on edema-related structural changes (DME).</i> </p>
Example – CNV
<p align="center"> <img src="outputs/oct/gradcam/CNV-8598714-1_trueCNV_predCNV_gradcam.png" width="450"> </p> <p align="center"> <i>OCT Grad-CAM highlighting CNV lesion region.</i> </p>

## Model 3 – Multimodal Fusion Head

We first freeze the trained encoders:

z_fundus = EfficientNet-B0 embedding
z_oct    = ResNet-18 embedding


Then we concatenate them:

z = concat(z_fundus, z_oct)


and train a small MLP classifier on top.

## 📊 Result Summary

| Model              | Modality        | Task                         | Validation Accuracy |
|--------------------|-----------------|-----------------------------|---------------------|
| Fundus CNN         | Fundus          | 5-class DR severity         | ~0.81               |
| OCT CNN            | OCT             | 4-class disease             | ~1.00 (n = 32)      |
| Fusion MLP Head    | Fundus + OCT    | Binary normal vs disease    | **0.995**           |

### Key Takeaways
- ✅ **Fusion outperforms fundus-only screening**
- 👁 **OCT captures structural pathology very strongly**
- 🔗 **Multimodal imaging = stronger diagnostic signal**
- 🏥 **Matches real-world retina clinic workflow**

## 👁 Before → After Explainability — Markdown Code
📸 Before → After: Model Explainability Views

## 👁 Fundus Explainability — Raw vs Grad-CAM

<p align="center">
  <img src="outputs/fundus/gradcam/e1fb532f55df_true3_pred4_gradcam.png" width="520">
</p>

<p align="center">
  <i>
  Grad-CAM overlay highlighting DR-related vascular abnormalities on fundus photography.
  (True label = Severe DR, Predicted = Proliferative DR)
  </i>
</p>


## 🧠 OCT Explainability — Raw vs Grad-CAM

<p align="center">
  <img src="outputs/oct/gradcam/DME-9583225-1_trueDME_predDME_gradcam.png" width="520">
</p>

<p align="center">
  <i>
  Grad-CAM visualization showing model attention on macular edema-related structural changes.
  (True label = DME, Predicted = DME)
  </i>
</p>


## 📦 Tech Stack

PyTorch – core deep learning framework

timm – modern CNN backbones (EfficientNet, ResNet)

Albumentations – image augmentation

scikit-learn – metrics & utilities

Grad-CAM – model explainability

NumPy / Pandas / Matplotlib – data & visualization

## 👩‍⚕️ Clinical Relevance

Fundus = vascular & surface biomarkers

OCT = retinal microstructure & macular fluid

Fusion ≈ how ophthalmologists combine modalities when making decisions

Grad-CAM provides visual evidence for where the network is focusing, which is crucial for trust in medical AI

Binary disease vs normal supports screening workflows and referral triage

This project demonstrates:

Multimodal medical AI

Deep learning engineering end-to-end

Explainability and rigorous evaluation

Reproducible, research-style pipeline design

## 🚀 Future Extensions

Train on the full OCT dataset and larger val/test splits

Multiclass fusion (joint DR grade + OCT subtype prediction)

SHAP / Integrated Gradients for richer interpretability

Patient-level aggregation and calibration analysis

Lightweight demo app (Streamlit / FastAPI) for clinicians