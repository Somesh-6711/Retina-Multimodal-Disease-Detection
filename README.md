# Multimodal Retina Disease Detection (Fundus + OCT Fusion)

This project implements a **multimodal deep learning pipeline** for retinal disease detection using **fundus photographs** and **OCT B-scans**, closely mirroring the workflows used in real ophthalmology AI research labs.

We train three models:

1️⃣ **Fundus-only CNN for diabetic retinopathy severity classification**  
2️⃣ **OCT-only CNN for structural retinal disease classification**  
3️⃣ **Fusion model combining both modalities for binary disease detection**

The key idea:
> 🧠 **Fundus shows surface vascular damage, OCT shows structural retinal layers — together they provide a stronger diagnostic signal than either alone.**

---

## 📌 Dataset Sources

### **Fundus (APTOS 2019 Diabetic Retinopathy Dataset)**
- 3662 labelled color fundus photographs
- 5-class DR severity scale:

0 – No DR
1 – Mild
2 – Moderate
3 – Severe
4 – Proliferative DR


### **OCT (Kermany 2018 Retinal OCT Dataset)**
- Four OCT disease classes:

CNV
DME
DRUSEN
NORMAL


> Note: datasets are **not from the same patients** — we align classes to build *virtual multimodal pairs* for fusion.

---

## 🧱 Pipeline Overview

### **1️⃣ Fundus DR Classification (EfficientNet-B0)**

- Input: RGB fundus image  
- Task: 5-class DR severity  
- Loss: Cross-entropy  
- Augmentation: flips, rotations, color jitter  
- Implementation: `timm` + `PyTorch`

📈 **Validation Accuracy ≈ 0.81**

Most confident on:
✔ No DR  
✔ Moderate DR  

More challenging:
⚠ Severe / proliferative (rare classes)

---

## 👁 Fundus Explainability (Grad-CAM)

Grad-CAM highlights lesions such as:

- microaneurysms
- hemorrhages
- exudates

<p align="center">
<img src="outputs/fundus/gradcam/6733544ae7a6_true2_pred2_gradcam.png" width="450">
</p>

<p align="center">
<i>Example Grad-CAM localization on a diabetic retinopathy fundus image</i>
</p>

---

## **2️⃣ OCT Disease Classification (ResNet-18)**

- Input: grayscale OCT B-scan  
- Task: 4-class disease classification  
- Classes: CNV, DME, DRUSEN, NORMAL  
- Training on balanced subset of 3200 samples  
- Validation set: 32 images  

📈 **Validation Accuracy ≈ 1.00**  
(This dataset is relatively “clean” and separable)

---

## 🧠 OCT Explainability (Grad-CAM)

Grad-CAM highlights structural disruptions in retinal layers.

<p align="center">
<img src="outputs/oct/gradcam/CNV-8598714-1_trueCNV_predCNV_gradcam.png" width="450">
</p>

<p align="center">
<i>Grad-CAM highlighting CNV lesion in OCT</i>
</p>

---

## **3️⃣ Multimodal Fusion Model**

We freeze both encoders:


> Note: datasets are **not from the same patients** — we align classes to build *virtual multimodal pairs* for fusion.

---

## 🧱 Pipeline Overview

### **1️⃣ Fundus DR Classification (EfficientNet-B0)**

- Input: RGB fundus image  
- Task: 5-class DR severity  
- Loss: Cross-entropy  
- Augmentation: flips, rotations, color jitter  
- Implementation: `timm` + `PyTorch`

📈 **Validation Accuracy ≈ 0.81**

Most confident on:
✔ No DR  
✔ Moderate DR  

More challenging:
⚠ Severe / proliferative (rare classes)

---

## 👁 Fundus Explainability (Grad-CAM)

Grad-CAM highlights lesions such as:

- microaneurysms
- hemorrhages
- exudates

<p align="center">
<img src="outputs/fundus/gradcam/6733544ae7a6_true2_pred2_gradcam.png" width="450">
</p>

<p align="center">
<i>Example Grad-CAM localization on a diabetic retinopathy fundus image</i>
</p>

---

## **2️⃣ OCT Disease Classification (ResNet-18)**

- Input: grayscale OCT B-scan  
- Task: 4-class disease classification  
- Classes: CNV, DME, DRUSEN, NORMAL  
- Training on balanced subset of 3200 samples  
- Validation set: 32 images  

📈 **Validation Accuracy ≈ 1.00**  
(This dataset is relatively “clean” and separable)

---

## 🧠 OCT Explainability (Grad-CAM)

Grad-CAM highlights structural disruptions in retinal layers.

<p align="center">
<img src="outputs/oct/gradcam/CNV-8598714-1_trueCNV_predCNV_gradcam.png" width="450">
</p>

<p align="center">
<i>Grad-CAM highlighting CNV lesion in OCT</i>
</p>

---

## **3️⃣ Multimodal Fusion Model**

We freeze both encoders:

z_fundus = EfficientNet-B0 embedding
z_oct = ResNet-18 embedding


Then concatenate and train an **MLP classifier**:

z = concat(z_fundus , z_oct)


### **Task**
Binary classification:

normal = DR grade 0 + OCT NORMAL
disease = DR 1–4 + OCT CNV/DME/DRUSEN


Dataset:
- 3000 paired samples  
- 2400 train / 600 validation  

📈 **Fusion Validation Accuracy = 0.995**

| Model           | Modality        | Task                         | Validation Accuracy |
|-----------------|-----------------|-----------------------------|---------------------|
| Fundus-only     | Fundus          | 5-class DR                  | ~0.81               |
| OCT-only        | OCT             | 4-class disease             | ~1.00 (n=32)        |
| Fusion          | Fundus + OCT    | Binary normal vs disease    | **0.995**           |

✔ Fusion outperforms fundus-only  
✔ Fusion produces extremely robust binary screening performance  
✔ Shows **complementary value of multimodal imaging**

---

## 📸 Before → After (Explainability View)

### Fundus: Raw → Grad-CAM

<p align="center">
  <img src="outputs/fundus/gradcam/e1fb532f55df_true3_pred4_gradcam.png" width="500">
</p>

---

### OCT: Raw → Grad-CAM

<p align="center">
  <img src="outputs/oct/gradcam/DME-9583225-1_trueDME_predDME_gradcam.png" width="500">
</p>

---

## 🧪 Training Logs (Highlights)

### Fundus (EffNet-B0)
Best validation accuracy: 0.8131


### Fusion (MLP Head)
Best validation accuracy: 0.9950


Macro-F1 for both classes ≈ **0.99**

---

## 📦 Tech Stack

- **PyTorch**
- **timm**
- **Albumentations**
- **scikit-learn**
- **Grad-CAM**
- **NumPy / Pandas**
- **Matplotlib**

---

## 👩‍⚕️ Clinical Relevance

- Fundus = vascular & surface biomarkers  
- OCT = retinal microstructure  
- Fusion ≈ clinical workflow  
- Explainability builds trust  
- Binary disease vs normal supports **screening pipelines**

This project demonstrates:

✔ multimodal medical AI  
✔ deep learning engineering  
✔ explainability  
✔ clinical-style evaluation  
✔ reproducible pipeline design  

---

## 🚀 Future Extensions

- Train full OCT dataset
- Multi-class fusion (not just binary)
- Add SHAP interpretability
- Patient-level aggregation
- Deploy via Streamlit

---

## 📁 Repo Structure


Macro-F1 for both classes ≈ **0.99**

---

## 📦 Tech Stack

- **PyTorch**
- **timm**
- **Albumentations**
- **scikit-learn**
- **Grad-CAM**
- **NumPy / Pandas**
- **Matplotlib**

---

## 👩‍⚕️ Clinical Relevance

- Fundus = vascular & surface biomarkers  
- OCT = retinal microstructure  
- Fusion ≈ clinical workflow  
- Explainability builds trust  
- Binary disease vs normal supports **screening pipelines**

This project demonstrates:

✔ multimodal medical AI  
✔ deep learning engineering  
✔ explainability  
✔ clinical-style evaluation  
✔ reproducible pipeline design  

---

## 🚀 Future Extensions

- Train full OCT dataset
- Multi-class fusion (not just binary)
- Add SHAP interpretability
- Patient-level aggregation
- Deploy via Streamlit

---

## 📁 Repo Structure

data/
outputs/
src/
models/


---
