# Explainable AI for Multi-Class Lung Disease Classification from Chest X-Ray Images

## Project Overview

This project presents a complete deep learning and Explainable Artificial Intelligence (XAI) framework for classifying lung diseases from chest X-ray images.

The main goal is not only to build accurate AI models, but also to make their predictions interpretable and understandable. This is especially important in healthcare, where model transparency, trust, and clinical relevance are essential.

The project compares multiple deep learning, transfer learning, hybrid machine learning, and ensemble models. Each team member implemented separate models, evaluated them using standard classification metrics, and applied explainability techniques such as Grad-CAM, LIME, SHAP, Integrated Gradients, and other XAI methods.

---

## Team Members and Contributions

| Team Member | Models / Contributions |
|---|---|
| Mohammed Essam | EfficientNetB0, InceptionV3, VGG16 + SVM |
| Omar Medhat | InceptionResNetV2, DenseNet121, ConvNeXtTiny, Weighted TTA Ensemble |
| Ahmed Ragheb | MobileNetV2, Xception, NASNetMobile, Soft-Voting Ensemble |
| Mostafa Magdy | VGG19, ResNet50, CNN + LSTM, Ensemble Model |

---

## Problem Statement

Chest X-ray imaging is one of the most widely used diagnostic tools for lung-related diseases. However, manual interpretation can be time-consuming, subjective, and dependent on radiologist experience.

Deep learning models can assist in automated chest X-ray classification, but many of these models behave as black boxes. In medical applications, this lack of interpretability limits trust and real-world adoption.

This project addresses the following goals:

- Classify chest X-ray images into multiple lung disease categories.
- Compare different deep learning and machine learning approaches.
- Apply Explainable AI techniques to understand model predictions.
- Analyze class-level performance and misclassifications.
- Build ensemble models to improve robustness and reliability.
- Provide a complete research-based implementation suitable for healthcare AI experimentation.

---

## Dataset

The project uses the **X-ray Lung Diseases Images 9 Classes** dataset from Kaggle.

Dataset link:

```text
https://www.kaggle.com/datasets/fernando2rad/x-ray-lung-diseases-images-9-classes
```

The dataset contains chest X-ray images divided into 9 classes:

| Class Index | Class Name |
|---|---|
| 00 | Anatomia Normal |
| 01 | Processos Inflamatórios Pulmonares (Pneumonia) |
| 02 | Maior Densidade (Derrame Pleural, Consolidação Atelectasica, Hidrotorax, Empiema) |
| 03 | Menor Densidade (Pneumotorax, Pneumomediastino, Pneumoperitonio) |
| 04 | Doenças Pulmonares Obstrutivas (Enfisema, Broncopneumonia, Bronquiectasia, Embolia) |
| 05 | Doenças Infecciosas Degenerativas (Tuberculose, Sarcoidose, Proteinose, Fibrose) |
| 06 | Lesões Encapsuladas (Abscessos, Nódulos, Cistos, Massas Tumorais, Metastases) |
| 07 | Alterações de Mediastino (Pericardite, Malformações Arteriovenosas, Linfonodomegalias) |
| 08 | Alterações do Tórax (Atelectasias, Malformações, Agenesia, Hipoplasias) |

The dataset is not perfectly balanced, so several notebooks used stratified splitting and class weights to reduce bias toward majority classes.

---

## Repository Structure

Recommended GitHub structure:

```text
DSAI305_XAI_Lung_Disease_Project/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   │
│   ├── 00_Preprocessing_EDA_FeatureEngineering.ipynb
│   │
│   ├── Mohammed_EfficientNetB0.ipynb
│   ├── Mohammed_InceptionV3.ipynb
│   ├── Mohammed_VGG16_SVM.ipynb
│   │
│   ├── Omar_InceptionResNetV2_XAI.ipynb
│   ├── Omar_DenseNet121_XAI.ipynb
│   ├── Omar_ConvNeXtTiny_XAI.ipynb
│   ├── Omar_Weighted_TTA_Ensemble.ipynb
│   │
│   ├── Ahmed_MobileNetV2.ipynb
│   ├── Ahmed_Xception.ipynb
│   ├── Ahmed_NASNetMobile.ipynb
│   ├── Ahmed_SoftVoting_Ensemble.ipynb
│   │
│   ├── Mostafa_VGG19.ipynb
│   ├── Mostafa_ResNet50.ipynb
│   ├── Mostafa_CNN_LSTM.ipynb
│   └── Mostafa_Ensemble.ipynb
│
├── results/
│   ├── figures/
│   ├── confusion_matrices/
│   ├── xai_outputs/
│   ├── model_metrics/
│   └── ensemble_results/
│
├── reports/
│   ├── research_paper.pdf
│   └── final_presentation.pptx
│
└── saved_models/
    └── README_saved_models.txt
```

Note: Large `.keras`, `.h5`, `.pkl`, or model checkpoint files may exceed GitHub upload limits. If needed, upload them to Google Drive and include the link in `saved_models/README_saved_models.txt`.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/DSAI305_XAI_Lung_Disease_Project.git
cd DSAI305_XAI_Lung_Disease_Project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

Recommended `requirements.txt`:

```text
tensorflow
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
pillow
lime
shap
scikit-image
kagglehub
jupyter
notebook
```

For Google Colab, most libraries are already installed, but some packages such as `lime`, `shap`, and `kagglehub` may need installation inside the notebook.

---

## How to Run the Project

### Step 1: Run the Preprocessing / EDA Notebook

Run:

```text
notebooks/00_Preprocessing_EDA_FeatureEngineering.ipynb
```

This notebook performs:

- Dataset loading
- Dataset structure inspection
- Corrupted image checking
- Class distribution analysis
- Sample image visualization
- Image size analysis
- CLAHE contrast enhancement visualization
- Train / validation / test split creation
- Saving shared split files

Expected outputs:

```text
train_split.csv
val_split.csv
test_split.csv
class_distribution.png
sample_images.png
feature_engineering_clahe.png
```

---

### Step 2: Run Individual Model Notebooks

Each model notebook should be run independently.

Each notebook generally includes:

- Loading image paths and labels
- Image preprocessing
- Data augmentation
- Model construction
- Training
- Fine-tuning
- Evaluation
- Confusion matrix
- Classification report
- XAI visualization
- Saving prediction probabilities for ensemble models

---

### Step 3: Run Ensemble Notebooks

After running the individual model notebooks, run the ensemble notebooks.

Examples:

```text
Omar_Weighted_TTA_Ensemble.ipynb
Ahmed_SoftVoting_Ensemble.ipynb
Mostafa_Ensemble.ipynb
```

The ensemble notebooks load saved prediction probabilities and combine model outputs using soft voting or weighted voting.

---

## Preprocessing and Feature Engineering

The preprocessing pipeline included:

- Loading image paths and labels into dataframes
- Removing corrupted or unreadable images
- Resizing images to the required model input size
- Applying model-specific preprocessing functions
- Normalizing image inputs
- Applying data augmentation during training
- Using stratified train / validation / test splitting
- Computing class weights for imbalanced classes
- Applying CLAHE visualization to improve local contrast in X-ray images

Common augmentation techniques included:

- Rotation
- Width shift
- Height shift
- Zoom
- Horizontal flipping
- Brightness adjustment
- Shear transformation

---

## Models Implemented

# Mohammed Essam Models

## 1. EfficientNetB0

EfficientNetB0 was selected because it provides a strong balance between classification accuracy and computational efficiency. It is suitable for chest X-ray classification because it can extract strong features while keeping the model relatively lightweight.

### Results

| Metric | Value |
|---|---:|
| Test Accuracy | 96.15% |
| Macro F1-score | 0.97 |
| Weighted F1-score | 0.96 |

EfficientNetB0 performed well across most classes. Some classes achieved excellent results, such as:

- Alterações de Mediastino
- Alterações do Tórax
- Lesões Encapsuladas

The Pneumonia class was more challenging and had lower F1-score compared with the strongest classes.

---

## 2. InceptionV3

InceptionV3 was selected because it uses inception modules that extract features at different scales. This is useful for chest X-rays because disease-related patterns may appear in different sizes and locations.

### Results

| Metric | Value |
|---|---:|
| Test Accuracy | 97.04% |
| Macro F1-score | 0.97 |
| Weighted F1-score | 0.97 |

The model performed strongly across most classes. It achieved high performance on classes such as:

- Alterações de Mediastino
- Alterações do Tórax
- Maior Densidade
- Lesões Encapsuladas

### Main Challenges

- Pneumonia had lower recall.
- Doenças Pulmonares Obstrutivas had lower precision.
- Some visually similar classes caused confusion.

### XAI Techniques Used

- LIME
- SHAP
- Grad-CAM

---

## 3. VGG16 + SVM

VGG16 was used as a pretrained feature extractor, while Support Vector Machine (SVM) was used as the final classifier.

This hybrid approach combines the feature extraction power of deep CNNs with the classification ability of traditional machine learning.

### Results

| Metric | Value |
|---|---:|
| Training Accuracy | 99.98% |
| Validation Accuracy | 99.21% |
| Test Accuracy | 99.21% |
| Macro F1-score | 0.99 |
| Weighted F1-score | 0.99 |

This model achieved the highest reported performance among Mohammed’s models.

The notebook also checked for:

- Overfitting
- Data leakage
- Train / validation / test path overlap
- Misclassification between visually similar classes

### XAI Techniques Used

- LIME
- SHAP

Grad-CAM was not directly applied because the final classifier was SVM, not a CNN classification head.

---

# Omar Medhat Models

## 1. InceptionResNetV2

InceptionResNetV2 combines Inception modules with residual connections. It was used as a pretrained transfer learning model with a custom classification head.

The classification head included:

```text
Global Average Pooling
Batch Normalization
Dropout
Dense layer with 512 neurons
Batch Normalization
Dropout
Softmax output layer
```

The model was trained in two stages:

1. Freeze the pretrained base and train the classification head.
2. Unfreeze the final layers and fine-tune with a smaller learning rate.

### Results

| Setting | Accuracy | Macro F1 |
|---|---:|---:|
| Standard Test | 95.85% | 96.33% |
| TTA Test | 96.84% | 97.28% |

TTA improved the model performance by averaging predictions over augmented versions of the same test image.

### XAI Techniques Used

- Grad-CAM
- LIME
- SHAP
- Integrated Gradients

---

## 2. DenseNet121

DenseNet121 was selected because dense connections improve feature reuse and gradient flow. It was used as a pretrained transfer learning model with a custom classification head.

### Results

| Setting | Accuracy | Macro F1 |
|---|---:|---:|
| Standard Test | 96.74% | 97.16% |
| TTA Test | 96.64% | 97.05% |

DenseNet121 achieved the strongest standard test accuracy among Omar’s individual models.

TTA did not improve DenseNet121, which suggests that the standard prediction was already stable.

### XAI Techniques Used

- Grad-CAM
- LIME
- SHAP
- Integrated Gradients

---

## 3. ConvNeXtTiny

ConvNeXtTiny is a modern convolutional neural network architecture. It was used to add architectural diversity to the project.

### Results

| Setting | Accuracy | Macro F1 |
|---|---:|---:|
| Standard Test | 95.85% | 96.54% |
| TTA Test | 96.15% | 96.75% |

TTA improved ConvNeXtTiny performance and made its predictions more robust.

### XAI Techniques Used

- Grad-CAM
- LIME
- SHAP
- Integrated Gradients

---

## 4. Weighted TTA Ensemble

Omar implemented a weighted soft-voting ensemble using:

- InceptionResNetV2
- DenseNet121
- ConvNeXtTiny

The ensemble used TTA prediction probabilities and grid search to find the best weights.

### Best Weights

| Model | Weight |
|---|---:|
| InceptionResNetV2 | 0.35 |
| DenseNet121 | 0.50 |
| ConvNeXtTiny | 0.15 |

### Ensemble Results

| Model/System | Accuracy | Macro F1 |
|---|---:|---:|
| Best Weighted TTA Ensemble | 97.63% | 97.86% |
| Manual Weighted TTA Ensemble | 97.43% | 97.74% |
| Standard Soft Voting Ensemble | 97.33% | 97.74% |
| TTA Soft Voting Ensemble | 97.23% | 97.62% |

### Bootstrap Confidence Intervals

| Metric | Mean | 95% Confidence Interval |
|---|---:|---:|
| Accuracy | 97.63% | 96.74% – 98.52% |
| Macro F1 | 97.89% | 97.06% – 98.71% |

The weighted TTA ensemble outperformed all individual models in Omar’s pipeline.

---

# Ahmed Ragheb Models

## 1. MobileNetV2

MobileNetV2 is a lightweight transfer learning model. It uses inverted residual blocks and depthwise separable convolutions to reduce computation while maintaining good feature extraction ability.

### Results

| Metric | Value |
|---|---:|
| Accuracy | 0.9437 |
| Precision | 0.9451 |
| Recall | 0.9437 |
| F1-score | 0.9437 |

MobileNetV2 was the best individual model in Ahmed’s experiments based on weighted F1-score.

---

## 2. Xception

Xception is a high-capacity transfer learning model based on depthwise separable convolutions.

### Results

| Metric | Value |
|---|---:|
| Accuracy | 0.8745 |
| Precision | 0.8751 |
| Recall | 0.8745 |
| F1-score | 0.8726 |

Xception was the weakest individual model in Ahmed’s run.

---

## 3. NASNetMobile

NASNetMobile is a neural architecture search model designed to balance efficiency and accuracy.

### Results

| Metric | Value |
|---|---:|
| Accuracy | 0.8933 |
| Precision | 0.8924 |
| Recall | 0.8933 |
| F1-score | 0.8925 |

---

## 4. Soft-Voting Ensemble

Ahmed implemented a soft-voting ensemble that combined individual model outputs.

### Results

| Model/System | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Soft-Voting Ensemble | 0.9466 | 0.9469 | 0.9466 | 0.9464 |
| MobileNetV2 | 0.9437 | 0.9451 | 0.9437 | 0.9437 |
| NASNetMobile | 0.8933 | 0.8924 | 0.8933 | 0.8925 |
| Xception | 0.8745 | 0.8751 | 0.8745 | 0.8726 |

### XAI Techniques Used

- Grad-CAM
- LIME
- SHAP
- Integrated Gradients
- Occlusion Sensitivity

Ahmed also computed additional XAI metrics:

- Central Focus Score
- Heatmap Entropy

---

# Mostafa Magdy Models

## 1. VGG19

VGG19 was used as a transfer learning model. The pretrained convolutional base extracted image features, while custom dense layers were added for final classification.

Fine-tuning was applied to improve the model’s ability to adapt to the lung disease dataset.

---

## 2. ResNet50

ResNet50 was selected because it uses residual connections, which help deeper networks train more effectively and reduce vanishing gradient problems.

It was used as a strong transfer learning backbone for extracting complex chest X-ray features.

---

## 3. CNN + LSTM

CNN + LSTM combines convolutional layers with recurrent layers.

- CNN layers extract spatial image features.
- LSTM layers process extracted feature sequences.

This architecture was included to compare a hybrid deep learning approach against standard CNN transfer learning models.

---

## 4. Ensemble Model

Mostafa implemented an ensemble model that combines predictions from multiple trained models.

The purpose of the ensemble was to improve reliability and reduce the effect of individual model errors.

### Evaluation Methods

Mostafa’s models were evaluated using:

- Accuracy
- Loss curves
- Classification reports
- Confusion matrices


---

## Explainable AI Techniques

The project used multiple XAI methods to explain model predictions.

## Grad-CAM

Grad-CAM generates heatmaps that show which regions of the image contributed most strongly to the prediction.

It is useful for checking whether the model focuses on relevant lung and chest regions instead of irrelevant background areas.

---

## LIME

LIME explains individual predictions by perturbing image regions and identifying which superpixels influenced the model decision.

It provides local explanations for specific test images.

---

## SHAP

SHAP estimates how much image regions or extracted features contribute positively or negatively to the model prediction.

It helps explain model behavior using feature attribution.

---

## Integrated Gradients

Integrated Gradients computes pixel-level attribution by comparing the input image to a baseline and accumulating gradients along the interpolation path.

It provides detailed attribution maps but can sometimes be noisy.

---

## Occlusion Sensitivity

Occlusion Sensitivity hides parts of the image and measures how prediction confidence changes.

This helps identify important regions used by the model.

---

## Central Focus Score

Central Focus Score measures whether an explanation heatmap focuses on the central chest or lung area.

This is useful for evaluating whether the model is paying attention to medically relevant regions.

---

## Heatmap Entropy

Heatmap Entropy measures how concentrated or spread out an explanation heatmap is.

Lower entropy usually indicates more focused attention, while higher entropy indicates more distributed attention.

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Weighted F1-score
- Confusion matrix
- Classification report
- Training / validation curves
- Test-Time Augmentation performance
- Bootstrap confidence intervals
- XAI explanation metrics

Accuracy alone was not considered sufficient because the dataset is multi-class and imbalanced. F1-score, macro F1-score, and weighted F1-score were used to provide a clearer view of class-level performance.

---

## Main Results Summary

| Team Member | Best Model/System | Best Reported Result |
|---|---|---:|
| Mohammed Essam | VGG16 + SVM | Test Accuracy = 99.21% |
| Omar Medhat | Weighted TTA Ensemble | Accuracy = 97.63%, Macro F1 = 97.86% |
| Ahmed Ragheb | Soft-Voting Ensemble | F1-score = 0.9464 |
| Mostafa Magdy | soft-voting ensemble | macro F1 = 99.57 |

---

## Key Findings

- Transfer learning was effective for multi-class lung disease classification.
- Ensemble models improved reliability compared with many individual models.
- XAI techniques made the predictions more understandable.
- Some classes were harder to classify due to visual similarity between lung findings.
- Pneumonia and obstructive or density-related findings were common sources of confusion.
- TTA improved some models, especially InceptionResNetV2 and ConvNeXtTiny.
- TTA did not improve every model, such as DenseNet121.
- Bootstrap confidence intervals helped validate the stability of the best ensemble.
- Hybrid methods such as VGG16 + SVM can perform strongly when deep features are effective.

---

## Challenges

The project faced several challenges:

- Dataset imbalance
- Visual similarity between medical classes
- High computational cost of deep learning models
- GPU memory limitations
- Expensive XAI computations, especially SHAP and Integrated Gradients
- Noisy LIME and Integrated Gradients explanations
- Potential overfitting in high-capacity models
- Different preprocessing requirements for different architectures
- Need to check data leakage for very high-performing models

---

## Applied Solutions

The team applied several solutions:

- Stratified train / validation / test splitting
- Class weights for imbalanced data
- Data augmentation
- Transfer learning
- Fine-tuning with small learning rates
- Early stopping
- ReduceLROnPlateau
- Model checkpointing
- Confusion matrix analysis
- Classification report analysis
- Data leakage checks
- Test-Time Augmentation
- Soft-voting ensembles
- Weighted ensembles
- Grid search for ensemble weights
- Bootstrap confidence intervals
- Multiple XAI methods per model

---

## Reproducing the Results

## 1. Prepare the Environment

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Run Preprocessing

Run:

```text
notebooks/00_Preprocessing_EDA_FeatureEngineering.ipynb
```

This creates:

```text
train_split.csv
val_split.csv
test_split.csv
class_distribution.png
sample_images.png
feature_engineering_clahe.png
```

---

## 3. Run Model Notebooks

Run each model notebook.

Each notebook should save outputs such as:

```text
metrics.json
training_history.json
confusion_matrix.png
classification_report
model_prediction_probabilities.npy
xai_visualizations.png
```

---

## 4. Run Ensemble Notebooks

After prediction files are saved, run the ensemble notebooks.

Ensemble notebooks may require files such as:

```text
*_test_pred_probs.npy
*_tta_test_pred_probs.npy
*_test_y_true.npy
class_indices.json
```

---

## 5. Review Outputs

Check the following folders:

```text
results/model_metrics/
results/confusion_matrices/
results/xai_outputs/
results/ensemble_results/
reports/
```

---

## Expected Outputs

The project produces:

- Class distribution plots
- Sample image grids
- CLAHE enhancement examples
- Training and validation curves
- Confusion matrices
- Classification reports
- LIME explanations
- SHAP explanations
- Grad-CAM heatmaps
- Integrated Gradients maps
- Occlusion Sensitivity maps
- Ensemble comparison tables
- Bootstrap confidence intervals
- Research paper
- Final presentation

---

## Hardware and Runtime Notes

Recommended environment:

- Google Colab
- GPU runtime for model training
- Python 3.10+
- TensorFlow 2.x
- At least 12 GB RAM

Notes:

- GPU is recommended for model training and XAI.
- CPU is enough for ensemble notebooks because they only combine saved prediction probabilities.
- Large models may require smaller batch sizes such as 8 or 16.
- SHAP and Integrated Gradients should be applied on a small number of samples to avoid memory issues.
- Restarting the runtime after training may help avoid GPU memory errors.

---

## Ethical and Legal Considerations

This project is for academic and research purposes only. It is not intended for real clinical diagnosis.

Important considerations:

- AI predictions should not replace medical professionals.
- Chest X-ray datasets may contain bias due to patient population, imaging devices, source institutions, or class imbalance.
- Explainability maps should be interpreted carefully because they do not guarantee clinical correctness.
- Patient privacy must be protected when working with medical data.
- External validation on independent clinical datasets is required before any real-world clinical use.
- The model should be used only as a decision-support tool, not as an autonomous diagnostic system.

---

## Limitations

- The dataset is not perfectly balanced.
- Some disease classes have visually similar patterns.
- X-ray images alone may not be enough for certain diagnoses.
- Clinical metadata such as age, symptoms, lab results, and medical history was not included.
- Some XAI visualizations are noisy or broad.
- Some models require high computational resources.
- Mostafa’s exact numeric results should be verified from notebook outputs if required.
- The models were not externally validated on an independent hospital dataset.

---

## Future Work

Future improvements may include:

- Using larger and more diverse datasets
- Adding CT scans or clinical/tabular patient data
- Applying multimodal learning
- Segmenting lung regions before classification
- Testing transformer-based medical imaging models
- Improving confidence calibration
- Performing external validation
- Adding clinician-based validation of XAI explanations
- Deploying the final model in a demo web application
- Improving ensemble optimization using more advanced search strategies

---

## Project Deliverables

The project deliverables include:

- Preprocessing / EDA notebook
- Separate model notebooks for each team member
- Ensemble notebooks
- README file
- Requirements file
- Research paper
- Final presentation
- Results figures and tables
- XAI visualizations
- Saved metrics and prediction outputs

---

## References

The research paper includes references related to:

- Chest X-ray classification
- Transfer learning
- Explainable AI
- Grad-CAM
- LIME
- SHAP
- Deep learning for lung disease diagnosis
- Medical image analysis


---

## Disclaimer

This project was developed for the DSAI305 course as an academic research implementation. The models, results, and visual explanations are not certified medical tools and must not be used for real diagnosis or treatment decisions.
````
