# CIFAKE Image Classification and Explainable Identification of AI-Generated Synthetic Images

A deep learning project for classifying images as **FAKE (AI-generated/synthetic)** or **REAL**, with an additional **Explainable AI (XAI)** component using **Grad-CAM** to visualize the image regions that contributed to the model's prediction.

---

## 📌 Project Overview

With the rapid development of generative AI, AI-generated images are becoming increasingly realistic and difficult to distinguish from genuine images.

This project explores how **Convolutional Neural Networks (CNNs)** can be used to automatically classify images into two categories:

* **FAKE** – AI-generated or synthetic images
* **REAL** – genuine images

The project goes beyond simple classification by incorporating **Grad-CAM (Gradient-weighted Class Activation Mapping)**. Grad-CAM provides a visual explanation of a prediction by highlighting regions of the input image that were important to the CNN's decision.

The complete workflow covers:

**Dataset Validation → Preprocessing → Augmentation → CNN Training → Evaluation → Misclassification Analysis → Grad-CAM Explainability → Result Saving**

---

## 🎯 Objectives

The main objectives of this project are:

* Build a binary image classification system for FAKE and REAL images.
* Prepare and validate the CIFAKE image dataset.
* Apply appropriate image preprocessing techniques.
* Use data augmentation to improve model generalization.
* Develop and train a CNN using TensorFlow/Keras.
* Evaluate the model using multiple performance metrics.
* Analyze correctly classified and misclassified images.
* Examine prediction confidence and uncertain predictions.
* Implement Grad-CAM for model explainability.
* Save the trained model and evaluation outputs for future use.

---

## 🧠 Technologies Used

| Technology           | Purpose                         |
| -------------------- | ------------------------------- |
| **Python**           | Main programming language       |
| **Jupyter Notebook** | Development and experimentation |
| **TensorFlow**       | Deep learning framework         |
| **Keras**            | CNN model development           |
| **NumPy**            | Numerical and array operations  |
| **Pandas**           | Dataset and result management   |
| **Matplotlib**       | Data visualization              |
| **Seaborn**          | Confusion matrix visualization  |
| **Scikit-learn**     | Model evaluation                |
| **PIL**              | Image processing and validation |

---

## 📂 Dataset

The project uses the **CIFAKE dataset**, containing two image categories:

```text
FAKE
REAL
```

The project maps the classes as follows:

| Class | Label |
| ----- | ----: |
| FAKE  |     0 |
| REAL  |     1 |

### Dataset Split

The dataset was divided using a stratified approach:

* **70% — Training**
* **15% — Validation**
* **15% — Testing**

The final test set contains **1,500 images**.

---

## 🔄 Project Workflow

```text
                  ┌─────────────────────┐
                  │     CIFAKE Dataset  │
                  │    FAKE / REAL      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Dataset Validation  │
                  │ • File checking     │
                  │ • Corruption check  │
                  │ • Class analysis    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Image Preprocessing │
                  │ • RGB conversion    │
                  │ • Resize to 32×32   │
                  │ • Normalization     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Data Augmentation   │
                  │ • Flip              │
                  │ • Rotation          │
                  │ • Zoom              │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │      CNN Model      │
                  │ 32 → 64 → 128       │
                  │ Convolution Filters │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Binary Output     │
                  │      Sigmoid        │
                  │    FAKE / REAL      │
                  └──────────┬──────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
      ┌──────────────────┐      ┌──────────────────┐
      │ Model Evaluation │      │     Grad-CAM     │
      │ Accuracy         │      │ Heatmap          │
      │ Precision        │      │ Visualization    │
      │ Recall           │      │ Explanation      │
      │ F1-Score         │      └──────────────────┘
      └──────────────────┘
```

---

# 🏗️ CNN Architecture

The project uses a custom Convolutional Neural Network.

```text
Input
32 × 32 × 3
      │
      ▼
Data Augmentation
      │
      ▼
Conv2D
32 Filters
      │
      ▼
MaxPooling
      │
      ▼
Conv2D
64 Filters
      │
      ▼
MaxPooling
      │
      ▼
Conv2D
128 Filters
      │
      ▼
MaxPooling
      │
      ▼
Flatten
      │
      ▼
Dense
128 Neurons
      │
      ▼
Dropout
0.5
      │
      ▼
Dense
1 Neuron
      │
      ▼
Sigmoid
      │
      ▼
FAKE / REAL
```

### Model Configuration

* Input size: **32 × 32 × 3**
* Convolutional layers: **3**
* Filters: **32, 64, 128**
* Dense layer: **128 neurons**
* Dropout: **0.5**
* Output activation: **Sigmoid**
* Optimizer: **Adam**
* Learning rate: **0.001**
* Loss: **Binary Cross-Entropy**
* Maximum epochs: **20**
* Early stopping: **Enabled**
* Model checkpointing: **Enabled**

---

# 🔍 Explainable AI with Grad-CAM

A major component of this project is the use of **Grad-CAM**.

A classification result such as:

```text
Prediction: FAKE
Confidence: 96.93%
```

provides the final decision, but it does not explain which parts of the image influenced that decision.

Grad-CAM addresses this by generating a heatmap from the CNN's convolutional feature maps.

### Grad-CAM Pipeline

```text
Input Image
     │
     ▼
Trained CNN
     │
     ▼
Final Convolutional Layer
     │
     ▼
Gradient Calculation
     │
     ▼
Feature Importance
     │
     ▼
Grad-CAM Heatmap
     │
     ▼
Overlay on Original Image
```

The final convolutional layer used for Grad-CAM in this project is:

```text
conv2d_2
```

The Grad-CAM visualization helps provide an interpretable view of the regions associated with the model's prediction.

> Note: Grad-CAM is an explanation of the model's learned prediction patterns. It should not be interpreted as proof that a highlighted region is definitively responsible for an image being AI-generated.

---

# 📊 Results

The final model was evaluated on **1,500 test images**.

| Metric                   |     Result |
| ------------------------ | ---------: |
| Test Images              |      1,500 |
| Correct Predictions      |      1,351 |
| Incorrect Predictions    |        149 |
| **Accuracy**             | **90.07%** |
| **Precision**            | **95.88%** |
| **Recall**               | **83.73%** |
| **F1-Score**             | **89.40%** |
| Error Rate               |      9.93% |
| Best Epoch               |          4 |
| Best Validation Accuracy |     88.53% |

### Confusion Matrix

```text
                 Predicted
               FAKE    REAL

Actual FAKE     723      27
Actual REAL     122     628
```

### Interpretation

* **723** FAKE images were correctly classified as FAKE.
* **27** FAKE images were classified as REAL.
* **122** REAL images were classified as FAKE.
* **628** REAL images were correctly classified as REAL.

Overall, the model correctly classified **1,351 out of 1,500 test images**.

---

# 📈 Evaluation Performed

The project evaluates the trained model using several metrics:

### Accuracy

Measures the proportion of all test images classified correctly.

### Precision

Measures how many samples predicted as a particular class were actually members of that class.

### Recall

Measures how many samples belonging to a class were successfully identified.

### F1-Score

Provides a combined measure of precision and recall.

### Confusion Matrix

Provides a detailed view of correct and incorrect predictions for both classes.

### Prediction Confidence

The model's sigmoid output is used to examine how strongly the model supports the predicted class.

Images with predictions close to the **0.5 decision threshold** are treated as lower-confidence cases for further analysis.

---

# 🧪 Challenges Faced

Several practical challenges were encountered during development.

### Dataset and File Handling

Image files had to be validated before training, including checking file extensions, dimensions, colour modes and possible corruption.

### Model Graph and Grad-CAM Integration

Implementing Grad-CAM with the trained Keras Sequential model required additional handling of the model's intermediate layers.

A separate functional graph was constructed using the trained layer configurations and weights so that the final convolutional feature maps could be connected correctly to the model output.

### Output File Handling

The original project directory contained a deeply nested Windows path. Although the output directory existed and was writable, Matplotlib encountered a file-saving error.

A shorter output directory was therefore used for reliable storage of generated results.

---

# 📁 Project Structure

A recommended repository structure is:

```text
CIFAKE-Image-Classification-XAI/
│
├── README.md
│
├── CIFAKE_Analysis.ipynb
│
├── CIFAKE_CNN_Final_Model.keras
│
├── Results/
│   │
│   ├── Training_Accuracy.png
│   ├── Training_Loss.png
│   ├── Confusion_Matrix.png
│   ├── GradCAM_Example.png
│   ├── CIFAKE_Final_Results.csv
│   ├── CIFAKE_Confusion_Matrix.csv
│   ├── CIFAKE_Classification_Report.csv
│   └── CIFAKE_Test_Predictions.csv
│
└── Dataset/
    └── README.md
```

> **Dataset note:** The complete dataset does not need to be uploaded to GitHub. A dataset download/source link and instructions can be provided separately, depending on the dataset's licensing and redistribution terms.

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd CIFAKE-Image-Classification-XAI
```

## 2. Create a Python Environment

Using Anaconda:

```bash
conda create -n cifake python=3.12
```

Activate it:

```bash
conda activate cifake
```

## 3. Install Required Libraries

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow jupyter
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
CIFAKE_Analysis.ipynb
```

## 5. Configure the Dataset Path

Update the dataset path in the notebook according to your local system.

For example:

```python
dataset_path = r"D:\Path\To\CIFAKE\Dataset"
```

The expected structure is:

```text
Dataset/
├── FAKE/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── REAL/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## 6. Run the Notebook

Run the notebook cells sequentially.

The notebook performs:

1. Dataset validation
2. Dataset analysis
3. Preprocessing
4. Data augmentation
5. Train/validation/test splitting
6. CNN model creation
7. Model training
8. Model evaluation
9. Misclassification analysis
10. Grad-CAM visualization
11. Result generation
12. Model saving

---

# 💾 Output Files

The project generates several useful outputs.

| File                               | Description                        |
| ---------------------------------- | ---------------------------------- |
| `CIFAKE_CNN_Final_Model.keras`     | Trained CNN model                  |
| `Training_Accuracy.png`            | Training/validation accuracy graph |
| `Training_Loss.png`                | Training/validation loss graph     |
| `Confusion_Matrix.png`             | Confusion matrix visualization     |
| `GradCAM_Example.png`              | Grad-CAM explanation               |
| `CIFAKE_Final_Results.csv`         | Final performance metrics          |
| `CIFAKE_Confusion_Matrix.csv`      | Confusion matrix values            |
| `CIFAKE_Classification_Report.csv` | Classification metrics             |
| `CIFAKE_Test_Predictions.csv`      | Test-image predictions             |

---

# 🔮 Future Enhancements

Possible future improvements include:

* Testing higher image resolutions.
* Applying transfer learning using architectures such as ResNet, EfficientNet or MobileNet.
* Training with larger and more diverse datasets.
* Testing images generated by different AI image-generation techniques.
* Comparing multiple XAI techniques.
* Performing robustness testing under image compression and transformations.
* Building a Streamlit-based web interface.
* Providing real-time image upload and prediction.
* Displaying Grad-CAM explanations directly in the web application.

---

# 🎓 Learning Outcomes

This project provided practical experience in:

* Python programming
* Image processing
* Dataset preparation
* Data preprocessing
* Data augmentation
* CNN architecture design
* TensorFlow/Keras
* Deep learning model training
* Binary classification
* Model evaluation
* Confusion matrix analysis
* Prediction confidence analysis
* Explainable AI
* Grad-CAM
* Data visualization
* Model saving
* Debugging machine learning workflows

---

# ⚠️ Limitations

The current project has some limitations:

* The classifier is trained for the two classes represented in the CIFAKE dataset.
* The input images are resized to **32 × 32 pixels**, which may remove some fine-grained visual information.
* The model may not generalize equally well to AI-generated images from sources or generation methods that differ significantly from the training data.
* A classification prediction does not establish the actual origin of an image.
* Grad-CAM provides a model-based visual explanation rather than a definitive forensic proof.

Therefore, the model should be considered an experimental image-classification system rather than a standalone forensic verification tool.

---

# 👨‍💻 Author

**Niraj Khankari**

This project was developed as part of an internship project focused on **Deep Learning, Computer Vision and Explainable AI**.

---

# 📜 Project Information

**Project Title:**
CIFAKE Image Classification and Explainable Identification of AI-Generated Synthetic Images

**Domain:**
Artificial Intelligence / Machine Learning / Computer Vision / Explainable AI

**Model:**
Convolutional Neural Network (CNN)

**Explainability Technique:**
Grad-CAM

**Classification Type:**
Binary Image Classification

**Classes:**
FAKE / REAL

---

# ⭐ Key Highlight

This project combines **CNN-based image classification with Explainable AI** rather than stopping at a simple prediction.

The final system achieved:

> **90.07% accuracy on 1,500 test images**

and uses **Grad-CAM** to provide visual insight into the model's predictions.

---

## 📌 Disclaimer

This project is intended for educational and research purposes. The predictions should not be treated as definitive proof of whether an image was created or modified by AI. Model performance depends on the dataset, preprocessing, training procedure and similarity between the training and real-world data.
