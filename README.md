# ECG Heartbeat Classification

This project is an **End-to-End deep Learning Pipeline** for classifying heartbeats from ECG (Electrocardiogram) signals into 5 categories. It uses a **1D Convolutional Neural Network (CNN)** to automatically learn patterns from ECG waveforms.

---

## 📌 Project Overview

- **Goal:** Classify individual ECG heartbeats into one of five types:
  | Label | Code | Description |
  |-------|------|-------------|
  | 0     | N    | Normal beat |
  | 1     | S    | Supraventricular ectopic beat |
  | 2     | V    | Ventricular ectopic beat |
  | 3     | F    | Fusion beat |
  | 4     | Q    | Unknown beat |

- **Datasets Used:**
  - **MIT-BIH Arrhythmia Dataset** – for arrhythmia classification.
  - **PTB Diagnostic ECG Database** – for distinguishing normal vs. myocardial infarction cases.

- **Each sample:** A single heartbeat, preprocessed into a **fixed-length signal of 187 values**.

---

## 🛠 Project Steps

1. **Data Loading**
   - Automatically searches the project folder and Desktop for CSV files (`mitbih_train*.csv` and `mitbih_test*.csv`).
   - Loads the datasets into Pandas DataFrames.

2. **Exploratory Data Analysis (EDA)**
   - Inspects the dataset (`head()`, `info()`, null values, duplicates).  
   - Visualizes class distribution and percentage of samples per class.  
   - Plots the **average ECG waveform per class** to understand signal patterns.

3. **Data Preprocessing**
   - Split features (`X`) and labels (`y`).  
   - Stratified train/validation/test split.  
   - Standard scaling using `StandardScaler`.  
   - Reshape data for 1D CNN input.  
   - Compute **class weights** to handle imbalance.

4. **Model Architecture**
   - Deep **1D CNN** with multiple Conv1D layers, BatchNorm, LeakyReLU, MaxPooling, and Dropout.  
   - Dense layers for class-specific feature learning.  
   - Output layer with `softmax` for 5-class classification.  
   - Compiled with `Adam` optimizer and `sparse_categorical_crossentropy` loss.  
   - Callbacks: `EarlyStopping` and `ReduceLROnPlateau`.

5. **Model Training**
   - Trains the model with training and validation sets.  
   - Visualizes **loss and accuracy curves** to monitor convergence and overfitting.

6. **Model Evaluation**
   - Predicts on the test set.  
   - Computes **accuracy, precision, recall, F1-score**.  
   - Generates **confusion matrix** and **normalized confusion matrix**.  
   - Highlights which heartbeat classes are well-predicted and which are challenging.

---

## 📈 Results

Example performance on the test set:
Accuracy : 0.9186
Precision: 0.9618
Recall   : 0.9186
F1 Score : 0.9338


- Confusion matrices help identify class-wise prediction quality.  
- Classes with fewer samples (e.g., F, S) may have lower precision but high recall due to class imbalance.

---

## 🛠 Requirements

- Python 3.x, (tested on 3.12.3)
- Libraries:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`
  - `scikit-learn`
  - `tensorflow` (>=2.x)

## 📬 Contact  
Created by FELLAH HANANE

📧 Email: hananefellah35@gmail.com

🌐 GitHub: hananefellah

## 📄 License  
MIT License  

Feel free to reach out for questions, collaborations, or suggestions!

Install dependencies with:

```bash
pip install -r requirements.txt

---

