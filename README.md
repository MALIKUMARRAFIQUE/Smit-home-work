# Framingham Heart Study – 10-Year CHD Risk Prediction

## 📌 Project Overview
This project implements a **Machine Learning classification pipeline** to predict the **10-year risk of coronary heart disease (CHD)** using the Framingham Heart Study dataset. The pipeline includes preprocessing, handling missing values, feature scaling, class imbalance handling, model training, comparison, and hyperparameter optimization.

## 📊 Dataset
- **Source**: Framingham Heart Study
- **Samples**: 4,240 patient records
- **Features**: 16 clinical and demographic features
- **Target**: `TenYearCHD` (0 = No CHD, 1 = CHD)
- **Imbalance**: ~85% negative, ~15% positive (handled using SMOTE)

## 🧠 Methodology
1. **Data Preprocessing**
   - Missing values imputed with median
   - No encoding needed (all numerical)
   
2. **Feature Scaling**
   - `StandardScaler` applied to all features

3. **Train-Test Split**
   - 80-20 stratified split

4. **Handling Imbalance**
   - SMOTE applied on training data only

5. **Models Implemented**
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
   - XGBoost

6. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-Score
   - Classification Report
   - Confusion Matrix

7. **Model Comparison & Optimization**
   - Selected best model based on F1-Score
   - Hyperparameter tuning using GridSearchCV

## 📈 Results (Example)
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.78 | 0.72 | 0.65 | 0.68 |
| Random Forest | 0.82 | 0.76 | 0.70 | 0.73 |
| SVM | 0.79 | 0.73 | 0.67 | 0.70 |
| **XGBoost** | **0.85** | **0.80** | **0.75** | **0.77** |

**Tuned XGBoost Performance**:  
- Accuracy: **0.87**  
- F1-Score: **0.80**  
- Recall: **0.78**  
- Precision: **0.82**

## 🛠️ Requirements
