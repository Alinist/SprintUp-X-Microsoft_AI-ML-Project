# Heart Disease Risk Prediction - Machine Learning Pipeline

## Project Overview

This project implements a **comprehensive machine learning pipeline** for analyzing and predicting heart disease risk using the **Heart Disease UCI Dataset**. The goal is to leverage various supervised and unsupervised learning techniques, alongside feature engineering and dimensionality reduction, to build accurate predictive models and facilitate insightful data exploration.

The pipeline includes:

- Data preprocessing and cleaning
- Feature selection via statistical and ML-based methods
- Dimensionality reduction using PCA
- Supervised classification with multiple algorithms
- Unsupervised clustering to identify patterns
- Hyperparameter tuning to optimize models
- Model export for deployment
- Bonus: Streamlit UI for interactive predictions and Ngrok for public access

---

## Project Objectives

- **Data Preprocessing & Cleaning**  
    Handle missing values, encode categorical variables, and scale features for optimal modeling.
    
- **Dimensionality Reduction**  
    Use PCA to reduce dimensionality while preserving important variance.
    
- **Feature Selection**  
    Apply Random Forest feature importance, Recursive Feature Elimination (RFE), and Chi-Square tests.
    
- **Supervised Learning**  
    Train and evaluate Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost classifiers.
    
- **Unsupervised Learning**  
    Perform K-Means and Hierarchical Clustering to discover inherent data structures.
    
- **Hyperparameter Tuning**  
    Optimize model parameters with GridSearchCV and RandomizedSearchCV.
    
- **Deployment** (Bonus)  
    Develop a Streamlit app for real-time prediction and deploy it using Ngrok.
    

---

## Technologies & Libraries Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Joblib (model saving/loading)
- Streamlit (UI, bonus)
- Ngrok (deployment, bonus)

---

## Project Structure

```
SprintUp-X-Microsoft_AI-ML-Project/
│── data/
│   ├── heart_disease.csv
│   ├── cleaned_heart_disease.csv
│   ├── reduced_features_has_disease.csv
│   ├── visualized_heart_disease.csv
│   ├── pca_train.csv
│   ├── pca_test.csv
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   ├── y_test.pkl
│── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│── models/
│   ├── all_models/
│   │    ├── Logistic_Regression.pkl
│   │    ├── Logistic_Regression_PCA.pkl
│   │    ├── Logistic_Regression_tuned.pkl
│   │    ├── Logistic_Regression_PCA_tuned.pkl
│   │    ├── Decision_Tree.pkl
│   │    ├── Decision_Tree_PCA.pkl
│   │    ├── Decision_Tree_tuned.pkl
│   │    ├── Decision_Tree_PCA_tuned.pkl
│   │    ├── Random_Forest.pkl
│   │    ├── Random_Forest_PCA.pkl
│   │    ├── Random_Forest_tuned.pkl
│   │    ├── Random_Forest_PCA_tuned.pkl
│   │    ├── SVM.pkl
│   │    ├── SVM_PCA.pkl
│   │    ├── SVM_tuned.pkl
│   │    ├── SVM_PCA_tuned.pkl
│   │    ├── XGBoost.pkl
│   │    ├── XGBoost_PCA.pkl
│   │    ├── XGBoost_tuned.pkl
│   │    ├── XGBoost_PCA_tuned.pkl
│   ├── best_model.pkl
│   ├── label_encoders.pkl
│   ├── pca_transformer.pkl
│   ├── scaler.pkl
│── ui/
│   ├── app.py
│── deployment/
│   ├── ngrok_setup.txt
│── results/
│   ├── evaluation_metrics.txt
│── README.md
│── requirements.txt
│── .gitignore
```

---

## Detailed Workflow & Deliverables

### 1. Data Preprocessing & Cleaning

- Load the raw Heart Disease UCI dataset
- Handle missing values via imputation/removal
- Encode categorical variables with one-hot encoding
- Scale numeric features with StandardScaler/MinMaxScaler
- Exploratory Data Analysis (histograms, correlation heatmaps, boxplots)  
    **Deliverable:** Cleaned dataset ready for modeling (`cleaned_heart_disease.csv`)

---

### 2. Dimensionality Reduction - PCA

- Apply PCA on cleaned data to reduce feature space    
- Analyze explained variance to select optimal components
- Visualize PCA components and variance retained  
    **Deliverable:** PCA-transformed datasets (`pca_train.csv`, `pca_test.csv`), variance plots

---

### 3. Feature Selection

- Rank features by Random Forest importance and XGBoost scores    
- Use Recursive Feature Elimination (RFE) to select top predictors
- Apply Chi-Square Test for feature significance
- Select most relevant features combining above methods  
    **Deliverable:** Reduced dataset with key features (`reduced_features_has_disease.csv`), importance plots

---

### 4. Supervised Learning - Classification

- Train Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost on PCA data    
- Evaluate with accuracy, precision, recall, F1-score, ROC AUC
- Visualize ROC curves for model comparison  
    **Deliverable:** Trained baseline models, evaluation metrics summary

---

### 5. Unsupervised Learning - Clustering

- Apply K-Means clustering with elbow method to find optimal K    
- Perform Hierarchical clustering and plot dendrogram
- Compare cluster assignments with actual labels  
    **Deliverable:** Clustering visualizations, cluster-label comparisons

---

### 6. Hyperparameter Tuning

- Use GridSearchCV and RandomizedSearchCV to optimize hyperparameters for all models    
- Compare tuned model performance with baseline
- Save best performing model for deployment  
    **Deliverable:** Optimized models, hyperparameter search results, saved best model (`best_model.pkl`)

---

### 7. Model Export & Deployment

- Save models using `joblib` in `.pkl` format    
- Ensure preprocessing + model reproducibility  
    **Deliverable:** Exported models ready for deployment

---

### 8. Streamlit Web UI [Bonus]

- Create an interactive app to input user health data    
- Display real-time prediction results and relevant visualizations  
    **Deliverable:** Functional Streamlit app (`ui/app.py`)

---

### 9. Deployment with Ngrok [Bonus]

- Deploy Streamlit app locally and create a public URL with Ngrok
- Share live app access link  
    **Deliverable:** Publicly accessible web app, deployment instructions (`deployment/ngrok_setup.txt`)

---

### 10. Project Upload & Documentation

- GitHub repository with all source code, notebooks, data, models, UI, and docs    
- `requirements.txt` for environment setup
- README with project overview and usage instructions

---

## Dataset

- **Heart Disease UCI Dataset**  
    Available publicly [here](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)    

---

## How to Run

1. Clone the repository
2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
3. Run notebooks sequentially from `01_data_preprocessing.ipynb` to `06_hyperparameter_tuning.ipynb` for full pipeline
4. Launch Streamlit UI
    ```bash
    streamlit run ui/app.py
    ```
5. Use Ngrok (optional) for external access as per `deployment/ngrok_setup.txt`
