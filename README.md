# DA5401 Assignment 8: Ensemble Learning for Complex Regression Modeling on Bike Share Data

- **Name**: Rishabh Gupta 
- **Roll Number**: DA25M024
---

## Overview

This assignment focuses on applying and comparing **ensemble learning techniques** — **Bagging**, **Boosting**, and **Stacking** — to solve a **complex regression task** using the **Bike Sharing Demand Dataset**. The objective is to accurately predict the total count of rented bikes (`cnt`) based on various temporal and environmental factors such as weather conditions, season, and time of day.

Through this project, we aim to explore how ensemble methods address **bias-variance trade-offs** and how combining diverse models can lead to superior predictive performance compared to single models.

---

## Dataset

- **Dataset Name**: UCI Bike Sharing Demand Dataset (Hourly Data)  
- **Source**: UCI Machine Learning Repository — Fanaee-T, Hadi, and Gama, J. (2014)  
- **Description**: Contains over 17,000 hourly records of bike rentals influenced by weather, temperature, humidity, windspeed, and time features.  
- **Target Variable**: `cnt` — total count of rented bikes per hour  
- **Key Features**:
  - Temporal: `season`, `mnth`, `hr`, `weekday`, `workingday`, `holiday`
  - Environmental: `temp`, `atemp`, `hum`, `windspeed`
  - Categorical features encoded using **One-Hot Encoding**

---

## Essential Files for Evaluation

1. **Assignment 8.ipynb**  
   – Main Jupyter Notebook containing:
     - Data preprocessing and feature engineering  
     - Baseline model implementation (Linear Regression and Decision Tree)  
     - Bagging, Boosting, and Stacking ensemble models  
     - RMSE calculation and comparison  
     - Final analysis with bias–variance interpretation  

2. **hour.csv**  
   – Dataset file containing hourly bike-sharing records used for training and testing.  

3. **README.md**  
   – This documentation file summarizing the assignment, dataset, methods, and conclusions.

---

## Implementation Details

### Key Libraries Used
- **pandas**, **numpy** → Data manipulation and analysis  
- **matplotlib**, **seaborn** → Data visualization and insight generation  
- **scikit-learn** → Machine learning models, preprocessing, and performance metrics  

### Models Implemented
1. **Linear Regression** — Baseline model for comparison  
2. **Decision Tree Regressor** — Single non-linear model (baseline alternative)  
3. **Bagging Regressor** — Variance reduction ensemble  
4. **Gradient Boosting Regressor** — Bias reduction ensemble  
5. **Stacking Regressor** — Meta-learning ensemble combining multiple models  

---

## Approach

### Part A: Data Preprocessing and Baseline
- Loaded and explored the dataset (`hour.csv`)  
- Dropped irrelevant columns: `instant`, `dteday`, `casual`, `registered`  
- Converted categorical variables (`season`, `mnth`, `hr`, `weekday`, etc.) using **One-Hot Encoding**  
- Performed train-test split for model evaluation  
- Built baseline models (Linear Regression, Decision Tree Regressor) and recorded RMSE  

### Part B: Ensemble Techniques
- **Bagging Regressor**: Implemented using Decision Tree as base estimator with 50 estimators to reduce variance.  
- **Gradient Boosting Regressor**: Implemented to iteratively minimize bias through sequential learning.  
- Both models evaluated using **Root Mean Squared Error (RMSE)** on test data.  

### Part C: Stacking for Optimal Performance
- Defined base learners (KNN, Bagging Regressor, Gradient Boosting Regressor)  
- Used **Ridge Regression** as the meta-learner (Level-1 model)  
- Implemented **Stacking Regressor** to optimally combine base learner predictions  
- Evaluated the model’s RMSE and compared performance with other methods  

### Part D: Final Analysis
- Compiled a performance summary table of all models  
- Interpreted results using **bias–variance trade-off** and **model diversity** concepts  
- Identified the best-performing model and provided theoretical justification  

---

## Key Results

### RMSE Comparison Summary:
| Model | RMSE | Type |
|--------|------|------|
| Baseline Single Model (Linear Regression) | 100.4459 | Single Model |
| Bagging Regressor | 112.3430 | Ensemble (Variance Reduction) |
| Gradient Boosting Regressor | 55.4251 | Ensemble (Bias Reduction) |
| Stacking Regressor | **53.1738** | Ensemble (Meta-Learning) |

### Key Observations:
- The **Stacking Regressor** achieved the **lowest RMSE (53.17)**, outperforming all other models.  
- **Gradient Boosting** significantly improved over the baseline by reducing bias.  
- **Bagging** did not outperform the baseline due to high bias in Decision Tree base estimators.  
- **Stacking** effectively combined multiple learners to achieve an optimal bias–variance balance.

---

## Final Conclusion

### a) Best-Performing Model
The **Stacking Regressor** is the best-performing model with the lowest RMSE (53.17), demonstrating superior predictive capability over the baseline and other ensemble techniques.

### b) Explanation of Superior Performance
The Stacking Regressor effectively integrates multiple base learners with diverse learning mechanisms — **KNN**, **Bagging**, and **Gradient Boosting** — using a **meta-learner (Ridge Regression)** to combine their predictions optimally.  
This hybrid structure captures complex non-linear relationships, reduces overfitting, and balances **low bias (from boosting)** with **low variance (from bagging)**.  

Hence, the **Stacking ensemble** achieves the best generalization and predictive accuracy by leveraging **model diversity** and managing the **bias–variance trade-off** efficiently.

---

## Technical Contributions

- Comprehensive implementation of **Bagging**, **Boosting**, and **Stacking** ensembles  
- Clear demonstration of the **bias–variance trade-off** using RMSE metrics  
- Insightful comparison of single and ensemble models  
- Structured visualization and clean analysis in Jupyter Notebook  
- Reproducible workflow with proper documentation  

---

## How to Run

1. Ensure all required libraries are installed:  
   `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`  
2. Place `hour.csv` in the same directory as `Assignment 8.ipynb`  
3. Run the notebook sequentially from top to bottom  
4. The notebook will automatically preprocess the dataset, train all models, and display RMSE results and conclusions  


