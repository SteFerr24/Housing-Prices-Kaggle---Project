# Housing Prices Prediction – Kaggle Competition

Machine learning project developed in **R** for the Kaggle competition **Housing Prices: Competition for Kaggle Learn Users**.  
The objective of this project is to predict residential property prices using structured real estate data and advanced machine learning techniques.

**Final ranking:** **227 / 4580 participants (Top ~5%)**

---

# Project Overview

This project implements a complete **end-to-end machine learning pipeline** for tabular regression.  
The workflow includes:

- data preprocessing
- missing value handling
- feature engineering
- feature transformation
- model training and hyperparameter tuning
- model comparison
- ensemble learning
- stacking
- final prediction generation for Kaggle submission

The pipeline was implemented entirely in **R**, using several machine learning libraries and extensive feature engineering.

---

# Dataset

The dataset is provided by the Kaggle competition and contains detailed information about residential properties such as:

- property size
- building quality
- construction year
- basement and garage characteristics
- neighborhood information
- additional structural features

The target variable is:

**SalePrice** – the final sale price of the house.

To improve modeling performance, the target variable was transformed using a **log transformation**.

---

# Data Preprocessing

Several preprocessing steps were applied to prepare the dataset for machine learning models.

### Handling Missing Values

Missing values were treated using **domain-specific logic**.

**Structural absence imputation**

Some categorical variables where missing values represent the absence of a feature were replaced with `"None"`:

- basement related features
- fireplace quality
- garage attributes
- pool quality
- fence
- miscellaneous features

**Numeric absence replacement**

For numerical variables related to missing structures, missing values were replaced with **0**:

- basement surface variables
- garage size and year
- masonry veneer area

**Neighborhood-based imputation**

`LotFrontage` was imputed using the **median value within each neighborhood**.

**Remaining missing values**

- categorical variables → imputed using the **mode**
- numerical variables → imputed using the **median**

---

# Feature Engineering

A significant part of the project focused on creating additional features to improve predictive power.

### Structural features

New variables were created including:

- **TotalSF**  
  Total house surface including basement and upper floors.

- **TotalBathrooms**  
  Weighted sum of full and half bathrooms.

- **HouseAge**  
  Age of the house at the time of sale.

- **RemodAge**  
  Years since the last remodeling.

- **Remodeled**  
  Binary indicator showing whether the house was remodeled.

---

### Interaction features

- **QualLivArea**  
  Interaction between overall building quality and living area.

---

### Exterior space features

- **TotalPorchSF**  
  Combined surface of all porch-related areas.

---

### Presence indicators

Binary variables were created to capture the presence of certain structural components:

- garage
- basement
- fireplace
- pool
- low quality finished area

These variables help tree-based models capture structural differences between houses.

---

# Ordinal Encoding

Several ordinal categorical variables were converted into **ordered numeric variables** based on domain knowledge.

Examples include:

- exterior quality
- basement quality
- heating quality
- kitchen quality
- garage quality
- pool quality

Each variable was mapped to a numeric scale representing increasing quality levels.

---

# Feature Transformation

To reduce skewness and stabilize variance, **log transformations** were applied to several highly skewed variables including:

- lot area
- living area
- basement surface
- porch surface
- garage area
- masonry veneer area

The transformation used was:

`log1p(x)`

which improves stability for variables with zero values.

---

# Multicollinearity Reduction

To avoid redundant predictors and reduce model complexity:

1. A **correlation matrix** was computed for all numerical variables.
2. Highly correlated variables were identified.
3. Predictors with correlation greater than **0.95** were removed using:

`caret::findCorrelation()`

---

# Encoding Categorical Variables

Categorical variables were transformed using **one-hot encoding** through:

`model.matrix()`

This produced a fully numeric dataset suitable for machine learning models.

---

# Outlier Detection

Extreme outliers were removed to prevent distortion in model training.

In particular, houses with:

- extremely large living area
- unusually low sale price

were filtered out.

---

# Machine Learning Models

Several models were trained and evaluated using cross-validation.

### LASSO Regression

Regularized linear regression using **glmnet**.

Used to perform feature selection and shrink coefficients of less relevant predictors.

---

### Decision Tree

Regression tree model implemented using **rpart**.

Provides a simple interpretable baseline model.

---

### Random Forest

Implemented with the **ranger** package.

Advantages:

- robust to noise
- handles nonlinear relationships
- captures feature interactions

---

### Gradient Boosting Machine

Implemented using **gbm**.

Boosting iteratively improves model performance by focusing on difficult observations.

---

### Neural Network

A feedforward neural network implemented using **nnet**.

Hyperparameters tuned included:

- hidden layer size
- regularization parameter

---

### XGBoost

Extreme Gradient Boosting was the **best performing individual model**.

Advantages include:

- strong performance on tabular data
- regularization to prevent overfitting
- efficient tree boosting

Hyperparameters tuned included:

- number of trees
- tree depth
- learning rate
- column sampling
- subsampling

---

# Model Evaluation

All models were evaluated using **cross-validation**.

Performance metric:

**Root Mean Squared Error (RMSE)**

Example results:

| Model | RMSE |
|------|------|
| Decision Tree | 0.0156 |
| Random Forest | 0.0104 |
| Neural Network | 0.0105 |
| LASSO | 0.00963 |
| Gradient Boosting | 0.00956 |
| XGBoost | **0.00902** |

---

# Ensemble Learning

To further improve performance, ensemble techniques were implemented.

### Weighted Ensemble

Predictions from multiple models were combined using weighted averaging:

0.6 * XGBoost  
0.25 * GBM  
0.15 * LASSO  

---

# Model Stacking

Stacking was implemented using a two-level approach.

### Level 1 – Base Models

Out-of-fold predictions were generated for:

- XGBoost
- Gradient Boosting
- LASSO

using K-fold cross-validation.

---

### Level 2 – Meta Model

The predictions from the base models were used as inputs to a **linear regression meta-model**, which learned the optimal combination of model predictions.

---

# Final Predictions

Final models were retrained on the **full training dataset**.

Predictions on the test set were generated using:

- XGBoost
- Gradient Boosting
- LASSO

The stacked meta-model produced the final predictions.

Because the target variable was log-transformed, predictions were converted back using:

`expm1()`

---

# Technologies Used

Language:

**R**

Main libraries:

- caret
- xgboost
- ranger
- gbm
- glmnet
- nnet
- dplyr
- ggplot2
- corrplot

---

# Repository Structure

```
housing-prices-prediction-kaggle-r
│
├── data
│
├── src
│   ├── data_cleaning.R
│   ├── feature_engineering.R
│   ├── model_training.R
│   ├── ensemble.R
│   └── stacking.R
│
├── outputs
│   └── submissions
│
└── README.md
```

---

# Author
FERRIGNO STEFANO 
