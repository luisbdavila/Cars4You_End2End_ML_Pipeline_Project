# Cars4You_End2End_ML_Pipeline_Project

Github Link: https://github.com/luisbdavila/Cars4You_End2End_ML_Pipeline_Project

Contains the complete implementation of the Cars4You machine learning pipeline project. An end-to-end workflow — from data preprocessing to model evaluation.

It was implemented using: python 3.13.3. However, with previous (or other) versions should also work.

The needed libraries are: matplotlib, seaborn, numpy, pandas, tqdm(python native), warnings(python native) and sklearn.

To run the code it expect to have a folder structure like this (if not, change the imports and/or folder paths on the jupyter). And the order tu run the nootebook is:
1) Cars4you_EDA_Preprocessing.ipynb.
2) Cars4you_Modeling.ipynb
3) Cars4you_Brand_Modeling.ipynb

```plaintext
CARS4YOU_END2END_ML_PIPELINE_PROJECT/
├── Code/
│   ├── predictions/                      # Folder to store the predictions of the Deployment
│   │
│   ├── Cars4you_Brand_Modeling.ipynb     # Notebook: Brand-specific modeling/analysis
│   ├── Cars4you_EDA_Preprocessing.ipynb  # Notebook: Exploratory Data Analysis and initial preprocessing
│   ├── Cars4you_Modeling.ipynb           # Notebook: Main modeling/training experiments
│   ├── utils.py                          # Utility functions for the project
│   └── final_model_col_dics.pkl          # Final objects fitted when predict using brands
│
├── Project Guidelines/                   # Folder for project statement
│   ├── HANDOUT_ML_Project_25_26.pdf      # Information of 1-delivery
│   └── ML_Project_25_26.pdf              # Information final-delivery
│
├── project_data/                         # Folder for datasets, gridsearch experiemnts and fitted objects
│   ├── DicBrand_DicTrans..._PrePro.pkl   # Preprocessing metadata/dictionaries (e.g., column mapping)
│   ├── imputers_scalars_experiments.csv  # Data used/produced during imputation/scaling experiments
│   ├── rfe_experiments.csv               # Data/results from Recursive Feature Elimination experiments
│   ├── sample_submission.csv             # Sample submission file
│   ├── test.csv                          # Test dataset (to summit predictions on kaggle)
│   ├── Train_Val_...ModelHO.pkl          # Entire X and log(Y) dataset used for HO, include encoder, scaler and imputation
│   ├── train.csv                         # Training dataset
│   ├── X_Ylog_ModelOptimization.pkl      # Entire X and log(Y) dataset used for CV
│   └── X_Ylog_Scale_Impute_ModelFull.pkl # Entire X and log(Y) dataset, including encoder, scaler and imputation
│
├── .gitignore
└── README.md
```

## Abstract
The rapid growth of Cars 4 You, a company relying on physical inspection by certified mechanics to determine car purchase prices, has led to increased customer waiting times and loss of market share. The primary goal of this project was to develop a machine learning solution to automate the preliminary car valuation process, thereby providing instant, data-driven price estimates to streamline operations and enhance customer experience.

To create a final solution, different model hyperparameters and preprocessing decisions were tested using Grid Search. This resulted in a comprehensive preprocessing pipeline and model choice.

Key steps included addressing missing values using a K-Nearest Neighbors Imputer (with k=10) for numerical features and mode imputation for categorical features. Robust Scaler was selected as the optimal scaling method. Feature engineering involved creating the Years_old variable, and the final model selection utilized a subset of features: mileage, mpg, engineSize, Years_old, transmission_manual, and transmission_semi-auto.

Instead of a single global model, the most effective approach was found to be training individual models for each specific car brand. The models chosen for the different brands include Random Forests, Bagging of Decision Trees, and K-Nearest Neighbors. This brand-specific strategy, tested using cross-validation (CV), achieved significantly superior performance compared to a general model.

The main result was a final system that delivered a mean Validation MAE (Mean Absolute Error) of £1662, with a low standard deviation of £17.67, and an overfitting gap of £136. The conclusion is that the brand-specific modeling approach successfully created a robust and highly accurate preliminary valuation system. This solution enables Cars 4 You to reduce dependency on manual inspections.

## Objectives
The primary objective of this project was to address the operational bottleneck caused by manual car inspections.

+ **Automate Valuation:** Develop a Machine Learning model capable of providing instant preliminary car price estimates.

+ **Reduce Wait Times:** Minimize the time customers spend waiting for a quote, improving overall satisfaction.

+ **Optimize MAE:** Minimize the Mean Absolute Error (MAE) of predictions to ensure the estimated prices are competitive and safe for the business.

## Dataset
The project utilizes a dataset of used cars containing various attributes relevant to valuation.

+ **Source:** train.csv (Training data) and test.csv (unseen test data for Kaggle submission).

+ **Target Variable:** price (The selling price of the car).

+ **Features:**

    + **Numerical:** mileage, mpg (miles per gallon), engineSize, tax, year of registration, paintQuality%, previousOwners, hasDamage.

    + **Categorical:** brand, model, transmission, fuelType.

    + **Engineered:** Years_old (Calculated from the registration year to represent vehicle age).

## Methodology
The solution was built using a rigorous experimental framework:

### Data Cleaning & Imputation:

**Numerical:** Handled using KNNImputer (k=10) to preserve local data structures.

**Categorical:** Missing values filled using the mode (most frequent value).

### Feature Engineering:

Creation of Years_old to better capture the depreciation factor than raw year.

### Preprocessing:

**One-Hot Encoding** for categorical variables.

**Scaling:** RobustScaler was chosen over Standard or MinMax scalers to handle outliers effectively (reducing their effect).

**Feature Selection:** Recursive Feature Elimination (RFE), LASSO, and DT feature importance identified the most predictive features: mileage, mpg, engineSize, Years_old, and specific transmission types.

### Model Selection Strategy:

Extensive Grid Search was performed to tune hyperparameters.

Brand-Specific Modeling: Hypothesis testing revealed that training separate models for each manufacturer (e.g., Audi, BMW, Toyota) yielded better results than a single global model.

Algorithms Used: Linear Regression as beanchmark, Desicion Trees (DT), Neural Networks (NN), Random Forest, Bagging of DT, AdaBoost, and K-Nearest Neighbors (KNN), selected dynamically based on the brand's data characteristics.

### Results
The final brand-specific pipeline demonstrated robust performance on the validation set:

+ Mean Absolute Error (MAE): £1662.

+ Standard Deviation: £17.67 (indicating stable performance across folds).

+ Overfitting Gap: £136 (difference between Train and Validation MAE), suggesting the model generalizes well to unseen data.

This approach significantly outperformed the baseline general model, validating the decision to segment the problem by car manufacturer.

## Conclusion

The project successfully delivered a high-performance ML pipeline for Cars 4 You. By shifting from a manual inspection-first model to a data-driven preliminary valuation, the company can now offer instant quotes to customers. The use of brand-specific models combined with Robust Scaling and KNN Imputation proved to be the winning strategy, offering a balance of accuracy and generalization that will directly impact the company's efficiency and market share.
