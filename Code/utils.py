# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# Functions


def avg_scores(X,
               y,
               CV,
               imputer,
               scalar,
               model,
               score_train_dic,
               score_val_dic,
               dic_key,
               log_transform=False):

    # create lists to store the results from the different models
    score_train = []
    score_val = []
    # do the interaction for every k fold
    for train_index, test_index in tqdm(CV.split(X, y)):
        # split
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # scale
        scaler_ = scalar
        X_train_scl = scaler_.fit_transform(X_train)
        X_test_scl = scaler_.transform(X_test)
        # impute
        imputer_ = imputer
        X_train_scl_imp = imputer_.fit_transform(X_train_scl)
        X_test_scl_imp = imputer_.transform(X_test_scl)
        # fit and predict
        model_ = model
        modelfit = model_.fit(X_train_scl_imp, y_train)
        pred_train = modelfit.predict(X_train_scl_imp)
        pred_test = modelfit.predict(X_test_scl_imp)
        # if log transform, revert the transformation to normal scale
        if log_transform:
            pred_train = np.exp(pred_train)
            y_train = np.exp(y_train)
            pred_test = np.exp(pred_test)
            y_test = np.exp(y_test)
        # calculate MAE
        score_train.append(mean_absolute_error(y_train, pred_train))
        score_val.append(mean_absolute_error(y_test, pred_test))

    # calculate the average and the std
    avg_train = round(np.mean(score_train), 2)
    avg_val = round(np.mean(score_val), 2)
    std_train = round(np.std(score_train), 2)
    std_val = round(np.std(score_val), 2)

    score_train_dic[dic_key] = [avg_train, std_train]
    score_val_dic[dic_key] = [avg_val, std_val]
