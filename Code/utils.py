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


def graph_actual_vs_predicted(model, X_train, y_train, X_val, y_val):

    # 1. Train the model
    model.fit(X_train, y_train)

    # 2. Make predictions & Inverse Transform (Log -> Real Money)
    # Training Data
    y_train_pred = np.exp(model.predict(X_train))
    y_train_actual = np.exp(y_train)

    # Validation Data
    y_val_pred = np.exp(model.predict(X_val))
    y_val_actual = np.exp(y_val)

    # 3. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Define data pairs for iteration to avoid code repetition
    plot_data = [
        (axes[0], y_train_actual, y_train_pred, "Training Set"),
        (axes[1], y_val_actual, y_val_pred, "Validation Set")
    ]

    for ax, y_true, y_pred, title in plot_data:
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')

        # dynamic limits for the perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', lw=2, label='Perfect prediction')

        ax.set_xlabel('Actual Price (£)', fontsize=11)
        ax.set_ylabel('Predicted Price (£)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # Get the model name dynamically
    model_name = model.__class__.__name__
    plt.suptitle(f'{model_name} - Actual vs Predicted Price (Original Scale)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def model_performance(model, x_train, y_train, x_val, y_val):

    model.fit(x_train, y_train)

    print('Train MAE:', mean_absolute_error(np.exp(y_train),
                                            np.exp(model.predict(x_train))))
    print('Validation MAE:', mean_absolute_error(np.exp(y_val),
                                                 np.exp(model.predict(x_val))))


def grid_score(x_train,
               y_train,
               x_val,
               y_val,
               model,
               score_train_dic,
               score_val_dic,
               dic_key,
               log_transform=False):

    # fit and predict
    model_ = model
    modelfit = model_.fit(x_train, y_train)
    pred_train = modelfit.predict(x_train)
    pred_val = modelfit.predict(x_val)

    # if log transform, revert the transformation to normal scale
    if log_transform:
        pred_train = np.exp(pred_train)
        y_train_real = np.exp(y_train)
        pred_val = np.exp(pred_val)
        y_val_real = np.exp(y_val)
    else:
        y_train_real = y_train
        y_val_real = y_val

    # Calculate and save MAE
    score_train_dic[dic_key] = mean_absolute_error(y_train_real, pred_train)
    score_val_dic[dic_key] = mean_absolute_error(y_val_real, pred_val)


def print_cv_results(key, dic_train_MAE, dic_val_MAE):
    print(f'CV Results - {key}')
    print(f'Train MAE: {dic_train_MAE[key][0]}, Train std: {dic_train_MAE[key][1]}')
    print(f'Validation MAE: {dic_val_MAE[key][0]}, Validatin std: {dic_val_MAE[key][1]}')
