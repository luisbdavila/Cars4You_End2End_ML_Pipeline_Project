# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from tqdm import tqdm

from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator, TransformerMixin


# Functions

def avg_scores(X: pd.DataFrame,
               y: pd.Series,
               CV: BaseCrossValidator,
               imputer: TransformerMixin,
               scalar: TransformerMixin,
               model: BaseEstimator,
               score_train_dic: dict,
               score_val_dic: dict,
               dic_key: str | tuple | int,
               log_transform: bool = False) -> None:
    """
    Cross-Validated Mean Absolute Error (MAE) Computation
    -----------------------------------------------------
    Evaluates a machine learning model using cross-validation. For each fold,
    the function preprocesses the data (encoding categorical variables,
    scaling, imputing missing values), fits the model, makes predictions,
    and computes MAE on both training and validation sets. If log-transformed
    targets are used, predictions and targets are exponentiated back to the
    original scale before calculating MAE. Stores the average MAE and standard
    deviation across folds.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix containing numerical and categorical columns.
    y : pd.Series
        Target vector (log-transformed if log_transform=True).
    CV : BaseCrossValidator
        Cross-validation splitter (e.g., KFold, RepeatedKFold) defining folds.
    imputer : TransformerMixin
        Imputer for handling missing numerical values (e.g., KNNImputer).
    scalar : TransformerMixin
        Scaler for standardizing numerical features (e.g., StandardScaler).
    model : BaseEstimator
        ML model implementing fit() and predict() (e.g., RandomForestRegressor)
    score_train_dic : dict
        Dictionary where training results [mean_MAE, std_MAE] will be stored.
    score_val_dic : dict
        Dictionary where validation results [mean, std] will be stored.
    dic_key : str | tuple | int
        Key under which results are saved.
    log_transform : bool
        Whether to reverse log-transform before MAE.

    Returns
    -------
    None
        Results are stored in the provided dictionaries.
    """

    # Define the Encoding for Categorical Variables
    # Transmission OHE
    ohe_transmission = OneHotEncoder(
        categories=[['automatic', 'manual', 'semi-auto']],
        drop=['automatic'],
        handle_unknown='ignore',
        sparse_output=False,
        dtype=int
        )
    # FuelType OHE
    ohe_fuel = OneHotEncoder(
        categories=[['diesel', 'other', 'petrol']],
        drop=['diesel'],
        handle_unknown='ignore',
        sparse_output=False,
        dtype=int
        )

    # Start with the transformers of transmission
    transformers_columns = [
        ('enc_trans', ohe_transmission, ['transmission'])
        ]

    # Conditionally add the fuel transformer
    # Check if 'fuelType' exists in your training dataframe columns
    FuelType_exists = 'fuelType' in X.columns
    if FuelType_exists:
        transformers_columns.append(('enc_fuel', ohe_fuel, ['fuelType']))

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers_columns,  # <--- Pass the list here
        remainder='passthrough',         # Keeps rest variables
        verbose_feature_names_out=False  # To avoid long feature names
        )

    # Ensure output is a DataFrame
    preprocessor.set_output(transform="pandas")

    # create lists to store the results from the different models
    score_train = []
    score_val = []
    # do the interaction for every k fold
    for train_index, test_index in tqdm(CV.split(X, y)):
        # split
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # impute categorial variables
            # Find mode to fill
        mode_to_fill_transmission = X_train['transmission'].mode()[0]
        if FuelType_exists:
            mode_to_fill_fueltype = X_train['fuelType'].mode()[0]
            # Fill on train
        X_train['transmission'].fillna(mode_to_fill_transmission, inplace=True)
        if FuelType_exists:
            X_train['fuelType'].fillna(mode_to_fill_fueltype, inplace=True)
            # Fill on test
        X_test['transmission'].fillna(mode_to_fill_transmission, inplace=True)
        if FuelType_exists:
            X_test['fuelType'].fillna(mode_to_fill_fueltype, inplace=True)
        # encode
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        # scale
        scaler_ = scalar
        X_train_scl = scaler_.fit_transform(X_train)
        X_test_scl = scaler_.transform(X_test)
        # impute numerical variables
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


def graph_actual_vs_predicted(model: BaseEstimator,
                              X_train: pd.DataFrame,
                              y_train: pd.Series,
                              X_val: pd.DataFrame,
                              y_val: pd.Series,
                              log_transform: bool = True) -> None:

    """
    Plot Actual vs Predicted Values
    --------------------------------
    Fits the model on training data, generates predictions for both training
    and validation sets, and creates scatter plots of actual vs predicted
    values. If log-transformed targets are used, both predictions and actuals
    are exponentiated to the original scale. The plots include a perfect
    prediction line (y=x) for reference, helping assess model fit and potential
    overfitting.

    Parameters
    ----------
    model : BaseEstimator
        Model implementing fit() and predict().
    X_train : pd.DataFrame
        Preprocessed training features.
    y_train : pd.Series
        Training targets (log-scaled if log_transform=True).
    X_val : pd.DataFrame
        Preprocessed validation features.
    y_val : pandas.Series
        Validation targets (log-scaled if log_transform=True).
    log_transform : bool, default=True
        If True, exponentiate predictions and targets for plotting.

    Returns
    -------
    None
        Displays matplotlib plots.
    """

    # 1. Train the model
    model.fit(X_train, y_train)

    # 2. Make predictions & Inverse Transform (Log -> Real Money)
    # Training Data
    if log_transform:
        y_train_pred = np.exp(model.predict(X_train))
        y_train_actual = np.exp(y_train)

        # Validation Data
        y_val_pred = np.exp(model.predict(X_val))
        y_val_actual = np.exp(y_val)

    else:
        y_train_pred = model.predict(X_train)
        y_train_actual = y_train

        # Validation Data
        y_val_pred = model.predict(X_val)
        y_val_actual = y_val

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


def model_performance(model: BaseEstimator,
                      x_train: pd.DataFrame,
                      y_train: pd.Series,
                      x_val: pd.DataFrame,
                      y_val: pd.Series,
                      log_transform: bool = True) -> None:

    """
    Print Model MAE Performance Summary
    -----------------------------------
    Fits the model on training data and prints the Mean Absolute Error (MAE)
    for both training and validation sets. If log-transformed targets are used,
    predictions and targets are exponentiated before calculating MAE.

    Parameters
    ----------
    model : BaseEstimator
        ML model to evaluate.
    x_train : pd.DataFrame
        Preprocessed training features.
    y_train : pd.Series
        Training targets (log-scaled if log_transform=True).
    x_val : pd.DataFrame
        Preprocessed validation features.
    y_val : pd.Series
        Validation targets (log-scaled if log_transform=True).
    log_transform : bool, default=True
        If True, exponentiate predictions and targets before MAE.

    Returns
    -------
    None
        Prints MAE for training and validation sets.
    """

    model.fit(x_train, y_train)
    
    if log_transform:
        print('Train MAE:', mean_absolute_error(np.exp(y_train),
                                                np.exp(model.predict(x_train))))
        print('Validation MAE:', mean_absolute_error(np.exp(y_val),
                                                     np.exp(model.predict(x_val))))
    else:
        print('Train MAE:', mean_absolute_error(y_train,
                                                model.predict(x_train)))
        print('Validation MAE:', mean_absolute_error(y_val,
                                                     model.predict(x_val)))

def grid_score(x_train: pd.DataFrame,
               y_train: pd.Series,
               x_val: pd.DataFrame,
               y_val: pd.Series,
               model: BaseEstimator,
               score_train_dic: dict,
               score_val_dic: dict,
               dic_key: str | tuple | int,
               log_transform: bool = False) -> None:
    """
    Single-Fit MAE Evaluation (Grid Search Helper)
    -----------------------------------------------
    Fits the model once on training data, generates predictions for both
    training and validation sets, and computes MAE. Handles log-transformed
    targets by exponentiating predictions and targets.

    Parameters
    ----------
    x_train : pd.DataFrame
        Preprocessed training features.
    y_train : pd.Series
        Training targets (log-scaled if log_transform=True).
    x_val : pd.DataFrame
        Preprocessed validation features.
    y_val : pd.Series
        Validation targets (log-scaled if log_transform=True).
    model : BaseEstimator
        Model to evaluate (e.g., with specific hyperparameters).
    score_train_dic : dict
        Dictionary to store training MAE (single float value).
    score_val_dic : dict
        Dictionary to store validation MAE (single float value).
    dic_key : str | tuple | int
        Key under which MAE values are saved in the dictionaries.
    log_transform : bool, default=False
        If True, exponentiate predictions and targets before MAE.

    Returns
    -------
    None
    """

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


def print_cv_results(key: str,
                     dic_train_MAE: dict[str, list[float]],
                     dic_val_MAE: dict[str, list[float]],) -> None:
    """
    Print Cross-Validation Results Summary
    --------------------------------------
    Displays the stored cross-validated MAE mean and standard deviation for
    a specific model key. Useful for quick inspection of CV performance.

    Parameters
    ----------
    key : str
        Dictionary key referencing the stored metrics (e.g., model name).
    dic_train_MAE : dict[str, list[float]]
        Contains [mean_MAE, std_MAE] for training sets.
    dic_val_MAE : dict[str, list[float]]
        Contains [mean_MAE, std_MAE] for validation sets.

    Returns
    -------
    None
        Prints the MAE results to console.
    """

    print(f'CV Results - {key}')
    print(f'Train MAE: {dic_train_MAE[key][0]}, '
          f'Train std: {dic_train_MAE[key][1]}')
    print(f'Validation MAE: {dic_val_MAE[key][0]}, '
          f'Validation std: {dic_val_MAE[key][1]}')


def predict_test_set(test_df: pd.DataFrame,
                     dict_brand_mapping: dict[str, list[str]],
                     dict_transmission_mapping: dict,
                     actual_year: int,
                     mode_to_fill_transmission: str,
                     train_columns: list[str],
                     encoder: OneHotEncoder,
                     scaler: TransformerMixin,
                     imputer: TransformerMixin,
                     model: BaseEstimator,
                     filename: str,
                     log_transform: bool = False,) -> pd.DataFrame:
    """
    Generate Predictions on Test Set
    ---------------------------------
    Processes raw test data through the full preprocessing pipeline (cleaning,
    encoding, scaling, imputation), applies the trained model to generate
    predictions, and saves them to a CSV file. Handles categorical mappings,
    missing value imputation, and optional log-transform reversal.

    Parameters
    ----------
    test_df : pd.DataFrame
        Raw test dataset with columns like 'carID', 'Brand', 'transmission',
        etc.
    dict_brand_mapping : dict[str, list[str]]
        Maps brand categories (to solve inconsistencies).
    dict_transmission_mapping : Dict[str, List[str]]
        Maps transmission categories (to solve inconsistencies).
    actual_year : int
        Current year for calculating 'Years_old' feature.
    mode_to_fill_transmission : str
        Mode value to fill missing 'transmission' (e.g., 'manual').
    train_columns : list[str]
        Column names from training data to ensure feature alignment.
    encoder : OneHotEncoder
        Fitted encoder for categorical variables.
    scaler : TransformerMixin
        Fitted scaler for numerical features.
    imputer : TransformerMixin
        Fitted imputer for missing numerical values.
    model : BaseEstimator
        Trained model for generating predictions.
    filename : str
        Base name for output CSV (e.g., 'predictions' -> 'predictions.csv').
    log_transform : bool, default=False
        If True, exponentiate predictions to original scale.

    Returns
    -------
    pd.DataFrame
        DataFrame with predictions indexed by carID.
    """
    # Copy test data
    df_test = test_df.copy()
    df_test.set_index('carID', inplace=True)

    # Drop irrelevant columns
    df_test.drop(['model',
                  'tax',
                  'previousOwners',
                  'fuelType',
                  'paintQuality%',
                  'hasDamage'], inplace=True, axis=1)

    # Correct numeric columns (negative and impossible values)
    df_test['year'] = df_test['year'].round().astype('Int64')
    df_test['mileage'] = pd.to_numeric(df_test['mileage'],
                                       errors='coerce').abs()
    df_test['mpg'] = pd.to_numeric(df_test['mpg'], errors='coerce').abs()
    df_test['engineSize'] = pd.to_numeric(df_test['engineSize'],
                                          errors='coerce').abs()
    df_test.loc[df_test['engineSize'] < 0.9, 'engineSize'] = 0.9

    # Clean strings
    df_test = df_test.applymap(lambda x: x.replace(" ", "").lower()
                               if isinstance(x, str) else x)

    # Map categorical values
    for key, values in dict_brand_mapping.items():
        df_test.loc[df_test['Brand'].isin(values), 'Brand'] = key

    for key, values in dict_transmission_mapping.items():
        df_test.loc[df_test['transmission'].isin(values),
                    'transmission'] = key if key != 'NAN' else np.nan

    # Fill missing categorical values
    df_test['transmission'].fillna(mode_to_fill_transmission, inplace=True)

    # years old
    df_test['Years_old'] = actual_year - df_test['year']
    df_test.drop('year', inplace=True, axis=1)

    # Encode categorical variables as dummies
    df_test = encoder.transform(df_test)

    # Ensure column order matches training
    df_test = df_test[train_columns]

    # Scale
    df_test_scaled = scaler.transform(df_test)

    # Impute missing values
    df_test_imputed = imputer.transform(df_test_scaled)
    df_test_to_predict = pd.DataFrame(df_test_imputed,
                                      columns=df_test.columns,
                                      index=df_test.index)

    # Generate predictions
    predictions = model.predict(df_test_to_predict)

    if log_transform:
        predictions = np.exp(predictions)

    # Create predictions DataFrame
    df_predictions = pd.DataFrame({'price': predictions},
                                  index=df_test_to_predict.index)

    # Save predictions
    df_predictions.to_csv(f"predictions/{filename}.csv")

    return df_predictions


def predict_test_set_brand(
    test_df,
    dict_brand_mapping,
    dict_transmission_mapping,
    actual_year,
    train_columns,
    brands_information,
    filename,
    log_transform=False,
):
    # Copy test data
    df_test = test_df.copy()
    df_test.set_index('carID', inplace=True)

    # Drop irrelevant columns
    df_test.drop(['model',
                  'tax',
                  'previousOwners',
                  'fuelType',
                  'paintQuality%',
                  'hasDamage'], inplace=True, axis=1)

    # Correct numeric columns (negative and impossible values)
    df_test['year'] = df_test['year'].round().astype('Int64')
    df_test['mileage'] = pd.to_numeric(df_test['mileage'], errors='coerce').abs()
    df_test['mpg'] = pd.to_numeric(df_test['mpg'], errors='coerce').abs()
    df_test['engineSize'] = pd.to_numeric(df_test['engineSize'], errors='coerce').abs()
    df_test.loc[df_test['engineSize'] < 0.9, 'engineSize'] = 0.9

    # Clean strings
    df_test = df_test.applymap(lambda x: x.replace(" ", "").lower() if isinstance(x, str) else x)

    # Map categorical values
    for key, values in dict_brand_mapping.items():
        df_test.loc[df_test['Brand'].isin(values), 'Brand'] = key

    for key, values in dict_transmission_mapping.items():
        df_test.loc[df_test['transmission'].isin(values), 'transmission'] = key if key != 'NAN' else np.nan

    # years old
    df_test['Years_old'] = actual_year - df_test['year']
    df_test.drop('year', inplace=True, axis=1)

    # Prepare saved brands list (exclude unknown)
    saved_brands = list(brands_information.keys())
    if 'unknown' in saved_brands:
        saved_brands.remove('unknown')

    # DataFrame to accumulate predictions
    df_predictions = pd.DataFrame(columns=['price'])

    # do everithing for each brand
    for brand in saved_brands:
        # filter by brand
        test_brand = df_test.loc[df_test['Brand'] == brand].copy()

        # if there is not brand of that type go to the next
        if test_brand.shape[0] == 0:
            continue

        # Fill missing categorical values
        test_brand['transmission'].fillna(brands_information[brand]['mode_transmission'], inplace=True)

        # Encode categorical variables as dummies
        test_brand = brands_information[brand]['encoder'].transform(test_brand)

        # Ensure column order matches training
        test_brand = test_brand[train_columns]

        # Scale
        df_test_scaled = brands_information[brand]['scalar'].transform(test_brand)

        # Impute missing values
        df_test_imputed = brands_information[brand]['imputer'].transform(df_test_scaled)
        df_test_to_predict = pd.DataFrame(df_test_imputed, columns=test_brand.columns, index=test_brand.index)

        # Generate predictions
        predictions = brands_information[brand]['model'].predict(df_test_to_predict)

        if log_transform:
            predictions = np.exp(predictions)

        df_pred_brand = pd.DataFrame({'price': predictions}, index=df_test_to_predict.index)
        df_predictions = pd.concat([df_predictions, df_pred_brand], axis=0)

        # If we've predicted all rows, save and return the dataframe
        if df_predictions.shape[0] == df_test.shape[0]:
            df_predictions.index.name = 'carID'
            df_predictions.to_csv(f"predictions/{filename}.csv")
            return df_predictions

    # Remaining rows -> use 'unknown' (general) model
    test_unknown = df_test.loc[~(df_test['Brand'].isin(saved_brands))].copy()

    # Fill missing categorical values
    test_unknown['transmission'].fillna(brands_information['unknown']['mode_transmission'], inplace=True)

    # Encode categorical variables as dummies
    test_unknown = brands_information['unknown']['encoder'].transform(test_unknown)

    # Ensure column order matches training
    test_unknown = test_unknown[train_columns]

    # Scale
    df_test_scaled = brands_information['unknown']['scalar'].transform(test_unknown)

    # Impute missing values
    df_test_imputed = brands_information['unknown']['imputer'].transform(df_test_scaled)
    df_test_to_predict = pd.DataFrame(df_test_imputed, columns=test_unknown.columns, index=test_unknown.index)

    # Generate predictions
    predictions = brands_information['unknown']['model'].predict(df_test_to_predict)

    if log_transform:
        predictions = np.exp(predictions)

    df_pred_brand = pd.DataFrame({'price': predictions}, index=df_test_to_predict.index)
    df_predictions = pd.concat([df_predictions, df_pred_brand], axis=0)

    # Finalize: save and return the dataframe
    df_predictions.index.name = 'carID'
    df_predictions.to_csv(f"predictions/{filename}.csv")
    return df_predictions
