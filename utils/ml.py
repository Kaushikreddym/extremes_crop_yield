import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from utils.utils import get_logger

from sklearn.neighbors import BallTree
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

logger = get_logger("ml")

def preprocess_dataframe(df, indices, yield_col):
    df = df.copy()

    # 1. Replace 0s in the yield column with NaN
    df[yield_col] = df[yield_col].where(df[yield_col] >= 1, 0)
    df[yield_col] = df[yield_col].replace(0, np.nan)

    # Drop rows with NaN in any of the specified columns
    drop_cols = indices + [yield_col]
    df = df.dropna(subset=drop_cols)

    # 2. Add year column
    df['year'] = pd.to_datetime(df['time']).dt.year

    # 3. Add location column: divide into 10 spatial regions with equal number of points
    coords = df[['lat', 'lon']].drop_duplicates().copy()

    # Use quantile-based binning on lat/lon jointly
    coords['region_bin'] = pd.qcut(
        np.arange(len(coords)), 
        q=10, 
        labels=False
    )

    # Merge region info back into original dataframe
    df = df.merge(coords[['lat', 'lon', 'region_bin']], on=['lat', 'lon'], how='left')
    df.rename(columns={'region_bin': 'location'}, inplace=True)

    return df

def leave_location_and_time_out_expanding_window(
    data,
    year_col,
    space_col,
    min_train_years=5,
    test_years=1,
    test_frac=0.3,
    random_state=42
):
    """
    Leave-Location-and-Time-Out with Expanding Window (LLTO-EW)

    Trains on an expanding window of years starting from the earliest year.
    Leaves out a random 30% of locations for testing in the next `test_years`.

    Each split:
    - Trains on all years up to a certain year using 70% of locations
    - Tests on the immediately following `test_years` using the remaining 30% of locations

    Parameters:
    ----------
    data : pd.DataFrame
        Input dataframe containing temporal and spatial columns.
    year_col : str
        Column name representing the year (int or datetime).
    space_col : str
        Column name representing the spatial unit (e.g., district, grid).
    min_train_years : int, default=5
        Minimum number of years to begin training.
    test_years : int, default=1
        Number of years to use for testing.
    test_frac : float, default=0.3
        Fraction of locations to leave out for testing.
    random_state : int, default=42
        Seed for reproducibility.

    Returns:
    -------
    splits : list of (train_indices, test_indices)
        Index tuples for training and testing data in each LLTO-EW fold.
    """
    data = data.copy()
    splits = []

    all_years = sorted(data[year_col].unique())
    all_locations = np.array(data[space_col].unique())
    rng = np.random.default_rng(seed=random_state)

    print("\nLeave-Location-and-Time-Out with Expanding Window (LLTO-EW)\n" + "-" * 60)

    for end_train_idx in range(min_train_years, len(all_years) - test_years):
        train_year_start = all_years[0]
        train_year_end = all_years[end_train_idx - 1]
        test_year_start = all_years[end_train_idx]
        test_year_end = all_years[end_train_idx + test_years - 1]

        # Random 70-30 location split
        shuffled_locations = rng.permutation(all_locations)
        split_idx = int((1 - test_frac) * len(shuffled_locations))
        train_locs = shuffled_locations[:split_idx]
        test_locs = shuffled_locations[split_idx:]

        train_mask = (
            data[year_col].between(train_year_start, train_year_end) &
            data[space_col].isin(train_locs)
        )
        test_mask = (
            data[year_col].between(test_year_start, test_year_end) &
            data[space_col].isin(test_locs)
        )

        train_idx = data[train_mask].index
        test_idx = data[test_mask].index

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((list(train_idx), list(test_idx)))
            print(f"Train: {train_year_start}-{train_year_end} | Test: {test_year_start}-{test_year_end} | "
                  f"Train Locs: {len(train_locs)} | Test Locs: {len(test_locs)} "
                  f"({len(train_idx)} train, {len(test_idx)} test)")

    return splits

def robust_scale_train_test(train_df, test_df):
    """
    Applies RobustScaler to train and test DataFrames based on train data statistics.
    NaN values are ignored during fitting and preserved in output.

    Parameters:
    - train_df (pd.DataFrame): Training dataset
    - test_df (pd.DataFrame): Testing dataset

    Returns:
    - scaled_train (pd.DataFrame): Robust-scaled training data
    - scaled_test (pd.DataFrame): Robust-scaled test data
    """

    # Initialize the scaler
    scaler = RobustScaler()

    # Fit only on non-NaN values in training data
    scaler.fit(train_df.dropna())

    # Transform while preserving original index and column names
    scaled_train = pd.DataFrame(
        scaler.transform(train_df),
        index=train_df.index,
        columns=train_df.columns
    )

    scaled_test = pd.DataFrame(
        scaler.transform(test_df),
        index=test_df.index,
        columns=test_df.columns
    )

    # Preserve original NaN values
    scaled_train[train_df.isna()] = pd.NA
    scaled_test[test_df.isna()] = pd.NA

    return scaled_train, scaled_test

def make_objective_xgb(df, indices, target_col, group_index):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_loguniform("gamma", 0.001, 5),
            "subsample": trial.suggest_loguniform("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.1, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 0.0001, 10),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 0.0001, 10)
        }

        logger.info(f"\n\n\nTrial {trial.number} - Params: {params}")

        model = XGBRegressor(
            **params,
            tree_method="hist", 
            predictor="gpu_predictor", 
            device="cuda"
        )
        
        losses = []
        for fold, (train_index, test_index) in enumerate(group_index):
            X_train = df.loc[train_index, indices]
            y_train = df.loc[train_index, target_col]

            X_test = df.loc[test_index, indices]
            y_test = df.loc[test_index, target_col]

            X_train_scaled, X_test_scaled = robust_scale_train_test(X_train, X_test)

            model.fit(X_train_scaled, y_train)
            test_preds = model.predict(X_test_scaled)
            r2 = r2_score(y_test,test_preds)
            loss = np.sqrt(mean_squared_error(y_test, test_preds))
            losses.append(loss)

            logger.info(f"\tFold {fold}, Test RMSE: {loss:.4f}, Test r2: {r2:.2f}")

        average_loss = np.mean(losses)
        logger.info(f"\tTrial {trial.number} - Average Test RMSE: {average_loss:.4f}\n")

        return average_loss

    return objective

from sklearn.ensemble import RandomForestRegressor

def make_objective_rf(df, indices, target_col, group_index):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

        logger.info(f"\n\n\nTrial {trial.number} - Params: {params}")

        model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)

        losses = []
        for fold, (train_index, test_index) in enumerate(group_index):
            X_train = df.loc[train_index, indices]
            y_train = df.loc[train_index, target_col]
            X_test = df.loc[test_index, indices]
            y_test = df.loc[test_index, target_col]

            X_train_scaled, X_test_scaled = robust_scale_train_test(X_train, X_test)

            model.fit(X_train_scaled, y_train)
            test_preds = model.predict(X_test_scaled)
            r2 = r2_score(y_test, test_preds)
            loss = np.sqrt(mean_squared_error(y_test, test_preds))
            losses.append(loss)

            logger.info(f"\tFold {fold}, Test RMSE: {loss:.4f}, Test R²: {r2:.2f}")

        avg_loss = np.mean(losses)
        logger.info(f"\tTrial {trial.number} - Avg RMSE: {avg_loss:.4f}\n")
        return avg_loss

    return objective

import torch
import torch.nn as nn
import torch.optim as optim

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def make_objective_dnn(df, indices, target_col, group_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        hidden_dims = [trial.suggest_int("hidden_dim_1", 32, 256)]
        if trial.suggest_categorical("use_second_layer", [True, False]):
            hidden_dims.append(trial.suggest_int("hidden_dim_2", 32, 128))
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
        epochs = trial.suggest_int("epochs", 10, 100)

        logger.info(f"\n\n\nTrial {trial.number} - Params: {{hidden_dims: {hidden_dims}, dropout: {dropout}, lr: {lr}, epochs: {epochs}}}")

        losses = []

        for fold, (train_index, test_index) in enumerate(group_index):
            X_train = df.loc[train_index, indices]
            y_train = df.loc[train_index, target_col]
            X_test = df.loc[test_index, indices]
            y_test = df.loc[test_index, target_col]

            X_train_scaled, X_test_scaled = robust_scale_train_test(X_train, X_test)

            X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
            X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

            model = SimpleMLP(input_dim=X_train_tensor.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                preds = model(X_train_tensor)
                loss = criterion(preds, y_train_tensor)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                preds = model(X_test_tensor).squeeze().cpu().numpy()
                y_test_np = y_test_tensor.squeeze().cpu().numpy()

            rmse = np.sqrt(mean_squared_error(y_test_np, preds))
            r2 = r2_score(y_test_np, preds)
            losses.append(rmse)

            logger.info(f"\tFold {fold}, Test RMSE: {rmse:.4f}, R²: {r2:.2f}")

        avg_loss = np.mean(losses)
        logger.info(f"\tTrial {trial.number} - Avg RMSE: {avg_loss:.4f}\n")
        return avg_loss

    return objective
