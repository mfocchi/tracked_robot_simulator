#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for generating and managing slip regression models.
"""
import os
import numpy as np
import pandas as pd
from termcolor import colored


class RegressorUtils:
    """
    Helper functions for slip model training and evaluation.
    """

    @staticmethod
    def collect_training_data(csv_files, output_file=None):
        """
        Combine multiple identification CSV files into one training dataset.

        Args:
            csv_files: list of CSV file paths
            output_file: if provided, save merged data to this file

        Returns:
            pd.DataFrame with columns: wheel_l, wheel_r, beta_l, beta_r, alpha
        """
        dfs = []
        for f in csv_files:
            if os.path.exists(f):
                df = pd.read_csv(f)
                # standardize column names if needed
                cols = ['wheel_l', 'wheel_r', 'beta_l', 'beta_r', 'alpha']
                if all(c in df.columns for c in cols):
                    dfs.append(df[cols])
                else:
                    print(f"Skipping {f}: missing required columns")

        if not dfs:
            print("No valid files found")
            return None

        merged = pd.concat(dfs, ignore_index=True)
        if output_file:
            merged.to_csv(output_file, index=False)
            print(f"Merged data saved to {output_file}")
        return merged

    @staticmethod
    def remove_outliers(df, column, n_std=3.0):
        """
        Remove outliers from a column based on standard deviation.

        Args:
            df: DataFrame
            column: column name
            n_std: number of standard deviations threshold

        Returns:
            filtered DataFrame
        """
        mean = df[column].mean()
        std = df[column].std()
        mask = np.abs(df[column] - mean) < n_std * std
        return df[mask]

    @staticmethod
    def train_catboost_model(X, y, output_path, **kwargs):
        """
        Train a CatBoost regression model and save it.

        Args:
            X: feature matrix (n_samples, n_features)
            y: target vector
            output_path: where to save the .cb file
            kwargs: additional CatBoost parameters

        Returns:
            trained model
        """
        try:
            import catboost as cb
        except ImportError:
            print(colored("CatBoost not installed. Install with: pip install catboost", "red"))
            return None

        model = cb.CatBoostRegressor(**kwargs)
        model.fit(X, y, verbose=False)
        model.save_model(output_path)
        print(f"Model saved to {output_path}")
        return model

    @staticmethod
    def train_rbf_model(X, y, smoothing=0.1):
        """
        Train a RBF interpolator model.

        Args:
            X: feature matrix
            y: target vector
            smoothing: smoothing parameter

        Returns:
            RBFInterpolator object
        """
        from scipy.interpolate import RBFInterpolator
        return RBFInterpolator(X, y, smoothing=smoothing)