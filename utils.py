"""
Type hints and helper functions for retail sales regression analysis.
"""

from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load clothing store dataset from CSV.
    
    Args:
        filepath: Path to Clothing.csv
        
    Returns:
        DataFrame with 400 stores and 13 variables
    """
    return pd.read_csv(filepath)


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test sets.
    
    Args:
        df: Input DataFrame
        test_size: Fraction for test set (default 0.2 = 80/20 split)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test


def calculate_vif(df: pd.DataFrame, target: str = "tsales") -> Dict[str, float]:
    """
    Calculate Variance Inflation Factor for multicollinearity check.
    
    Args:
        df: DataFrame with predictors
        target: Name of target variable to exclude
        
    Returns:
        Dictionary mapping predictor names to VIF scores
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    X = df.drop(columns=[target])
    vif_data = {}
    
    for i, col in enumerate(X.columns):
        vif = variance_inflation_factor(X.values, i)
        vif_data[col] = vif
        
    return vif_data


if __name__ == "__main__":
    print("Import this module in your notebook:")
    print("from utils import load_data, prepare_data, calculate_vif")
