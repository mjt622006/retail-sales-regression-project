"""
Unit tests for retail sales regression analysis.

Run with: pytest test_analysis.py -v
"""

import unittest
import pandas as pd
import numpy as np
import os


class TestDataLoading(unittest.TestCase):
    """Test data loading and preparation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.csv_path = "Clothing.csv"
    
    def test_csv_exists(self):
        """Test that Clothing.csv exists."""
        self.assertTrue(os.path.exists(self.csv_path), f"{self.csv_path} not found")
    
    def test_data_loads(self):
        """Test that data can be loaded."""
        df = pd.read_csv(self.csv_path)
        self.assertIsNotNone(df, "DataFrame should not be None")
    
    def test_data_shape(self):
        """Test that data has expected dimensions."""
        df = pd.read_csv(self.csv_path)
        self.assertEqual(df.shape[0], 400, "Dataset should have 400 observations")
        self.assertGreater(df.shape[1], 10, "Dataset should have >10 variables")
    
    def test_no_missing_values(self):
        """Test that there are no missing values."""
        df = pd.read_csv(self.csv_path)
        self.assertEqual(df.isnull().sum().sum(), 0, "Dataset contains missing values")
    
    def test_target_variable_exists(self):
        """Test that target variable 'tsales' exists."""
        df = pd.read_csv(self.csv_path)
        self.assertIn("tsales", df.columns, "Target variable 'tsales' not found")
    
    def test_numeric_data(self):
        """Test that all columns are numeric."""
        df = pd.read_csv(self.csv_path)
        for col in df.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]), 
                          f"Column {col} should be numeric")


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity checks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.read_csv("Clothing.csv")
    
    def test_positive_sales(self):
        """Test that sales values are positive."""
        self.assertTrue((self.df["tsales"] > 0).all(), 
                       "Sales values should be positive")
    
    def test_positive_store_size(self):
        """Test that store size values are positive."""
        if "size" in self.df.columns:
            self.assertTrue((self.df["size"] > 0).all(), 
                           "Store size should be positive")
    
    def test_reasonable_ranges(self):
        """Test that values are in reasonable ranges."""
        # Sales should be between 0 and reasonable max
        self.assertTrue(self.df["tsales"].max() < 10_000_000, 
                       "Max sales seems unreasonable")


class TestFileStructure(unittest.TestCase):
    """Test repository structure."""
    
    def test_readme_exists(self):
        """Test that README.md exists."""
        self.assertTrue(os.path.exists("README.md"), "README.md not found")
    
    def test_notebook_exists(self):
        """Test that Jupyter notebook exists."""
        self.assertTrue(os.path.exists("retail_sales_regression.ipynb"), 
                       "retail_sales_regression.ipynb not found")
    
    def test_requirements_exists(self):
        """Test that requirements.txt exists."""
        self.assertTrue(os.path.exists("requirements.txt"), "requirements.txt not found")


if __name__ == "__main__":
    unittest.main()
