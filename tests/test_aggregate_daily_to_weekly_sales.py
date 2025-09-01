import pandas as pd
import os

# Path to the data file relative to the test script
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'sales_data_weeks.csv')
DATA_PATH = os.path.abspath(DATA_PATH)

def test_file_exists():
    assert os.path.exists(DATA_PATH), f"Output file {DATA_PATH} does not exist!"
    print("✔ File existence check passed.")

def test_columns():
    df = pd.read_csv(DATA_PATH, nrows=10)
    expected_cols = {'Forecasting Group', 'Category', 'Country', 'Sales_week', 'Sales_volume'}
    assert expected_cols.issubset(set(df.columns)), f"Missing expected columns: {expected_cols - set(df.columns)}"
    print("✔ Column presence check passed.")

def test_no_missing_sales():
    df = pd.read_csv(DATA_PATH, usecols=['Sales_volume'])
    assert df['Sales_volume'].isnull().sum() == 0, "Missing values in Sales_volume!"
    print("✔ No missing sales values check passed.")

def test_no_negative_sales():
    df = pd.read_csv(DATA_PATH, usecols=['Sales_volume'])
    assert (df['Sales_volume'] >= 0).all(), "Negative values found in Sales_volume!"
    print("✔ No negative sales values check passed.")

def test_country_mapping():
    df = pd.read_csv(DATA_PATH, usecols=['Country'])
    valid_countries = {'Poland', 'Switzerland', 'Denmark'}
    assert df['Country'].isin(valid_countries).all(), "Unexpected values in Country column!"
    print("✔ Country mapping check passed.")

def test_sales_week_format():
    df = pd.read_csv(DATA_PATH, usecols=['Sales_week'])
    assert df['Sales_week'].str.match(r'^\d{4}-\d{2}$').all(), "Some Sales_week values are not in YYYY-WW format!"
    print("✔ Sales_week format check passed.")

if __name__ == "__main__":
    test_file_exists()
    test_columns()
    test_no_missing_sales()
    test_no_negative_sales()
    test_country_mapping()
    test_sales_week_format()
    print("\nAll tests passed successfully!")
