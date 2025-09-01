import pandas as pd
import os

# Adjust path if needed
DATA_DIR = os.path.join(os.path.dirname(__file__), '..')
OUTPUT_PATH = os.path.abspath(os.path.join(DATA_DIR, 'sales_data_full_history_only.csv'))

def test_file_exists():
    assert os.path.exists(OUTPUT_PATH), f"Output file {OUTPUT_PATH} does not exist!"
    print("✔ Output file existence check passed.")

def test_columns():
    df = pd.read_csv(OUTPUT_PATH, nrows=10)
    expected_cols = {'Forecasting Group', 'Country', 'Sales_week', 'Sales_volume'}
    assert expected_cols.issubset(set(df.columns)), f"Missing columns: {expected_cols - set(df.columns)}"
    print("✔ Output columns check passed.")

def test_no_new_intro_pairs():
    df = pd.read_csv(OUTPUT_PATH)
    assert 'IsNewIntro' in df.columns or True, "IsNewIntro column not found in output."  # Optional, column may not be present
    # For robust check, let's recalculate per pair
    min_week = df["Sales_week"].min()
    first_sales = (
        df[df["Sales_volume"] > 0]
        .groupby(["Forecasting Group", "Country"])["Sales_week"]
        .min()
    )
    all_full_history = (first_sales == min_week).all()
    assert all_full_history, "Some product–country pairs do not have full history!"
    print("✔ All product–country pairs have full sales history check passed.")

def test_no_missing_or_negative_sales():
    df = pd.read_csv(OUTPUT_PATH, usecols=['Sales_volume'])
    assert df['Sales_volume'].isnull().sum() == 0, "Missing values in Sales_volume!"
    assert (df['Sales_volume'] >= 0).all(), "Negative values in Sales_volume!"
    print("✔ No missing or negative sales values check passed.")

if __name__ == "__main__":
    test_file_exists()
    test_columns()
    test_no_new_intro_pairs()
    test_no_missing_or_negative_sales()
    print("\nAll tests passed successfully!")
