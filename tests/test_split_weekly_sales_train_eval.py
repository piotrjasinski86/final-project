import pandas as pd
import os

# Adjust the data path to your structure
DATA_DIR = os.path.join(os.path.dirname(__file__), '..')
TRAIN_PATH = os.path.abspath(os.path.join(DATA_DIR, 'sales_data_train.csv'))
EVAL_PATH = os.path.abspath(os.path.join(DATA_DIR, 'sales_data_eval.csv'))

def test_files_exist():
    assert os.path.exists(TRAIN_PATH), f"Output file {TRAIN_PATH} does not exist!"
    print("✔ Training file existence check passed.")
    assert os.path.exists(EVAL_PATH), f"Output file {EVAL_PATH} does not exist!"
    print("✔ Evaluation file existence check passed.")

def test_columns():
    for path, label in zip([TRAIN_PATH, EVAL_PATH], ["train", "eval"]):
        df = pd.read_csv(path, nrows=10)
        expected_cols = {'Forecasting Group', 'Category', 'Country', 'Sales_week', 'Sales_volume'}
        assert expected_cols.issubset(set(df.columns)), f"{label} set missing expected columns: {expected_cols - set(df.columns)}"
        print(f"✔ {label.capitalize()} column presence check passed.")

def test_no_missing_or_negative_sales():
    for path, label in zip([TRAIN_PATH, EVAL_PATH], ["train", "eval"]):
        df = pd.read_csv(path, usecols=['Sales_volume'])
        assert df['Sales_volume'].isnull().sum() == 0, f"Missing values in Sales_volume in {label} set!"
        assert (df['Sales_volume'] >= 0).all(), f"Negative values in Sales_volume in {label} set!"
        print(f"✔ {label.capitalize()} sales values check passed.")

def test_year_split():
    train = pd.read_csv(TRAIN_PATH, usecols=['Sales_week'])
    eval = pd.read_csv(EVAL_PATH, usecols=['Sales_week'])
    train_years = train['Sales_week'].str[:4].astype(int)
    eval_years = eval['Sales_week'].str[:4].astype(int)
    assert (train_years <= 2017).all(), "Train set has data from after 2017!"
    assert (eval_years >= 2018).all(), "Eval set has data from before 2018!"
    print("✔ Year-based split check passed.")

def test_no_overlap():
    train = pd.read_csv(TRAIN_PATH, usecols=['Sales_week'])
    eval = pd.read_csv(EVAL_PATH, usecols=['Sales_week'])
    overlap = set(train['Sales_week']) & set(eval['Sales_week'])
    assert not overlap, f"Overlap in Sales_week between train and eval sets: {overlap}"
    print("✔ No overlap in Sales_week between train and eval sets check passed.")

def test_sales_week_format():
    for path, label in zip([TRAIN_PATH, EVAL_PATH], ["train", "eval"]):
        df = pd.read_csv(path, usecols=['Sales_week'])
        assert df['Sales_week'].str.match(r'^\d{4}-\d{2}$').all(), f"{label} set has Sales_week values not in YYYY-WW format!"
        print(f"✔ {label.capitalize()} Sales_week format check passed.")

if __name__ == "__main__":
    test_files_exist()
    test_columns()
    test_no_missing_or_negative_sales()
    test_year_split()
    test_no_overlap()
    test_sales_week_format()
    print("\nAll tests passed successfully!")
