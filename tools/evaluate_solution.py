import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute RMSE between predictions and true ratings.")
    parser.add_argument("--pred_file", type=str, required=True,
                        help="Path to the CSV file with predictions (must contain a column 'rating').")
    parser.add_argument("--true_file", type=str, required=True,
                        help="Path to the CSV file with true ratings (must contain a column 'rating').")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load the CSV files
    df_pred = pd.read_csv(args.pred_file)
    df_true = pd.read_csv(args.true_file)

    # Check that both files contain the "rating" column
    if "rating" not in df_pred.columns or "rating" not in df_true.columns:
        print("Error: Both CSV files must contain a 'rating' column.")
        return

    # Compute RMSE using scikit-learn's mean_squared_error function
    mse = mean_squared_error(df_true["rating"], df_pred["rating"])
    rmse = np.sqrt(mse)

    print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()