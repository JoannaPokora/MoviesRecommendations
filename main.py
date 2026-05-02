import argparse
import os
import pickle

from modules.train import train
from modules.predict import predict


'''
Example usage:

(a) Training the model:

python main.py --mode train \
--train_file data/ratings.csv \
--model_path models_trained/model_NMF.pkl \
--alg NMF

(b) Prediction:
python main.py --mode predict \
--input_file sample_test.csv \
--model_path models_trained/model_NMF.pkl \
--output_file results/preds.csv \
--alg NMF

'''

def parse_arguments():
    parser = argparse.ArgumentParser(description="Movies recommendation system based on the previous user ratings.")

    parser.add_argument("--mode", type=str, required=True,
                        help="Mode of operation: 'train' or 'predict'.")

    parser.add_argument("--train_file", type=str, default="data/ratings.csv",
                        help="CSV file with training data (userId,movieId,rating).")

    parser.add_argument("--input_file", type=str, default="data/sample_test.csv",
                        help="CSV file with (userId,movieId) for prediction.")

    parser.add_argument("--model_path", type=str,
                        default="models_trained/model_NMF.pkl",
                        help="Path to save/load the trained model.")

    parser.add_argument("--output_file", type=str,
                        default="results/preds.csv",
                        help="Where to save predictions.")

    parser.add_argument("--alg", type=str,   required=True,
                        help="Algorithm to use (NMF, SVD1, SVD2, SGD or BEST.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    mode = args.mode.lower()
    alg = args.alg.upper()

    if alg not in ["NMF", "SVD1", "SVD2", "SGD", "BEST"]:
        print("--alg must be one of: NMF, SVD1, SVD2, SGD, BEST")
        return

    if mode == "train":
        print(f"Training mode activated: {alg}.")

        Z_approx, user_map, movie_map = train(args.train_file, alg)

        model_data = {
            "Z_approx": Z_approx,
            "user_map": user_map,
            "movie_map": movie_map
        }

        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

        with open(args.model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {args.model_path}")

    elif mode == "predict":
        print(f"Prediction mode activated: {alg}.")

        if not os.path.exists(args.model_path):
            print("Model file does not exist. Please run training first.")
            return

        with open(args.model_path, "rb") as f:
            model_data = pickle.load(f)

        predictions = predict(args.input_file, model_data)

        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

        with open(args.output_file, "w") as f:
            f.write("userId,movieId,rating\n")
            for row in predictions.itertuples():
                f.write(f"{row.userId},{row.movieId},{row.rating}\n")

        print(f"Predictions saved to {args.output_file}.")

    else:
        print("Invalid --mode. Use 'train' or 'predict'.")


if __name__ == "__main__":
    main()
