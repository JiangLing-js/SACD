import argparse
import torch
from contrastive_classifier import FeatureClassifierModel, compute_prototype
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import os

def load_and_pool_features(file_path):
    features = torch.load(file_path)
    return features

def prepare_data(features, label):
    X = []
    y = []
    for embedding in features.values():
        X.append(embedding[0])
        y.append(label)
    return X, y

def prepare_test_data(file_paths, labels):
    X, y, file_names = [], [], []
    for file_path, label in zip(file_paths, labels):
        features = load_and_pool_features(file_path)
        X_batch, y_batch = prepare_data(features, label)
        X.append(torch.stack(X_batch))
        y.append(torch.tensor(y_batch, dtype=torch.long))
        file_names.append(file_path)  # Store the file name
    return X, y, file_names

def evaluate_model(clf, X_test):
    z, outputs = clf(X_test)
    probabilities = nn.functional.softmax(outputs, dim=1)
    _, y_pred = torch.max(outputs, 1)
    return probabilities, y_pred

def main(model_path, data_paths, output_file):
    model = FeatureClassifierModel(num_channels=256, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_files = data_paths.split(',')
    test_labels = [1] * len(test_files)  # Assuming all labels are 1 for simplicity, update as needed
    X_tests, y_tests, file_names = prepare_test_data(test_files, test_labels)

    with open(output_file, 'w') as file:
        file.write("Image Path,Prediction,Label\n")
        for i, file_name in enumerate(file_names):
            X_test = X_tests[i]
            y_test = y_tests[i]
            probabilities, y_pred = evaluate_model(model, X_test)

            for prob, pred, true_label in zip(probabilities, y_pred, y_test):
                prob_value = prob[pred].item()
                file.write(f"{file_name},{prob_value:.6f},{true_label.item()}\n")
                print(f"Processed {file_name}: Prob={prob_value:.6f}, Pred={pred.item()}, Label={true_label.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on test datasets and save results.')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--data', type=str, required=True, help='Comma-separated paths to the test data files')
    parser.add_argument('--output', type=str, required=True, help='Output file to save results')

    args = parser.parse_args()
    main(args.model, args.data, args.output)
