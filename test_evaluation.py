import logging
import os
import random
import numpy as np
import torch
from transformers import BertTokenizer
from official_eval import official_f1
import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from model import RBERT
from utils import init_logger, load_tokenizer
from utils import get_label

# Function to calculate precision, recall, accuracy, and relation accuracies
def calculate_metrics(predictions_path, test_labels_path):
    model_dir = "./model" 
    args = torch.load(model_dir + "/training_args.bin")
    
    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = [line.strip() for line in f.readlines()]

    # Load test labels
    with open(test_labels_path, 'r') as f:
        test_labels = [line.split('\t')[0].strip() for line in f.readlines()]

    assert len(predictions) == len(test_labels), "The number of predictions and test labels must match"

    # Initialize counters
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    # Dictionary to store counts for each relation type
    relation_counts = {relation: {'tp': 0, 'fp': 0, 'fn': 0} for relation in get_label(args)}

    # Loop through predictions and test labels to calculate counts
    for pred, true_label in zip(predictions, test_labels):
        if pred == true_label:
            tp += 1
            relation_counts[true_label]['tp'] += 1
        else:
            if pred != 'Other':  # Assuming 'Other' is considered as a negative prediction
                fp += 1
                relation_counts[pred]['fp'] += 1
            if true_label != 'Other':  # Assuming 'Other' is a negative label
                fn += 1
                relation_counts[true_label]['fn'] += 1

    # Calculate overall metrics
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    accuracy = tp / len(test_labels)

    # Calculate metrics for each relation
    relation_accuracies = {}
    for relation, counts in relation_counts.items():
        relation_tp = counts['tp']
        relation_fp = counts['fp']
        relation_fn = counts['fn']
        relation_accuracy = relation_tp / (relation_tp + relation_fp + relation_fn) if relation_tp + relation_fp + relation_fn > 0 else 0
        relation_accuracies[relation] = relation_accuracy

    # Calculate micro-averaging F1 score
    micro_precision = precision
    micro_recall = recall
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    return precision, recall, accuracy, relation_accuracies, micro_f1

# Specify the paths to your prediction and test label files
predictions_path = os.path.join("./Test Output", "pred.tsv")
test_labels_path = os.path.join("./data", "test.tsv")

# Calculate the metrics
precision, recall, accuracy, relation_accuracies, micro_f1 = calculate_metrics(predictions_path, test_labels_path)

# Print the metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Micro-averaging F1: {micro_f1:.4f}")
print("\nAccuracy for each relation:")
for relation, acc in relation_accuracies.items():
    print(f"{relation}: {acc:.4f}")


