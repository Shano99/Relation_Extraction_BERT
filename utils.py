import logging
import os
import random
import numpy as np
import torch
from transformers import BertTokenizer
from official_eval import official_f1

# Additional special tokens for entity markers
ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

# Function to get the relation labels from the label file
def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8")]

# Function to load the tokenizer and add additional special tokens
def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer

# Function to write predictions to a file
def write_prediction(args, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))

# Function to initialize the logger
def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

# Function to set random seeds for reproducibility
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

# Function to compute metrics (accuracy and F1 score)
def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)

# Function to compute simple accuracy
def simple_accuracy(preds, labels):
    return (preds == labels).mean()

# Function to compute accuracy and F1 score
def acc_and_f1(preds, labels, average="macro"):
    acc = simple_accuracy(preds, labels)
    return {
        "acc": acc,
        "Macro-f1": official_f1(),  # Using an official F1 calculation function
    }
