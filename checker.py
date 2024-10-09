import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model import RBERT
from utils import get_label, init_logger, load_tokenizer
import torch.backends.mps as mps

logger = logging.getLogger(__name__)

def get_device(pred_config):
    """
    CODE CHANGE BEGIN: ------------------------------------------------------------------------------------------------------------------------------
    """
    # Function to determine the device (CPU or GPU)
    if mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
    """
    CODE CHANGE END: ------------------------------------------------------------------------------------------------------------------------------
    """
    #return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"

def get_args(pred_config):
    # Function to get the training arguments saved during model training
    return torch.load(os.path.join(pred_config.model_dir, "training_args.bin"))

def load_model(pred_config, args, device):
    # Function to load the pre-trained model
    # Check whether the model directory exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exist! Train first!")

    try:
        # Load the model from the specified directory
        model = RBERT.from_pretrained(pred_config.model_dir, args=args)
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model

def convert_input_sentence_to_tensor_dataset(pred_config, args, sentence,
                                             cls_token_segment_id=0, pad_token_segment_id=0,
                                             sequence_a_segment_id=0, mask_padding_with_zero=True):
    # Function to convert input sentence to a TensorDataset

    # Load the tokenizer
    tokenizer = load_tokenizer(args)

    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    # Tokenize the input sentence
    tokens = tokenizer.tokenize(sentence)

    # Get the positions of entity markers in the token list
    e11_p = tokens.index("<e1>")  # the start position of entity1
    e12_p = tokens.index("</e1>")  # the end position of entity1
    e21_p = tokens.index("<e2>")  # the start position of entity2
    e22_p = tokens.index("</e2>")  # the end position of entity2

    # Replace the entity markers with placeholders
    tokens[e11_p] = "$"
    tokens[e12_p] = "$"
    tokens[e21_p] = "#"
    tokens[e22_p] = "#"

    # Add 1 because of the [CLS] token
    e11_p += 1
    e12_p += 1
    e21_p += 1
    e22_p += 1

    # Account for [CLS] and [SEP] tokens
    if args.add_sep_token:
        special_tokens_count = 2
    else:
        special_tokens_count = 1

    # Truncate tokens if necessary
    if len(tokens) > args.max_seq_len - special_tokens_count:
        tokens = tokens[: (args.max_seq_len - special_tokens_count)]

    # Add [SEP] token
    if args.add_sep_token:
        tokens += [sep_token]

    # Add [CLS] token
    tokens = [cls_token] + tokens

    # Token type IDs for BERT (all 0s for single sentence classification)
    token_type_ids = [sequence_a_segment_id] * len(tokens)

    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Attention mask: 1 for real tokens and 0 for padding tokens
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length
    padding_length = args.max_seq_len - len(input_ids)
    input_ids = input_ids + ([pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    # Entity masks
    e1_mask = [0] * len(attention_mask)
    e2_mask = [0] * len(attention_mask)

    # Set entity masks to 1 for positions of entities
    for i in range(e11_p, e12_p + 1):
        e1_mask[i] = 1
    for i in range(e21_p, e22_p + 1):
        e2_mask[i] = 1

    # Convert everything to Tensor
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
    e1_mask = torch.tensor(e1_mask, dtype=torch.long).unsqueeze(0)
    e2_mask = torch.tensor(e2_mask, dtype=torch.long).unsqueeze(0)

    return input_ids, attention_mask, token_type_ids, e1_mask, e2_mask

def predict(pred_config, sentence):
    # Function to make predictions for a given sentence

    # Load model and arguments
    args = get_args(pred_config)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    logger.info(args)

    # Convert input sentence to TensorDataset
    input_ids, attention_mask, token_type_ids, e1_mask, e2_mask = convert_input_sentence_to_tensor_dataset(
        pred_config, args, sentence
    )

    # Predict
    inputs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "token_type_ids": token_type_ids.to(device),
        "labels": None,
        "e1_mask": e1_mask.to(device),
        "e2_mask": e2_mask.to(device),
    }

    outputs = model(**inputs)
    logits = outputs[0]

    preds = np.argmax(logits.detach().cpu().numpy(), axis=1)[0]

    # Get label list
    label_lst = get_label(args)

    # Print input sentence and predicted relation
    print("\n\nInput Sentence:", sentence)
    print("\nPredicted Relation:", label_lst[preds])

if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()

    # Get input sentence from command line argument
    sentence = input("Enter the input sentence: ")

    predict(pred_config, sentence)
