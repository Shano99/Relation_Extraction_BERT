import argparse
from data_loader import load_and_cache_examples
from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    # Load and cache the training and test datasets
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    # Initialize the trainer with datasets and training arguments
    trainer = Trainer(args, train_dataset=train_dataset, test_dataset=test_dataset)

    # If training is enabled, start the training process
    if args.do_train:
        trainer.train()

    # If evaluation is enabled, load the model and evaluate it on the test dataset
    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == "__main__":

    """
    CODE CHANGE BEGIN: ------------------------------------------------------------------------------------------------------------------------------
    """
    
    """
    # CODE-CHANGE : We introduced this segment to conduct a grid search over Dropout Rate, Weight Decay, and Learning Rate to identify the optimal combination based on calculated accuracies.
    # This code section is commented out since we have already identified the optimal combination for maximum accuracy, and those combinations are utilized in the subsequent code snippet.
    
    dropout_rate = [0.05, 0.1, 0.2]
    weight_decay = [0.01, 0.05, 0.1]
    learning_rate = [5e-5, 2e-5, 1e-5]
    c=0

    for dropout in dropout_rate:
        for decay in weight_decay:
            for lr in learning_rate:
                c+=1
                print(f"c: {c},Dropout Rate: {dropout}, Weight Decay: {decay}, Learning Rate: {lr}")

    """

    
    # the hyperparameters we have changed to optimize the model and increase the accuracy
    # Predefined values for various training hyperparameters
    val_dropout_rate, val_weight_decay, val_learning_rate = (0.1, 0.05, 5e-05)
    val_train_epochs = 5.0
    val_train_batch_size = 16
    val_seed = 90
    val_adam_epsilon = 1e-8
    val_max_grad_norm = 1.0
    
    """
    CODE CHANGE END: ------------------------------------------------------------------------------------------------------------------------------
    """
    
    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Define arguments for command line interface
    parser.add_argument("--task", default="semeval", type=str, help="The name of the task to train")
    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    parser.add_argument(
        "--eval_dir",
        default="./eval",
        type=str,
        help="Evaluation script, result directory",
    )
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Model Name or Path",
    )
    parser.add_argument("--seed", type=int, default=val_seed, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=val_train_batch_size, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
    parser.add_argument(
        "--max_seq_len",
        default=384,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--learning_rate",
        default=val_learning_rate,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=val_train_epochs,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--weight_decay", default=val_weight_decay, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--adam_epsilon", default=val_adam_epsilon, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=val_max_grad_norm, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--dropout_rate",
        default=val_dropout_rate,
        type=float,
        help="Dropout for fully-connected layers",
    )

    parser.add_argument("--logging_steps", type=int, default=250, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=250,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--add_sep_token",
        action="store_true",
        help="Add [SEP] token at the end of the sentence",
    )

    # Parse arguments from command line
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args)
