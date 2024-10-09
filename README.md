

```markdown
# MODEL  - Deep learning-based approach based on transformer models (BERT)


### Method Description

The provided code implements a deep learning-based approach for relation extraction using transformer models, specifically BERT. It trains a neural network model on a labeled dataset, evaluating its performance using F1-score, accuracy, and loss metrics. The trained model is then used to predict relations on test data, comparing the predictions with ground truth labels to compute accuracy, precision, recall, and Micro-F1 score. Additionally, the code includes functionality to extract relations from single input sentences, providing a convenient interface for users to interact with the trained model.### Method Description

The provided code implements a deep learning-based approach for relation extraction using transformer models, specifically BERT. It trains a neural network model on a labeled dataset, evaluating its performance using F1-score, accuracy, and loss metrics. The trained model is then used to predict relations on test data, comparing the predictions with ground truth labels to compute accuracy, precision, recall, and Micro-F1 score. Additionally, the code includes functionality to extract relations from single input sentences, providing a convenient interface for users to interact with the trained model.

-------------------------------------------------------------------------------------------------------------------------------------------
## How to Run for prediction on single input sentence

**PREDICTION ON SINGLE INPUT**
 
1) Installing relevant libraries:
	pip install -r requirements.txt

2) Relation Extraction on one given input:
	python checker.py
	
	After running the command, a prompt asks user to input the sentence annd displays the relation as output.

	Examples of input sentence-
	1.  <e1>Honey</e1> is produced by <e2>bees</e2> as they collect nectar from flowers.
	2.  The heavy rain <e1>storm</e1> caused severe <e2>flooding</e2> in the coastal area.

-------------------------------------------------------------------------------------------------------------------------------------------

## How to Run Code for training till testing

NOTE - These commands are intended for executing tasks when the user wants to train the model anew and make predictions based on the test dataset.

### 1. Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

### 2. Training and Evaluation

For training and evaluation:

```bash
python main.py --do_train --do_eval
```

Prediction will be written to `proposed_answers.txt` in the `eval` directory.

### 3. Prediction on Test Data

For prediction on test data:

```bash
python predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```

Example:

```bash
python predict.py --input_file "./data/test.tsv" --output_file "./Test Output/pred.tsv" --model_dir "./model"
```

### 4. Evaluation on Test Data

For evaluation on test data and predicted data:

```bash
python test_evaluation.py
```

This provides accuracy, precision, and recall on predicted data.

### 5. Relation Extraction on Single Input

For extracting relations from a single input:

```bash
python checker.py
```

After running the command, a prompt asks the user to input the sentence and displays the relation as output.

Example of input sentence:

```
<e1>Honey</e1> is produced by <e2>bees</e2> as they collect nectar from flowers.
```

### 6. Macro-F1 Score Calculation

To calculate the macro-F1 score:

```bash
python official_eval.py
```

## File Hierarchy

```
- Best_Archive.zip
- checker.py
- data_loader.py
- main.py
- model.py
- official_eval.py
- predict.py
- README.md
- requirements.txt
- sample_pred_in.txt
- test_evaluation.py
- trainer.py 
- utils.py
- data/
  - cached_test_semeval_bert-base-uncased_384
  - cached_train_semeval_bert-base-uncased_384
  - label.txt
  - test.tsv 
  - train.tsv
- eval/
  - answer_keys.txt
  - proposed_answers.txt
  - result.txt
  - semeval2010_task8_scorer-v1.2.pl
- model/
  - bert_tokenizer-0.1.5.tar.gz
  - config.json
  - model.safetensors
  - training_args.bin
  - bert_tokenizer-0.1.5/
  - Checkpoint/
  - Output/
- Test Output/
  - pred.tsv
- __pycache__/
```

## References

Monologg. (n.d.). Monologg/R-Bert: Pytorch implementation of R-bert: “enriching pre-trained language model with entity information for relation classification.” GitHub. [https://github.com/monologg/R-BERT](https://github.com/monologg/R-BERT)
```

