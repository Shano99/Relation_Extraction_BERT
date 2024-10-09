import os

# Directory for evaluation files
EVAL_DIR = "eval"

# Function to compute macro-averaged F1 score using an official scorer script
def official_f1():
    # Run the perl script
    try:
        cmd = "perl {0}/semeval2010_task8_scorer-v1.2.pl {0}/proposed_answers.txt {0}/answer_keys.txt > {0}/result.txt".format(
            EVAL_DIR
        )
        os.system(cmd)
    except:
        raise Exception("perl is not installed or proposed_answers.txt is missing")

    # Parse the result file to extract the macro-averaged F1 score
    with open(os.path.join(EVAL_DIR, "result.txt"), "r", encoding="utf-8") as f:
        macro_result = list(f)[-1]  # Get the last line which contains the macro-averaged F1 score
        macro_result = macro_result.split(":")[1].replace(">>>", "").strip()  # Extract the score
        macro_result = macro_result.split("=")[1].strip().replace("%", "")  # Remove "%" and extract the value
        macro_result = float(macro_result) / 100  # Convert to float and normalize to range [0, 1]

    return macro_result

# Entry point of the script
if __name__ == "__main__":
    print("macro-averaged F1 = {}%".format(official_f1() * 100))  # Print the macro-averaged F1 score in percentage
