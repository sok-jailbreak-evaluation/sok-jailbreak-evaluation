import csv
import random
import argparse
from ptlp import MappingPtlp


JAILBREAK_PROMPTS_PATH = "jailbreak_prompts.csv"
MALICIOUS_PROMPTS_PATH = "malicious_prompts.csv"


def get_jailbreak_prompt(index=None):
    prompts = []
    followups = []
    with open(JAILBREAK_PROMPTS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "user_input" in row and "followup" in row:
                prompts.append(row["user_input"].strip())
                followups.append(row["followup"].strip())
    if not prompts:
        return None, None
    if index is not None:
        if 0 <= index < len(prompts):
            return prompts[index], followups[index]
        else:
            raise IndexError(
                f"Jailbreak prompt index {index} out of range (0-{len(prompts)-1})"
            )
    idx = random.randint(0, len(prompts) - 1)
    return prompts[idx], followups[idx]


def get_malicious_prompt(index=None):
    prompts = []
    with open(MALICIOUS_PROMPTS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "tasks" in row:
                prompts.append(row["tasks"].strip())
    if not prompts:
        return None
    if index is not None:
        if 0 <= index < len(prompts):
            return prompts[index]
        else:
            raise IndexError(
                f"Malicious prompt index {index} out of range (0-{len(prompts)-1})"
            )
    return random.choice(prompts)


def combine_prompts(jailbreak_prompt, malicious_prompt):
    """
    Concatenate the jailbreak and malicious prompts with two newlines in between.
    """
    return f"{jailbreak_prompt}\n\n{malicious_prompt}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate adversarial prompt combinations."
    )
    parser.add_argument(
        "--jailbreak-index",
        type=int,
        default=None,
        help="Index of jailbreak prompt to use",
    )
    parser.add_argument(
        "--malicious-index",
        type=int,
        default=None,
        help="Index of malicious prompt to use",
    )
    parser.add_argument(
        "--puzzle-type",
        type=str,
        choices=["ptlp"],
        default="ptlp",
        help="Type of puzzle to use: ptlp (Pseudo Time-Lock Puzzle)",
    )
    args = parser.parse_args()

    try:
        jailbreak_prompt, jailbreak_followup = get_jailbreak_prompt(args.jailbreak_index)
    except IndexError as e:
        print(e)
        return
    try:
        malicious_prompt = get_malicious_prompt(args.malicious_index)
    except IndexError as e:
        print(e)
        return
    if not jailbreak_prompt or not malicious_prompt:
        print("Could not load prompts.")
        return
    combined_prompt = combine_prompts(jailbreak_prompt, malicious_prompt)
    print(f"{combined_prompt}\n")
    print("-" * 40 + "\n")
    
    puzzle = MappingPtlp()
    encoded = puzzle.encode(combined_prompt)
    print(f"{puzzle.prefix}\n")
    print(f"{puzzle.description}\n")
    print(f"{puzzle.key_prefix}\n\n{puzzle.key}\n")
    print(f"{puzzle.ciphertext_prefix}\n\nBEGINNING OF CIPHERTEXT\n{encoded}\nEND OF CIPHERTEXT\n")
    print(f"{puzzle.postfix}\n")

    print("-" * 40 + "\n")
    print(f"{jailbreak_followup}")


if __name__ == "__main__":
    main()
