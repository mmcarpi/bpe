"""
train.py

Train a BPE tokenizer.
"""

import argparse
from pathlib import Path

from tokenizer import Tokenizer, SPLIT_PATTERNS


def main():
    parser = argparse.ArgumentParser(
        prog="train",
        description="Train a Byte Pair Encoding merge table and vocabulary from text.",
    )

    parser.add_argument("name", help="Name of the model to be saved.")
    parser.add_argument("vocabsize", type=int, help="Maximum size for the vocabulary.")

    parser.add_argument(
        "inputfile",
        help="UTF-8 encoded file that will be used to train the tokenizer.",
    )

    parser.add_argument(
        "--pattern",
        choices=SPLIT_PATTERNS.keys(),
        default="default",
        help="How to split the text.",
    )

    parser.add_argument(
        "--outpath",
        nargs="?",
        default=Path("."),
        help="Directory to save the file. Defaults to current directory.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="Early stop criteria. If the pair to be merged does not appear more than THRESHOLD the program stops. Defaults to 1.",
    )

    parser.add_argument(
        "--run-dry",
        action="store_true",
        help="Train vocabulary, but don't write anything in disk.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print which merges are being made at each step.",
    )

    args = parser.parse_args()

    print("NAME:", args.name)
    print("VOCABSIZE:", args.vocabsize)
    print("THRESHOLD:", args.threshold)
    print("PATTERN:", args.pattern)
    print("INPUTFILE:", args.inputfile)
    print("OUTPATH:", args.outpath)

    tokenizer = Tokenizer(pattern=SPLIT_PATTERNS[args.pattern])
    with open(args.inputfile, "r", encoding="utf-8") as f:
        tokenizer.train(
            f.read(),
            vocabsize=args.vocabsize,
            threshold=args.threshold,
            verbose=args.verbose,
        )

    if not args.run_dry:
        tokenizer.save(args.outpath, args.name)


if __name__ == "__main__":
    main()
