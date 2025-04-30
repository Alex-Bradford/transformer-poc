#!/usr/bin/env python3
"""
Script to download WikiText-2 raw dataset splits and save to local files.
"""
import os
from datasets import load_dataset

def save_split(split_name, lines, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"wiki.{split_name}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        count = 0
        for line in lines:
            if line and line.strip():
                f.write(line + "\n")
                count += 1
    print(f"Saved '{split_name}' split: {count} non-empty lines → {out_path}")


def main():
    # Load the raw WikiText-2 dataset
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    for split in ["train", "validation", "test"]:
        save_split(split, raw[split]["text"])


if __name__ == '__main__':
    main()
