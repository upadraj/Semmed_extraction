"""
convert_few_shot_to_json.py

This script converts few-shot examples from TSV format (data/few_shot.out)
to JSON format (data/few_shot.json) for use in the DataModule.
"""

import pandas as pd
import json
import os


def load_few_shot_from_tsv(tsv_path="data/few_shot.out", nrows=None ):
    """
    Load few-shot examples from TSV file.
    
    Args:
        tsv_path: Path to the TSV file containing few-shot examples
        
    Returns:
        pd.DataFrame: Loaded dataframe with standard columns
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Few-shot file not found: {tsv_path}")
    
    # Load TSV file (same format as train/test files)
    df = pd.read_csv(tsv_path, sep="\t", header=None, keep_default_na=False, nrows=nrows)
    
    # Assign column names (same as in DataModule.load_and_prep)
    df.columns = [
        "idx", "label", "sentence", "subject_text", "object_text",
        "subject_type", "object_type", "subject_group", "object_group", "label_choices"
    ]
    
    return df


def convert_to_few_shot_format(df):
    """
    Convert dataframe to few-shot examples format.
    
    Args:
        df: DataFrame with few-shot data
        
    Returns:
        list: List of dictionaries in FEW_SHOT_EXAMPLES format
    """
    few_shot_examples = []
    
    for _, row in df.iterrows():
        example = {
            "sentence": row["sentence"],
            "subject_text": row["subject_text"],
            "subject_type": row["subject_type"],
            "object_text": row["object_text"],
            "object_type": row["object_type"],
            "label": row["label"]
        }
        few_shot_examples.append(example)
    
    return few_shot_examples


def save_to_json(examples, json_path="data/few_shot.json"):
    """
    Save few-shot examples to JSON file.
    
    Args:
        examples: List of few-shot example dictionaries
        json_path: Path to save the JSON file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Save with pretty formatting
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(examples)} few-shot examples to: {json_path}")


def main():
    """
    Main conversion function.
    """
    print("=" * 80)
    print("FEW-SHOT EXAMPLES CONVERTER")
    print("=" * 80)
    
    tsv_path = "data/few_shot.out"
    json_path = "data/few_shot.json"
    
    # Step 1: Load TSV file
    print(f"\n[1/3] Loading few-shot examples from: {tsv_path}")
    try:
        df = load_few_shot_from_tsv(tsv_path, nrows=100)
        print(f"✓ Loaded {len(df)} examples")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease ensure 'data/few_shot.out' exists with your few-shot examples.")
        return
    
    # Step 2: Convert to few-shot format
    print(f"\n[2/3] Converting to few-shot format...")
    examples = convert_to_few_shot_format(df)
    print(f"✓ Converted {len(examples)} examples")
    
    # Display sample
    if examples:
        print("\nSample example:")
        print(json.dumps(examples[0], indent=2))
    
    # Step 3: Save to JSON
    print(f"\n[3/3] Saving to JSON file...")
    save_to_json(examples, json_path)
    
    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE!")
    print("=" * 80)
    print(f"\nFew-shot examples saved to: {json_path}")
    print(f"Total examples: {len(examples)}")
    print("\nNext steps:")
    print("  1. Review the generated JSON file")
    print("  2. The DataModule will automatically load from this file")
    print("  3. Run train.py or test.py with USE_FEW_SHOT = True")


if __name__ == "__main__":
    main()

