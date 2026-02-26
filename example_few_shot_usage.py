"""
example_few_shot_usage.py

This script demonstrates how to use the few-shot learning feature.
"""

from DataModule import (
    load_model_and_tokenizer,
    format_prompt,
    FEW_SHOT_EXAMPLES,
    BASE_MODEL,
    MAX_SEQ_LENGTH,
    DTYPE,
    LOAD_IN_4BIT
)
import pandas as pd

# Example: Load model and tokenizer
print("Loading model and tokenizer...")
model, tokenizer = load_model_and_tokenizer(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT
)
print("✓ Model loaded\n")

# Example data row
example_row = {
    "sentence": "Metformin is used to treat type 2 diabetes.",
    "subject_text": "Metformin",
    "subject_type": "Chemical",
    "object_text": "type 2 diabetes",
    "object_type": "Disease",
    "label": "TREATS"
}

print("=" * 80)
print("EXAMPLE 1: Prompt WITHOUT Few-Shot Examples")
print("=" * 80)

# Format prompt without few-shot
prompt_no_few_shot = format_prompt(
    example_row,
    tokenizer,
    include_output=False,
    use_few_shot=False
)

print(prompt_no_few_shot)
print("\n")

print("=" * 80)
print("EXAMPLE 2: Prompt WITH Few-Shot Examples (3 examples)")
print("=" * 80)

# Format prompt with few-shot
prompt_with_few_shot = format_prompt(
    example_row,
    tokenizer,
    include_output=False,
    use_few_shot=True,
    num_examples=3
)

print(prompt_with_few_shot)
print("\n")

print("=" * 80)
print("EXAMPLE 3: Prompt WITH Custom Few-Shot Examples")
print("=" * 80)

# Custom examples
custom_examples = [
    {
        "sentence": "Chemotherapy treats cancer.",
        "subject_text": "Chemotherapy",
        "subject_type": "Treatment",
        "object_text": "cancer",
        "object_type": "Disease",
        "label": "TREATS"
    },
    {
        "sentence": "Obesity causes heart disease.",
        "subject_text": "Obesity",
        "subject_type": "Condition",
        "object_text": "heart disease",
        "object_type": "Disease",
        "label": "CAUSES"
    }
]

prompt_custom = format_prompt(
    example_row,
    tokenizer,
    include_output=False,
    use_few_shot=True,
    few_shot_examples=custom_examples,
    num_examples=2
)

print(prompt_custom)
print("\n")

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Prompt length without few-shot: ~{len(prompt_no_few_shot)} characters")
print(f"Prompt length with 3 few-shot examples: ~{len(prompt_with_few_shot)} characters")
print(f"Prompt length with 2 custom examples: ~{len(prompt_custom)} characters")
print("\nTo enable few-shot in training/testing:")
print("  1. Edit DataModule.py")
print("  2. Set USE_FEW_SHOT = True")
print("  3. NUM_FEW_SHOT_EXAMPLES is automatically set to match the number of examples")
print("  4. Run python train.py or python test.py")

