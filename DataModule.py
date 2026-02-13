"""
DataModule.py
Shared module containing model/tokenizer loading, data preparation utilities,
configuration constants, and shared utilities for fine-tuning.
"""

import os
import json
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Hardware Configuration
GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

# Model Configuration
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16
LOAD_IN_4BIT = False

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj"
]
LORA_BIAS = "none"
USE_GRADIENT_CHECKPOINTING = "unsloth"
RANDOM_STATE = 3407

# Training Configuration
TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_STEPS = 5
MAX_STEPS = 100
LEARNING_RATE = 2e-4
BF16 = True
LOGGING_STEPS = 5
OUTPUT_DIR = "outputs"
OPTIM = "adamw_8bit"

# Data Configuration
DATA_TRAIN_PATH = "data/train.out"
DATA_VAL_PATH = "data/validation.out"
DATA_TEST_PATH = "data/test.out"

# Label Configuration
ALL_LABELS = [
    "ADMINISTERED_TO", "AFFECTS", "ASSOCIATED_WITH", "AUGMENTS", "CAUSES",
    "COEXISTS_WITH", "COMPARED_WITH", "DIAGNOSES", "DISRUPTS", "INHIBITS",
    "INTERACTS_WITH", "ISA", "LOCATION_OF", "None", "PART_OF", "PRECEDES",
    "PREDISPOSES", "PREVENTS", "PROCESS_OF", "PRODUCES", "STIMULATES", "TREATS", "USES"
]

# Model Paths
LORA_MODEL_PATH = "lora_model"
MERGED_MODEL_PATH = "model_merged"

# Few-Shot Learning Configuration
USE_FEW_SHOT = True  # Set to True to enable few-shot examples
FEW_SHOT_JSON_PATH = "data/few_shot.json"  # Path to few-shot examples JSON file

# Few-shot examples - loaded from JSON file or fallback to defaults
def _load_few_shot_examples():
    """
    Load few-shot examples from JSON file.
    Falls back to default examples if file doesn't exist.

    Returns:
        list: List of few-shot example dictionaries
    """
    # Try to load from JSON file first
    if os.path.exists(FEW_SHOT_JSON_PATH):
        try:
            with open(FEW_SHOT_JSON_PATH, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            print(f"✓ Loaded {len(examples)} few-shot examples from {FEW_SHOT_JSON_PATH}")
            return examples
        except Exception as e:
            print(f"⚠ Warning: Could not load {FEW_SHOT_JSON_PATH}: {e}")
            print("  Using default few-shot examples instead.")

    # Fallback to default examples if JSON file doesn't exist
    print(f"ℹ Using default few-shot examples (JSON file not found: {FEW_SHOT_JSON_PATH})")
    return [
        {
            "sentence": "Aspirin reduces the risk of heart attack.",
            "subject_text": "Aspirin",
            "subject_type": "Chemical",
            "object_text": "heart attack",
            "object_type": "Disease",
            "label": "PREVENTS"
        },
        {
            "sentence": "Diabetes is associated with increased cardiovascular risk.",
            "subject_text": "Diabetes",
            "subject_type": "Disease",
            "object_text": "cardiovascular risk",
            "object_type": "Disease",
            "label": "ASSOCIATED_WITH"
        },
        {
            "sentence": "Insulin is used to treat diabetes mellitus.",
            "subject_text": "Insulin",
            "subject_type": "Chemical",
            "object_text": "diabetes mellitus",
            "object_type": "Disease",
            "label": "TREATS"
        },
        {
            "sentence": "The liver is part of the digestive system.",
            "subject_text": "liver",
            "subject_type": "Organ",
            "object_text": "digestive system",
            "object_type": "System",
            "label": "PART_OF"
        },
        {
            "sentence": "Smoking causes lung cancer.",
            "subject_text": "Smoking",
            "subject_type": "Behavior",
            "object_text": "lung cancer",
            "object_type": "Disease",
            "label": "CAUSES"
        }
    ]

# Load few-shot examples (will be loaded once when module is imported)
FEW_SHOT_EXAMPLES = _load_few_shot_examples()

# This ensures it always matches the actual number of examples available
NUM_FEW_SHOT_EXAMPLES = len(FEW_SHOT_EXAMPLES)

# ============================================================================
# MODEL AND TOKENIZER LOADING
# ============================================================================

def load_model_and_tokenizer(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT
):
    """
    Load the base model and tokenizer.
    
    Args:
        model_name: Name or path of the model to load
        max_seq_length: Maximum sequence length
        dtype: Data type for model weights
        load_in_4bit: Whether to load model in 4-bit quantization
        
    Returns:
        tuple: (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def setup_lora_model(
    model,
    r=LORA_R,
    target_modules=LORA_TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    random_state=RANDOM_STATE
):
    """
    Apply LoRA (Low-Rank Adaptation) to the model.
    
    Args:
        model: The base model to apply LoRA to
        r: LoRA rank
        target_modules: List of module names to apply LoRA to
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout rate for LoRA layers
        bias: Bias configuration
        use_gradient_checkpointing: Gradient checkpointing strategy
        random_state: Random seed
        
    Returns:
        model: Model with LoRA applied
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
    )
    return model


# ============================================================================
# DATA PREPARATION UTILITIES
# ============================================================================

def format_prompt(row, tokenizer, include_output=True, use_few_shot=USE_FEW_SHOT,
                  few_shot_examples=None, num_examples=None):
    """
    Format a data row into a chat-style prompt with optional few-shot examples.

    Args:
        row: DataFrame row containing sentence, subject, object, and label
        tokenizer: Tokenizer to use for formatting
        include_output: Whether to include the assistant's response (label)
        use_few_shot: Whether to include few-shot examples in the prompt
        few_shot_examples: List of example dictionaries (uses FEW_SHOT_EXAMPLES if None)
        num_examples: Number of few-shot examples to include (uses all if None)

    Returns:
        str: Formatted prompt text
    """
    # Start with system message
    messages = [
        {
            "role": "system",
            "content": "You are a biomedical expert. Classify the relationship. Output ONLY the label string from the provided list."
        }
    ]

    # Add few-shot examples if enabled
    if use_few_shot:
        if few_shot_examples is None:
            few_shot_examples = FEW_SHOT_EXAMPLES

        # Default to using all available examples
        if num_examples is None:
            num_examples = len(few_shot_examples)

        # Add the specified number of examples
        for example in few_shot_examples[:num_examples]:
            # User message with example
            messages.append({
                "role": "user",
                "content": f"Sentence: {example['sentence']}\nSubject: {example['subject_text']} ({example['subject_type']})\nObject: {example['object_text']} ({example['object_type']})"
            })
            # Assistant response with label
            messages.append({
                "role": "assistant",
                "content": example['label']
            })

    # Add the actual query
    messages.append({
        "role": "user",
        "content": f"Sentence: {row['sentence']}\nSubject: {row['subject_text']} ({row['subject_type']})\nObject: {row['object_text']} ({row['object_type']})"
    })

    # Add the expected output if training
    if include_output:
        messages.append({"role": "assistant", "content": row['label']})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not include_output
    )


def load_and_prep(path, tokenizer, use_few_shot=USE_FEW_SHOT,
                  few_shot_examples=None, num_examples=None, nrows=None):
    """
    Load and prepare data from a TSV file.

    Args:
        path: Path to the TSV data file
        tokenizer: Tokenizer to use for formatting prompts
        use_few_shot: Whether to include few-shot examples in prompts
        few_shot_examples: List of example dictionaries (uses FEW_SHOT_EXAMPLES if None)
        num_examples: Number of few-shot examples to include (uses all if None)
        nrows: Number of rows to load from file (None = load all)

    Returns:
        pd.DataFrame: Prepared dataframe with formatted text column
    """
    # Default to using all available examples
    if num_examples is None:
        num_examples = len(FEW_SHOT_EXAMPLES)

    # Load data with optional row limit
    df = pd.read_csv(path, sep="\t", header=None, keep_default_na=False, nrows=nrows)
    df.columns = [
        "idx", "label", "sentence", "subject_text", "object_text",
        "subject_type", "object_type", "subject_group", "object_group", "label_choices"
    ]
    df['text'] = df.apply(
        lambda x: format_prompt(x, tokenizer, use_few_shot=use_few_shot,
                               few_shot_examples=few_shot_examples,
                               num_examples=num_examples),
        axis=1
    )
    return df


def prepare_datasets(train_path, val_path, tokenizer, use_few_shot=USE_FEW_SHOT,
                    few_shot_examples=None, num_examples=None, nrows=None):
    """
    Prepare training and validation datasets.

    Args:
        train_path: Path to training data file
        val_path: Path to validation data file
        tokenizer: Tokenizer to use for formatting
        use_few_shot: Whether to include few-shot examples in prompts
        few_shot_examples: List of example dictionaries (uses FEW_SHOT_EXAMPLES if None)
        num_examples: Number of few-shot examples to include (uses all if None)
        nrows: Number of rows to load from each file (None = load all)

    Returns:
        tuple: (train_df, val_df, train_ds, val_ds)
    """
    # Default to using all available examples
    if num_examples is None:
        num_examples = len(FEW_SHOT_EXAMPLES)

    # Load data with optional row limit
    train_df = load_and_prep(train_path, tokenizer, use_few_shot, few_shot_examples, num_examples, nrows)
    val_df = load_and_prep(val_path, tokenizer, use_few_shot, few_shot_examples, num_examples, nrows)

    train_ds = Dataset.from_pandas(train_df[['text']])
    val_ds = Dataset.from_pandas(val_df[['text']])
    return train_df, val_df, train_ds, val_ds

