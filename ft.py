#!/usr/bin/env python
# coding: utf-8
"""
Fine-tuning script for biomedical relation extraction using TSV data
Adapted from Subject-object/ft.py for train.out, test.out, validation.out format
"""

import os
import sys
print(os.environ.pop("CUDA_VISIBLE_DEVICES", None))

import torch
import numpy as np
import pandas as pd
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from tqdm import tqdm
from unsloth import FastLanguageModel, is_bfloat16_supported
import json
import re

# Setup
os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Check GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

np.random.seed(48)

# Model configuration
# base_model = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
base_model = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"  # Use this for smaller GPU

print(f"Loading base model: {base_model}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model,
    max_seq_length=512,
    dtype=torch.float16,
    load_in_4bit=True,
    device_map="auto",
    attn_implementation="flash_attention_2",
    rope_scaling=None,
    cache_dir=None,
)

model.gradient_checkpointing_enable()

# Data loading function
def load_tsv_data(tsv_path: str) -> pd.DataFrame:
    """Load TSV data file with standard column names."""
    df = pd.read_csv(tsv_path, sep="\t", header=None, keep_default_na=False)
    df.columns = ["idx", "label", "sentence", "subject_text", "object_text", 
                  "subject_type", "object_type", "subject_group", "object_group", "label_choices"]
    return df

# All possible relation labels in the dataset
ALL_LABELS = [
    "ADMINISTERED_TO",
    "AFFECTS",
    "ASSOCIATED_WITH",
    "AUGMENTS",
    "CAUSES",
    "COEXISTS_WITH",
    "COMPARED_WITH",
    "DIAGNOSES",
    "DISRUPTS",
    "INHIBITS",
    "INTERACTS_WITH",
    "ISA",
    "LOCATION_OF",
    "None",
    "PART_OF",
    "PRECEDES",
    "PREDISPOSES",
    "PREVENTS",
    "PROCESS_OF",
    "PRODUCES",
    "STIMULATES",
    "TREATS",
    "USES"
]

# Prompt template
prompt = f"""You are an expert in biomedical relation extraction.
Your task is to classify the relationship between two entities in a biomedical sentence.

Given a sentence with a subject entity and an object entity, predict the relationship label.

Possible labels: {', '.join(ALL_LABELS)}

Output ONLY the label name, nothing else.
If there is no clear relationship, output "None"."""

def train_tokenize(few_shot, df, prompt):
    """Convert dataframe to tokenized chat format for training."""
    formatted_texts = []

    for i in range(len(df)):
        chat = [{"role": "system", "content": prompt}]
        chat = chat + few_shot

        row = df.iloc[i]
        # User message: sentence with subject and object entities
        user_content = f"""Sentence: {row['sentence']} Subject: {row['subject_text']} (Type: {row['subject_type']}) Object: {row['object_text']} (Type: {row['object_type']})"""
        chat.append({"role": "user", "content": user_content})

        # Assistant response: ONLY the label
        label = row['label']
        chat.append({"role": "assistant", "content": label})

        formatted_text = tokenizer.apply_chat_template(chat, tokenize=False)
        formatted_texts.append(formatted_text)

    return Dataset.from_dict({"text": formatted_texts})

# Load data
print("\nLoading data...")
df_train = load_tsv_data("data/train.out")
df_test = load_tsv_data("data/test.out")
df_val = load_tsv_data("data/validation.out")

print(f"Train samples: {len(df_train)}")
print(f"Test samples: {len(df_test)}")
print(f"Validation samples: {len(df_val)}")

# Create few-shot examples from training samples
# Use one example for each of the most common labels
few_shot = []
few_shot_indices = []  # Track indices of few-shot examples to exclude from training

for label in ["LOCATION_OF", "PROCESS_OF", "INTERACTS_WITH", "None"]:
    sample_df = df_train[df_train['label'] == label]
    sample = sample_df.iloc[0]
    # Get the actual index in the original dataframe
    sample_idx = sample_df.index[0]
    few_shot_indices.append(sample_idx)

    user_content = f"""Sentence: {sample['sentence']}
Subject: {sample['subject_text']} (Type: {sample['subject_type']})
Object: {sample['object_text']} (Type: {sample['object_type']})"""
    few_shot.append({"role": "user", "content": user_content})
    few_shot.append({"role": "assistant", "content": sample['label']})

print(f"\nFew-shot examples created from indices: {few_shot_indices}")
print(f"These {len(few_shot_indices)} examples will be excluded from training data")

# Setup output directory
ft_pth = f"./finetuned_model/{base_model.replace('/', '_')}"
if not os.path.exists(ft_pth):
    os.makedirs(ft_pth)

print(f"\nModel will be saved to: {ft_pth}")

# Prepare training data (excluding few-shot examples)
print("\nPreparing training data...")
df_train_filtered = df_train.drop(few_shot_indices).reset_index(drop=True)
print(f"Original training samples: {len(df_train)}")
print(f"Filtered training samples (excluding few-shot): {len(df_train_filtered)}")
data = train_tokenize(few_shot, df_train_filtered, prompt)
print(f"Formatted {len(data)} training examples")


def training(tokenizer, model, data, ft_pth):
    """Train the model using SFT (Supervised Fine-Tuning)."""
    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data,
        args=SFTConfig(
            packing=False,
            dataset_num_proc=2,
            dataset_text_field="text",
            learning_rate=3e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            warmup_steps=4,
            num_train_epochs=2,
            fp16=True,
            bf16=False,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="checkpoints",
            report_to="none",
            logging_dir="logs",
        ),
    )

    train_result = trainer.train()
    print(f"\nTraining completed!")
    print(train_result)

    print(f"Saving model to {ft_pth}...")
    trainer.save_model(ft_pth)
    tokenizer.save_pretrained(ft_pth)
    print("Model saved successfully!")


def convert_label(output_text):
    """Extract label from model output."""
    # Clean the output
    output_text = output_text.strip()

    # Check if output is one of the valid labels
    if output_text in ALL_LABELS:
        return output_text

    # Try to find a valid label in the output
    for label in ALL_LABELS:
        if label in output_text:
            return label

    # Default to None if no valid label found
    return "None"


def model_inference(few_shot, model, tokenizer, df, prompt, device):
    """Run inference on test data."""
    print("\nRunning inference...")
    model = FastLanguageModel.for_inference(model).to(device)
    predicted_labels = []

    for i in tqdm(range(len(df)), desc="Inference"):
        chat = [{"role": "system", "content": prompt}]
        chat = chat + few_shot

        row = df.iloc[i]
        user_content = f"""Sentence: {row['sentence']}
Subject: {row['subject_text']} (Type: {row['subject_type']})
Object: {row['object_text']} (Type: {row['object_type']})"""
        chat.append({"role": "user", "content": user_content})

        f_input = tokenizer.apply_chat_template(chat, tokenize=False)
        tokenized_input = tokenizer(
            f_input, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        input_length = tokenized_input["input_ids"].shape[1]
        output = model.generate(**tokenized_input, max_new_tokens=50)
        generated_tokens = output[0, input_length:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        label = convert_label(output_text)
        predicted_labels.append(label)

    return predicted_labels


# Uncomment to train
# training(tokenizer, model, data, ft_pth)

print("\n" + "="*80)
print("Training setup complete!")
print("To train the model, uncomment the training() call at the end of this script")
print("="*80)

