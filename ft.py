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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import re
import pickle

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
    max_seq_length=1024,  # Increased from 512 to handle longer sequences
    dtype=torch.float16,
    load_in_4bit=True,
    device_map="auto",
    attn_implementation="flash_attention_2",
    rope_scaling=None,
    cache_dir=None,
)

model.gradient_checkpointing_enable()

# Add LoRA adapters for training on quantized model
print("\nAdding LoRA adapters for efficient fine-tuning...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",     # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None, # And LoftQ
)
print("LoRA adapters added successfully!")

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

# Note: We always re-tokenize to ensure compatibility with current model settings
# If you want to cache, make sure max_seq_length hasn't changed
print("\nTokenizing training data (this may take a while)...")
data = train_tokenize(few_shot, df_train_filtered, prompt)
print(f"Formatted {len(data)} training examples")

# Save tokenized data for inspection
tokenized_data_path = "data/tokenized_train_data.pkl"
print(f"Saving tokenized data to {tokenized_data_path}...")
os.makedirs("data", exist_ok=True)
with open(tokenized_data_path, 'wb') as f:
    pickle.dump(data, f)
print("Tokenized data saved!")

# Also save a human-readable sample
sample_path = "data/tokenized_sample.txt"
with open(sample_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("SAMPLE OF TOKENIZED TRAINING DATA (First 3 examples)\n")
    f.write("="*80 + "\n\n")
    for i in range(min(3, len(data))):
        f.write(f"\n{'='*80}\n")
        f.write(f"Example {i+1}:\n")
        f.write(f"{'='*80}\n")
        f.write(data[i]['text'])
        f.write("\n")
print(f"Sample of tokenized data saved to {sample_path}")

# Prepare validation data for evaluation during training
print("\nPreparing validation data...")
print("Tokenizing validation data...")
val_data = train_tokenize(few_shot, df_val, prompt)
print(f"Formatted {len(val_data)} validation examples")

# Save validation data
tokenized_val_path = "data/tokenized_val_data.pkl"
with open(tokenized_val_path, 'wb') as f:
    pickle.dump(val_data, f)
print(f"Tokenized validation data saved to {tokenized_val_path}")


def training(tokenizer, model, data, val_data, ft_pth):
    """Train the model using SFT (Supervised Fine-Tuning) with validation."""
    print("\nStarting training...")
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Training samples: {len(data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Learning rate: 3e-4")
    print(f"Batch size: 2 (per device)")
    print(f"Gradient accumulation steps: 2")
    print(f"Effective batch size: {2 * 2} (batch_size * grad_accum)")
    print(f"Epochs: 2")
    print(f"Optimizer: adamw_8bit")
    print("="*80 + "\n")

    print("HOW LOSS IS CALCULATED:")
    print("-" * 80)
    print("During training, the model:")
    print("1. Receives the input (system prompt + few-shot + user question)")
    print("2. Predicts the next tokens (the label)")
    print("3. Compares predictions with TRUE LABEL from your data")
    print("4. Calculates cross-entropy loss between predicted and true tokens")
    print("5. Updates model weights to minimize this loss")
    print("-" * 80 + "\n")

    # Set this right before creating trainer (required by Unsloth 2024.11+)
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data,
        eval_dataset=val_data,  # Add validation dataset
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
            # Evaluation settings
            eval_strategy="steps",  # Evaluate during training
            eval_steps=50,  # Evaluate every 50 steps
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
    )

    train_result = trainer.train()
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Training steps: {train_result.global_step}")
    print("="*80 + "\n")

    print(f"Saving model to {ft_pth}...")
    trainer.save_model(ft_pth)
    tokenizer.save_pretrained(ft_pth)
    print("Model saved successfully!")

    return trainer


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
        user_content = f"""Sentence: {row['sentence']} Subject: {row['subject_text']} (Type: {row['subject_type']}) Object: {row['object_text'] }(Type: {row['object_type']})"""
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


def evaluate_model(few_shot, model, tokenizer, df, prompt, device, dataset_name="Test"):
    """Evaluate model and print detailed metrics."""
    print(f"\n{'='*80}")
    print(f"EVALUATING ON {dataset_name.upper()} SET")
    print("="*80)

    # Get predictions
    predicted_labels = model_inference(few_shot, model, tokenizer, df, prompt, device)
    true_labels = df['label'].tolist()

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\n{dataset_name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Print classification report
    print(f"\n{'-'*80}")
    print("CLASSIFICATION REPORT:")
    print("-"*80)
    print(classification_report(true_labels, predicted_labels, zero_division=0))

    # Save predictions to file
    results_df = df.copy()
    results_df['predicted_label'] = predicted_labels
    results_df['correct'] = results_df['label'] == results_df['predicted_label']

    output_file = f"results/{dataset_name.lower()}_predictions.tsv"
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(output_file, sep='\t', index=False)
    print(f"\nPredictions saved to: {output_file}")

    # Show some examples of correct and incorrect predictions
    print(f"\n{'-'*80}")
    print("SAMPLE CORRECT PREDICTIONS:")
    print("-"*80)
    correct_samples = results_df[results_df['correct']].head(3)
    for _, row in correct_samples.iterrows():
        print(f"\nSentence: {row['sentence'][:100]}...")
        print(f"True: {row['label']} | Predicted: {row['predicted_label']} ✓")

    print(f"\n{'-'*80}")
    print("SAMPLE INCORRECT PREDICTIONS:")
    print("-"*80)
    incorrect_samples = results_df[~results_df['correct']].head(3)
    for _, row in incorrect_samples.iterrows():
        print(f"\nSentence: {row['sentence'][:100]}...")
        print(f"True: {row['label']} | Predicted: {row['predicted_label']} ✗")

    print("="*80 + "\n")

    return accuracy, predicted_labels


print("\n" + "="*80)
print("STARTING TRAINING AND EVALUATION PIPELINE")
print("="*80)

# Train the model
trainer = training(tokenizer, model, data, val_data, ft_pth)

# Evaluate on validation set
print("\n" + "="*80)
print("EVALUATING ON VALIDATION SET")
print("="*80)
val_accuracy, _ = evaluate_model(few_shot, model, tokenizer, df_val, prompt, device, "Validation")

# Evaluate on test set
print("\n" + "="*80)
print("EVALUATING ON TEST SET")
print("="*80)
test_accuracy, _ = evaluate_model(few_shot, model, tokenizer, df_test, prompt, device, "Test")

# Final summary
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print("="*80)


