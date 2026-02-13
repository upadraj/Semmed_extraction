import os
import torch
import gc
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score, classification_report

# 1. HARDWARE SETUP
GPU_ID = "0" 
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

# 2. CONFIGURATION
base_model = "unsloth/Meta-Llama-3.1-8B-Instruct" 
max_seq_length = 2048 
dtype = torch.bfloat16 
load_in_4bit = False   

# 3. LOAD MODEL & TOKENIZER
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 4. LoRA SETUP
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 5. DATA PREPARATION
ALL_LABELS = ["ADMINISTERED_TO", "AFFECTS", "ASSOCIATED_WITH", "AUGMENTS", "CAUSES", 
              "COEXISTS_WITH", "COMPARED_WITH", "DIAGNOSES", "DISRUPTS", "INHIBITS", 
              "INTERACTS_WITH", "ISA", "LOCATION_OF", "None", "PART_OF", "PRECEDES", 
              "PREDISPOSES", "PREVENTS", "PROCESS_OF", "PRODUCES", "STIMULATES", "TREATS", "USES"]

def format_prompt(row, include_output=True):
    messages = [
        {"role": "system", "content": "You are a biomedical expert. Classify the relationship. Output ONLY the label string from the provided list."},
        {"role": "user", "content": f"Sentence: {row['sentence']}\nSubject: {row['subject_text']} ({row['subject_type']})\nObject: {row['object_text']} ({row['object_type']})"}
    ]
    if include_output:
        messages.append({"role": "assistant", "content": row['label']})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not include_output)

def load_and_prep(path):
    df = pd.read_csv(path, sep="\t", header=None, keep_default_na=False)
    df.columns = ["idx", "label", "sentence", "subject_text", "object_text", 
                  "subject_type", "object_type", "subject_group", "object_group", "label_choices"]
    df['text'] = df.apply(lambda x: format_prompt(x), axis=1)
    return df

train_df = load_and_prep("data/train.out")
val_df = load_and_prep("data/validation.out")
train_ds = Dataset.from_pandas(train_df[['text']])
val_ds = Dataset.from_pandas(val_df[['text']])

# 6. TRAINING
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=SFTConfig(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_steps=100, 
        learning_rate=2e-4,
        bf16=True, 
        logging_steps=5,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=3407,
    ),
)

trainer.train()

# 7. SAVE ADAPTERS AND MERGED MODEL
print("Saving model components...")
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
model.save_pretrained_merged("model_merged", tokenizer, save_method="merged_16bit")

# 8. INFERENCE & DETAILED EVALUATION
FastLanguageModel.for_inference(model)

def run_evaluation(df, output_csv="test_results_full.csv"):
    predictions = []
    raw_responses = []
    
    print(f"Predicting on {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt_text = format_prompt(row, include_output=False)
        inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
        
        # We use temperature=0 for consistent, greedy decoding in classification
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20, 
            use_cache=True,
            temperature=0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Extract only the newly generated text
        raw_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        raw_responses.append(raw_text)
        
        # Match logic
        final_pred = "None"
        for l in ALL_LABELS:
            if l.lower() in raw_text.lower():
                final_pred = l
                break
        predictions.append(final_pred)

    # Attach results to the dataframe
    df['model_raw_response'] = raw_responses
    df['predicted_label'] = predictions
    
    # Save the dataframe for inspection
    df.to_csv(output_csv, index=False, sep="\t")
    return predictions

print("Starting Final Evaluation...")
test_df = load_and_prep("data/test.out")
preds = run_evaluation(test_df)

# Final Reporting
print("\n--- PERFORMANCE SUMMARY ---")
print(classification_report(test_df['label'], preds))
print(f"Detailed logs saved to test_results_full.csv")