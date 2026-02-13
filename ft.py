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

# 2. CONFIGURATION - High Precision Mode
base_model = "unsloth/Meta-Llama-3.1-8B-Instruct" # Using the non-quantized base
max_seq_length = 2048 
dtype = torch.bfloat16 # Native H100 format
load_in_4bit = False   # DISABLED QUANTIZATION

# 3. LOAD MODEL & TOKENIZER
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 4. LoRA SETUP (Still recommended for stability even without 4-bit)
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

# 5. DATA PREPARATION (Labels and Formatting)
ALL_LABELS = ["ADMINISTERED_TO", "AFFECTS", "ASSOCIATED_WITH", "AUGMENTS", "CAUSES", 
              "COEXISTS_WITH", "COMPARED_WITH", "DIAGNOSES", "DISRUPTS", "INHIBITS", 
              "INTERACTS_WITH", "ISA", "LOCATION_OF", "None", "PART_OF", "PRECEDES", 
              "PREDISPOSES", "PREVENTS", "PROCESS_OF", "PRODUCES", "STIMULATES", "TREATS", "USES"]

def format_prompt(row, include_output=True):
    messages = [
        {"role": "system", "content": "You are a biomedical expert. Classify the relationship. Output ONLY the label."},
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

# 7. SAVE MODEL
print("Saving model...")
model.save_pretrained("lora_model")  # Saves LoRA adapters
tokenizer.save_pretrained("lora_model")  # Saves tokenizer

# Optionally save merged model (LoRA + base model)
print("Saving merged model...")
model.save_pretrained_merged("model_merged", tokenizer, save_method="merged_16bit")

# 8. INFERENCE & EVALUATION
FastLanguageModel.for_inference(model)

def predict(df):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt_text = format_prompt(row, include_output=False)
        inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=20, use_cache=True)
        prediction = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Exact or partial match logic
        final_pred = "None"
        for l in ALL_LABELS:
            if l.lower() in prediction.lower():
                final_pred = l
                break
        results.append(final_pred)
    return results

print("Evaluating Test Set...")
test_df = load_and_prep("data/test.out")
preds = predict(test_df)
print(classification_report(test_df['label'], preds))