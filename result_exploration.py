import os
import torch
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from sklearn.metrics import classification_report

# 1. HARDWARE & PATHS
GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
LORA_PATH = "lora_model" 
TEST_DATA_PATH = "data/test.out"

# 2. CONFIGURATION
max_seq_length = 2048
dtype = torch.bfloat16
load_in_4bit = False 

# 3. LOAD MODEL & TOKENIZER
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = LORA_PATH, 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

# 4. LABEL DEFINITIONS
ALL_LABELS = ["ADMINISTERED_TO", "AFFECTS", "ASSOCIATED_WITH", "AUGMENTS", "CAUSES", 
              "COEXISTS_WITH", "COMPARED_WITH", "DIAGNOSES", "DISRUPTS", "INHIBITS", 
              "INTERACTS_WITH", "ISA", "LOCATION_OF", "None", "PART_OF", "PRECEDES", 
              "PREDISPOSES", "PREVENTS", "PROCESS_OF", "PRODUCES", "STIMULATES", "TREATS", "USES"]

# 5. IMPROVED UTILITIES
def format_prompt(row):
    """
    Improved prompt: Provides clear options and constraints to the model.
    """
    labels_str = ", ".join(ALL_LABELS)
    messages = [
        {
            "role": "system", 
            "content": f"You are a specialized biomedical natural language processor. Your task is to classify the relationship between a Subject and an Object based on the provided sentence. \n\nAllowed Labels: {labels_str}\n\nConstraint: Output ONLY the exact label name. If no specific relationship exists, output 'None'."
        },
        {
            "role": "user", 
            "content": f"Sentence: {row['sentence']}\nSubject: {row['subject_text']} ({row['subject_type']})\nObject: {row['object_text']} ({row['object_type']})\n\nRelationship Label:"
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def map_response_to_label(generated_text, label_list):
    """
    Better mapping logic: 
    1. Checks for exact matches first.
    2. Filters out 'None' as a substring check to avoid it overriding others.
    """
    gen_clean = generated_text.strip().upper()
    
    # 1. Try Exact Match
    for l in label_list:
        if gen_clean == l.upper():
            return l
            
    # 2. Try Substring Match (Prioritize non-None labels)
    for l in label_list:
        if l != "None" and l.lower() in generated_text.lower():
            return l
            
    return "None"

def load_test_data(path):
    # increased nrows or removed for full eval
    df = pd.read_csv(path, sep="\t", header=None, keep_default_na=False, nrows=100)
    df.columns = ["idx", "label", "sentence", "subject_text", "object_text", 
                  "subject_type", "object_type", "subject_group", "object_group", "label_choices"]
    return df

# 6. UPDATED INFERENCE ENGINE
def run_test_and_save(df, output_csv="final_test_results.csv"):
    predictions = []
    raw_responses = []
    
    print(f"Running inference on {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt_text = format_prompt(row)
        inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
        
        # Increased max_new_tokens slightly just in case model is chatty
        outputs = model.generate(
            **inputs, 
            max_new_tokens=10, 
            use_cache=True,
            temperature=0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        raw_responses.append(generated_text)
        
        # Use the new mapping logic
        final_pred = map_response_to_label(generated_text, ALL_LABELS)
        predictions.append(final_pred)

    df['model_raw_response'] = raw_responses
    df['predicted_label'] = predictions
    df.to_csv(output_csv, index=False, sep="\t")
    return predictions

# --- EXECUTION ---
test_df = load_test_data(TEST_DATA_PATH)
preds = run_test_and_save(test_df)

print("\n" + "="*30)
print("TEST SET EVALUATION")
print("="*30)
# Use zero_division=0 to clean up the printed report
print(classification_report(test_df['label'], preds, zero_division=0))