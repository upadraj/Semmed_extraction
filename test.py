"""
test.py
Testing and evaluation script for the fine-tuned model.
"""

import argparse
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from unsloth import FastLanguageModel
from DataModule import (
    # Configuration
    LORA_MODEL_PATH,
    MAX_SEQ_LENGTH,
    DTYPE,
    LOAD_IN_4BIT,
    DATA_TEST_PATH,
    ALL_LABELS,
    USE_FEW_SHOT,
    FEW_SHOT_EXAMPLES,
    load_and_prep,
    format_prompt,
)


def load_model_for_inference(model_path=LORA_MODEL_PATH):
    """
    Load the fine-tuned model for inference.
    
    Args:
        model_path: Path to the saved model (LoRA adapters)
        
    Returns:
        tuple: (model, tokenizer) ready for inference
    """
    print(f"Loading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # Set model to inference mode
    FastLanguageModel.for_inference(model)
    print("✓ Model loaded and set to inference mode")
    
    return model, tokenizer


def run_evaluation(model, tokenizer, df, output_csv="test_results_full.csv",
                   use_few_shot=USE_FEW_SHOT, few_shot_examples=None,
                   num_examples=None):
    """
    Run evaluation on a dataset and save detailed results.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        df: DataFrame containing test data
        output_csv: Path to save detailed results
        use_few_shot: Whether to include few-shot examples in prompts
        few_shot_examples: List of example dictionaries (uses FEW_SHOT_EXAMPLES if None)
        num_examples: Number of few-shot examples to include (uses all if None)

    Returns:
        list: Predicted labels
    """
    # Default to using all available examples
    if num_examples is None:
        num_examples = len(FEW_SHOT_EXAMPLES)

    predictions = []
    raw_responses = []

    print(f"\nPredicting on {len(df)} samples...")
    if use_few_shot:
        print(f"Using {num_examples} few-shot examples in prompts")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        # Format the prompt without the output (for inference)
        prompt_text = format_prompt(row, tokenizer, include_output=False,
                                   use_few_shot=use_few_shot,
                                   few_shot_examples=few_shot_examples,
                                   num_examples=num_examples)
        inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
        
        # Generate prediction with greedy decoding (temperature=0)
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            use_cache=True,
            temperature=0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Extract only the newly generated text
        raw_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        raw_responses.append(raw_text)
        
        # Match the generated text to one of the valid labels
        final_pred = "None"
        for label in ALL_LABELS:
            if label.lower() in raw_text.lower():
                final_pred = label
                break
        predictions.append(final_pred)
    
    # Attach results to the dataframe
    df['model_raw_response'] = raw_responses
    df['predicted_label'] = predictions
    
    # Save detailed results
    df.to_csv(output_csv, index=False)
    print(f"✓ Detailed results saved to: {output_csv}")
    
    return predictions


def print_performance_summary(true_labels, predicted_labels):
    """
    Print performance metrics.
    
    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(classification_report(true_labels, predicted_labels))
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print("=" * 80)


def main():
    """
    Main evaluation function.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate the fine-tuned model')
    parser.add_argument('--nrows', type=int, default=None,
                       help='Number of rows to load from test data (default: all)')
    args = parser.parse_args()

    print("=" * 80)
    print("MODEL EVALUATION SCRIPT")
    print("=" * 80)

    # Display configuration
    if args.nrows is not None:
        print(f"\n🔧 Data limit: Loading first {args.nrows} rows")
    else:
        print("\n🔧 Data limit: Loading all rows")

    if USE_FEW_SHOT:
        print(f"⚙️  Few-shot learning: ENABLED ({len(FEW_SHOT_EXAMPLES)} examples)")
    else:
        print("⚙️  Few-shot learning: DISABLED")

    # Step 1: Load Model
    print("\n[1/4] Loading fine-tuned model...")
    model, tokenizer = load_model_for_inference(LORA_MODEL_PATH)

    # Step 2: Load Test Data
    print("\n[2/4] Loading test data...")
    test_df = load_and_prep(DATA_TEST_PATH, tokenizer,
                           use_few_shot=USE_FEW_SHOT,
                           num_examples=len(FEW_SHOT_EXAMPLES),
                           nrows=args.nrows)
    print(f"✓ Test samples: {len(test_df)}")

    # Step 3: Run Evaluation
    print("\n[3/4] Running evaluation...")
    predictions = run_evaluation(model, tokenizer, test_df,
                                use_few_shot=USE_FEW_SHOT,
                                num_examples=len(FEW_SHOT_EXAMPLES))

    # Step 4: Print Results
    print("\n[4/4] Generating performance report...")
    print_performance_summary(test_df['label'], predictions)

    print("\n✓ Evaluation complete!")
    print("  Detailed results saved to: test_results_full.csv")


if __name__ == "__main__":
    main()

