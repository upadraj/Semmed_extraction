"""
train.py
Training script for fine-tuning the model with LoRA.
"""

import argparse
import torch
from trl import SFTTrainer, SFTConfig
from DataModule import (
    # Configuration
    BASE_MODEL,
    MAX_SEQ_LENGTH,
    DTYPE,
    LOAD_IN_4BIT,
    TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    WARMUP_STEPS,
    MAX_STEPS,
    LEARNING_RATE,
    BF16,
    LOGGING_STEPS,
    OUTPUT_DIR,
    OPTIM,
    RANDOM_STATE,
    DATA_TRAIN_PATH,
    DATA_VAL_PATH,
    LORA_MODEL_PATH,
    MERGED_MODEL_PATH,
    USE_FEW_SHOT,
    FEW_SHOT_EXAMPLES,
    # Functions
    load_model_and_tokenizer,
    setup_lora_model,
    prepare_datasets,
)


def main():
    """
    Main training function.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train the model with LoRA fine-tuning')
    parser.add_argument('--nrows', type=int, default=None,
                       help='Number of rows to load from each data file (default: all)')
    args = parser.parse_args()

    print("=" * 80)
    print("FINE-TUNING TRAINING SCRIPT")
    print("=" * 80)

    # Display configuration
    if args.nrows is not None:
        print(f"\n🔧 Data limit: Loading first {args.nrows} rows from each file")
    else:
        print("\n🔧 Data limit: Loading all rows")

    if USE_FEW_SHOT:
        print(f"⚙️  Few-shot learning: ENABLED ({len(FEW_SHOT_EXAMPLES)} examples)")
    else:
        print("⚙️  Few-shot learning: DISABLED")

    # Step 1: Load Model and Tokenizer
    print("\n[1/6] Loading base model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT
    )
    print(f"✓ Loaded model: {BASE_MODEL}")

    # Step 2: Apply LoRA
    print("\n[2/6] Applying LoRA configuration...")
    model = setup_lora_model(model)
    print("✓ LoRA applied successfully")

    # Step 3: Prepare Datasets
    print("\n[3/6] Preparing training and validation datasets...")
    train_df, val_df, train_ds, val_ds = prepare_datasets(
        DATA_TRAIN_PATH,
        DATA_VAL_PATH,
        tokenizer,
        use_few_shot=USE_FEW_SHOT,
        num_examples=len(FEW_SHOT_EXAMPLES),
        nrows=args.nrows
    )
    print(f"✓ Training samples: {len(train_df)}")
    print(f"✓ Validation samples: {len(val_df)}")
    
    # Step 4: Configure Trainer
    print("\n[4/6] Configuring trainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # Updated from 'tokenizer' parameter
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            dataset_text_field="text",  # Moved here from SFTTrainer
            max_seq_length=MAX_SEQ_LENGTH,  # Moved here from SFTTrainer
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            bf16=BF16,
            logging_steps=LOGGING_STEPS,
            output_dir=OUTPUT_DIR,
            optim=OPTIM,
            seed=RANDOM_STATE,
        ),
    )
    print("✓ Trainer configured")
    
    # Step 5: Train
    print("\n[5/6] Starting training...")
    print("-" * 80)
    trainer.train()
    print("-" * 80)
    print("✓ Training completed")
    
    # Step 6: Save Models
    print("\n[6/6] Saving model components...")
    
    # Save LoRA adapters
    print(f"  - Saving LoRA adapters to: {LORA_MODEL_PATH}")
    model.save_pretrained(LORA_MODEL_PATH)
    tokenizer.save_pretrained(LORA_MODEL_PATH)
    print(f"  ✓ LoRA adapters saved")
    
    # Save merged model
    print(f"  - Saving merged model to: {MERGED_MODEL_PATH}")
    model.save_pretrained_merged(
        MERGED_MODEL_PATH,
        tokenizer,
        save_method="merged_16bit"
    )
    print(f"  ✓ Merged model saved")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel artifacts saved:")
    print(f"  - LoRA adapters: {LORA_MODEL_PATH}/")
    print(f"  - Merged model: {MERGED_MODEL_PATH}/")
    print(f"  - Training outputs: {OUTPUT_DIR}/")
    print("\nYou can now run test.py to evaluate the model.")


if __name__ == "__main__":
    main()

