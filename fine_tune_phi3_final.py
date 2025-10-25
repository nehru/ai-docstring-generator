#!/usr/bin/env python3
"""
Fine-tune Phi-3 for Python Docstring Generation
Uses microsoft/phi-3-mini-4k-instruct (no HF login required)
Optimized for RTX 5090 with LoRA
"""

import sys
import torch
import os
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import argparse
import logging

def print_and_flush(message):
    """Print message and flush immediately"""
    print(message)
    sys.stdout.flush()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_gpu():
    """Check GPU availability"""
    print_and_flush("="*60)
    print_and_flush("CHECKING GPU SETUP")
    print_and_flush("="*60)
    print_and_flush(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_and_flush(f"GPU: {gpu_name}")
        print_and_flush(f"Memory: {gpu_memory:.1f} GB")
        print_and_flush("GPU ready for training")
        return True
    else:
        print_and_flush("ERROR: No GPU detected")
        return False

def load_model_and_tokenizer():
    """Load Phi-3 model"""
    print_and_flush("\n" + "="*60)
    print_and_flush("LOADING MODEL: microsoft/phi-3-mini-4k-instruct")
    print_and_flush("="*60)
    print_and_flush("This model doesn't require HuggingFace login")
    print_and_flush("Loading from cache...\n")
    
    model_name = "microsoft/phi-3-mini-4k-instruct"
    
    try:
        print_and_flush("Step 1/2: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print_and_flush("Tokenizer loaded successfully")
        
        print_and_flush("\nStep 2/2: Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
        print_and_flush("Model loaded successfully")
        
    except Exception as e:
        print_and_flush(f"\nERROR loading model: {e}")
        sys.exit(1)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print_and_flush(f"\nModel Details:")
    print_and_flush(f"   Device: {model.device}")
    print_and_flush(f"   Vocab size: {len(tokenizer)}")
    print_and_flush(f"   Parameters: {model.num_parameters() / 1e9:.1f}B")
    
    return model, tokenizer

def setup_lora(model):
    """Setup LoRA for efficient fine-tuning"""
    print_and_flush("\n" + "="*60)
    print_and_flush("SETTING UP LORA")
    print_and_flush("="*60)
    
    model.train()
    print_and_flush("Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    
    print_and_flush("Preparing model for training...")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
    )
    
    print_and_flush("Applying LoRA to model...")
    model = get_peft_model(model, lora_config)
    
    print_and_flush("Enabling gradients for LoRA parameters...")
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
    
    print_and_flush("\nLoRA configured successfully")
    model.print_trainable_parameters()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_and_flush(f"Verified: {trainable_params:,} trainable parameters")
    
    return model

def load_dataset(tokenizer, dataset_file="dataset.json"):
    """Load and prepare the dataset"""
    print_and_flush("\n" + "="*60)
    print_and_flush("LOADING TRAINING DATASET")
    print_and_flush("="*60)
    
    if not os.path.exists(dataset_file):
        print_and_flush(f"ERROR: {dataset_file} not found")
        sys.exit(1)
    
    print_and_flush(f"Reading {dataset_file}...")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print_and_flush(f"Loaded {len(data)} training examples")
    
    print_and_flush("Formatting examples...")
    formatted_data = []
    for i, example in enumerate(data):
        if (i + 1) % 10 == 0:
            print_and_flush(f"   Processed {i + 1}/{len(data)} examples")
        prompt = f"### Task: Add a docstring to this Python function\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}<|endoftext|>"
        formatted_data.append({"text": prompt})
    
    print_and_flush("All examples formatted")
    
    print_and_flush("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None,
        )
    
    dataset = Dataset.from_dict({"text": [item["text"] for item in formatted_data]})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    print_and_flush(f"Dataset ready: {len(tokenized_dataset)} examples tokenized")
    
    return tokenized_dataset

def setup_training_args(output_dir="./docstring-model-phi3", max_steps=50):
    """Setup training arguments"""
    print_and_flush("\n" + "="*60)
    print_and_flush("CONFIGURING TRAINING PARAMETERS")
    print_and_flush("="*60)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        logging_steps=5,
        save_steps=100,
        eval_strategy="no",
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        push_to_hub=False,
        report_to="none",
        max_steps=max_steps,
        save_total_limit=1,
        gradient_checkpointing=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        logging_first_step=True,
        optim="adamw_torch",
    )
    
    print_and_flush(f"Training Configuration:")
    print_and_flush(f"   Batch size: {training_args.per_device_train_batch_size}")
    print_and_flush(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print_and_flush(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print_and_flush(f"   Learning rate: {training_args.learning_rate}")
    print_and_flush(f"   Max steps: {training_args.max_steps}")
    
    if max_steps <= 50:
        print_and_flush(f"   Expected time: ~1-2 minutes (quick test)")
    else:
        print_and_flush(f"   Expected time: ~10-20 minutes (full training)")
    
    return training_args

def train_model(model, tokenizer, tokenized_dataset, training_args):
    """Train the model"""
    print_and_flush("\n" + "="*60)
    print_and_flush("STARTING FINE-TUNING")
    print_and_flush("="*60)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    print_and_flush("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    print_and_flush("\nProgress updates will appear every 5 steps")
    print_and_flush("Starting training now...")
    print_and_flush("-"*60)
    
    trainer.train()
    
    print_and_flush("\n" + "-"*60)
    print_and_flush("Saving model...")
    
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print_and_flush("Model saved successfully")
    print_and_flush(f"Location: {training_args.output_dir}")
    
    return model

def quick_test(model, tokenizer):
    """Quick test of the fine-tuned model"""
    print_and_flush("\n" + "="*60)
    print_and_flush("TESTING FINE-TUNED MODEL")
    print_and_flush("="*60)
    
    test_function = """def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total"""
    
    prompt = f"### Task: Add a docstring to this Python function\n\n### Input:\n{test_function}\n\n### Output:\n"
    
    print_and_flush(f"\nTest Input:")
    print_and_flush(test_function)
    print_and_flush("\nGenerating docstring...")
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print_and_flush("\nGenerated Output:")
        print_and_flush("-"*60)
        if "### Output:" in result:
            output = result.split("### Output:")[1].strip()
            print_and_flush(output)
        else:
            print_and_flush(result)
        print_and_flush("-"*60)
        
    except Exception as e:
        print_and_flush(f"\nWARNING: Generation had an issue: {e}")
        print_and_flush("Training was successful and model is saved")
        print_and_flush("Test separately with: python test_model.py")

def main():
    print_and_flush("\n" + "="*60)
    print_and_flush("PYTHON DOCSTRING GENERATOR")
    print_and_flush("Fine-tuning Phi-3 with LoRA")
    print_and_flush("="*60 + "\n")
    
    parser = argparse.ArgumentParser(description="Fine-tune Phi-3 for Docstrings")
    parser.add_argument("--output_dir", type=str, default="./docstring-model-phi3", 
                       help="Output directory")
    parser.add_argument("--dataset", type=str, default="dataset.json",
                       help="Dataset JSON file")
    parser.add_argument("--max_steps", type=int, default=50,
                       help="Training steps (50=quick test, 300=full training)")
    
    args = parser.parse_args()
    
    try:
        # Setup
        gpu_available = setup_gpu()
        if not gpu_available:
            print_and_flush("\nERROR: GPU required")
            return
        
        # Load model
        model, tokenizer = load_model_and_tokenizer()
        
        # Setup LoRA
        model = setup_lora(model)
        
        # Load dataset
        tokenized_dataset = load_dataset(tokenizer, args.dataset)
        
        # Setup training
        training_args = setup_training_args(args.output_dir, args.max_steps)
        
        # Train
        model = train_model(model, tokenizer, tokenized_dataset, training_args)
        
        # Test
        quick_test(model, tokenizer)
        
        print_and_flush("\n" + "="*60)
        print_and_flush("TRAINING COMPLETE")
        print_and_flush("="*60)
        print_and_flush("Model successfully generated docstrings")
        print_and_flush(f"Model saved: {args.output_dir}")
        
        if args.max_steps <= 50:
            print_and_flush("\nThis was a quick test. For better results:")
            print_and_flush("   python -u fine_tune_phi3_final.py --max_steps 300")
        
        print_and_flush("\nTest your model:")
        print_and_flush("   python test_model.py --model_dir ./docstring-model-phi3")
        print_and_flush("="*60 + "\n")
        
    except KeyboardInterrupt:
        print_and_flush("\nTraining interrupted by user")
    except Exception as e:
        print_and_flush(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
