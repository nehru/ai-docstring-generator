#!/usr/bin/env python3
"""
Test the fine-tuned Python Docstring Generator model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

def load_model(model_dir="./docstring-model-phi3"):
    """Load the fine-tuned model"""
    print(f"Loading model from {model_dir}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        print("Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("Make sure you've run fine_tune_phi3_final.py first")
        return None, None

def generate_docstring(model, tokenizer, function_code):
    """Generate docstring for a given function"""
    
    prompt = f"### Task: Add a docstring to this Python function\n\n### Input:\n{function_code}\n\n### Output:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=250,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the output part
    if "### Output:" in result:
        return result.split("### Output:")[1].strip()
    else:
        return result

def main():
    parser = argparse.ArgumentParser(description="Test the fine-tuned docstring generator")
    parser.add_argument("--model_dir", type=str, default="./docstring-model-phi3",
                       help="Directory containing the fine-tuned model")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode - enter functions manually")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_dir)
    if model is None:
        return
    
    print("\n" + "="*80)
    print("PYTHON DOCSTRING GENERATOR - TEST MODE")
    print("="*80)
    
    # Test examples
    test_functions = [
        """def add_numbers(a, b):
    return a + b""",
        
        """def find_duplicates(lst):
    seen = set()
    duplicates = []
    for item in lst:
        if item in seen:
            duplicates.append(item)
        else:
            seen.add(item)
    return duplicates""",
        
        """def parse_csv(filename):
    import csv
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data""",
        
        """def validate_password(password):
    if len(password) < 8:
        return False
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    return has_upper and has_lower and has_digit""",
        
        """def merge_sorted_arrays(arr1, arr2):
    result = []
    i = j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result"""
    ]
    
    if args.interactive:
        print("\nInteractive Mode - Enter your Python function (type 'quit' to exit)")
        print("   (Paste your function and press Enter twice)\n")
        
        while True:
            print("-" * 80)
            print("Enter function code:")
            lines = []
            while True:
                try:
                    line = input()
                    if line.lower() == 'quit':
                        print("\nGoodbye")
                        return
                    if not line and lines:
                        break
                    if line:
                        lines.append(line)
                except EOFError:
                    break
            
            if not lines:
                continue
            
            function_code = '\n'.join(lines)
            
            print("\nGenerating docstring...\n")
            result = generate_docstring(model, tokenizer, function_code)
            print("Result:")
            print(result)
            print()
    
    else:
        # Run through test examples
        for i, func in enumerate(test_functions, 1):
            print(f"\nTest Example {i}/{len(test_functions)}")
            print("-" * 80)
            print("INPUT (Function without docstring):")
            print(func)
            print("\nGenerating docstring...")
            
            result = generate_docstring(model, tokenizer, func)
            
            print("\nOUTPUT (Function with generated docstring):")
            print(result)
            print("="*80)
        
        print("\nAll tests completed")
        print("\nTip: Run with --interactive flag to test your own functions")
        print("   Example: python test_model.py --interactive")

if __name__ == "__main__":
    main()
