# Python Docstring Generator - LLM Fine-tuning Project

An AI-powered tool that automatically generates professional Google-style docstrings for Python functions using fine-tuned LLMs.

## Project Overview

This project demonstrates **fine-tuning a small language model (Phi-3 3.8B) using LoRA (Low-Rank Adaptation)** to automatically generate high-quality docstrings for undocumented Python code. The model learns to understand Python function syntax and generate appropriate documentation following Google docstring conventions.

### Why This Project?

- **Practical Use Case**: Saves developers hours of manual documentation work
- **Code Quality**: Helps maintain documentation standards in codebases
- **Technical Skills**: Demonstrates LLM fine-tuning, LoRA, and real-world NLP application
- **Efficient**: Completes in 20-30 minutes with minimal GPU resources

## Features

- Fine-tunes small code models (1.5B-7B parameters) using LoRA
- Generates Google-style docstrings with Args, Returns, and descriptions
- Optimized for RTX 5090 (works on other GPUs too)
- Training dataset of 30 Python function examples included
- Interactive testing mode
- Fast training (~10-20 minutes)

## Results Comparison

### Before Fine-tuning (Raw Input)
```python
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)
```

### After Fine-tuning (Model Output)
```python
def calculate_average(numbers):
    """
    Calculate the arithmetic mean of a list of numbers.
    
    Args:
        numbers (list): List of numeric values
        
    Returns:
        float: The average of all numbers in the list
    """
    total = sum(numbers)
    return total / len(numbers)
```

### Another Example

**Input:**
```python
def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
```

**Output:**
```python
def validate_email(email):
    """
    Validate if a string is a properly formatted email address.
    
    Args:
        email (str): The email string to validate
        
    Returns:
        bool: True if email format is valid, False otherwise
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
```

## Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (tested on RTX 5090)
- 8GB+ GPU VRAM

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/python-docstring-generator.git
cd python-docstring-generator
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify GPU setup:**
```bash
python diagnostic.py
```

## Usage

### 1. Fine-tune the Model

Run the training script (takes ~10-20 minutes):

```bash
python fine_tune_phi3_final.py --max_steps 300
```

**Optional arguments:**
```bash
python fine_tune_phi3_final.py --output_dir ./my-model --max_steps 500 --dataset dataset.json
```

**Training output example:**
```
CHECKING GPU SETUP
GPU: NVIDIA GeForce RTX 5090
Memory: 34.0 GB
GPU ready for training

LOADING MODEL: microsoft/phi-3-mini-4k-instruct
Model loaded successfully

SETTING UP LORA
trainable params: 25,165,824 || all params: 3,846,245,376 || trainable%: 0.65%

STARTING FINE-TUNING
[Training progress with loss decreasing from 0.76 to 0.03]

TRAINING COMPLETE
Model saved to ./docstring-model-phi3
```

### 2. Test the Model

#### Run Batch Tests
```bash
python test_model.py
```

This will test the model on 5 predefined functions and show before/after comparisons.

#### Interactive Mode
```bash
python test_model.py --interactive
```

Paste your Python functions interactively to generate docstrings in real-time.

**Example interactive session:**
```
Interactive Mode - Enter your Python function (type 'quit' to exit)
Enter function code:
def find_max(numbers):
    return max(numbers)

Generating docstring...

Result:
def find_max(numbers):
    """
    Find the maximum value in a list.
    
    Args:
        numbers (list): List of numeric values
        
    Returns:
        float: The maximum value in the list
    """
    return max(numbers)
```

## Project Structure

```
python-docstring-generator/
│
├── fine_tune_phi3_final.py    # Main training script
├── test_model.py              # Inference and testing script
├── dataset.json               # Training dataset (30 examples)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── diagnostic.py              # System check script
│
└── docstring-model-phi3/      # Output directory (created after training)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files...
```

## Technical Details

### Model Architecture
- **Base Model**: Microsoft Phi-3-mini-4k-instruct (3.8B parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Config**:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: qkv_proj, o_proj, gate_up_proj, down_proj
  - Dropout: 0.1

### Training Configuration
- **Precision**: bfloat16 (optimal for modern GPUs)
- **Batch size**: 2 (per device)
- **Gradient accumulation**: 4 steps
- **Effective batch size**: 8
- **Learning rate**: 2e-4
- **Max steps**: 300
- **Warmup steps**: 10
- **Training time**: ~10-20 minutes on RTX 5090

### Dataset
- **Size**: 30 Python function examples
- **Format**: JSON with input/output pairs
- **Coverage**: Algorithms, data structures, file operations, utilities
- **Docstring style**: Google-style format

## What This Demonstrates

1. **LLM Fine-tuning**: Practical application of transfer learning
2. **LoRA Efficiency**: Training with <1% of original parameters
3. **Code Understanding**: Model learns Python syntax and documentation patterns
4. **Production Skills**: Complete ML pipeline from data to deployment

## Customization

### Use Different Models

Modify the model name in `fine_tune_phi3_final.py`:
```python
model_name = "microsoft/phi-3-mini-4k-instruct"  # 3.8B parameters (default)
```

### Extend the Dataset

Add more examples to `dataset.json`:
```json
{
    "input": "def your_function():\n    pass",
    "output": "def your_function():\n    \"\"\"\n    Your docstring here.\n    \"\"\"\n    pass"
}
```

### Adjust Training Duration

For better results, increase training steps:
```bash
python fine_tune_phi3_final.py --max_steps 500
```

## Performance Metrics

- **Training Loss**: Decreases from ~0.76 to ~0.03
- **Generation Quality**: Produces valid Python with proper docstring format
- **Inference Speed**: ~1-2 seconds per function on RTX 5090
- **Memory Usage**: ~4-6GB VRAM during inference

## Limitations

- Works best with functions <50 lines of code
- Requires GPU for training (CPU inference possible but slow)
- Docstring quality depends on function complexity
- May occasionally generate incomplete docstrings for very complex functions

## Future Improvements

- Add support for class docstrings
- Include type hints in generated docstrings
- Train on larger dataset (1000+ examples)
- Add evaluation metrics (BLEU, ROUGE)
- Support multiple docstring styles (NumPy, Sphinx)
- Create web demo with Gradio/Streamlit

## License

MIT License - feel free to use this project for learning and portfolio purposes.

## Contributing

Pull requests welcome! Ideas for improvement:
- Better dataset examples
- Support for more programming languages
- Evaluation benchmarks
- Web interface

## Contact

For questions or suggestions, please open an issue on GitHub.

---

## Resume Talking Points

When discussing this project:
1. "Fine-tuned a 3.8B parameter model using LoRA with only 0.65% trainable parameters"
2. "Reduced documentation time by automatically generating professional docstrings"
3. "Optimized training pipeline to complete in 20 minutes on RTX 5090"
4. "Demonstrated practical application of transfer learning for code understanding"
5. "Built end-to-end ML pipeline including data preparation, training, and inference"

## Acknowledgments

- Built using HuggingFace Transformers
- Uses Microsoft's Phi-3 model
- LoRA implementation from PEFT library

## Training Output

### GPU Setup
```
CUDA Available: ✓
GPU: NVIDIA GeForce RTX 5090
Memory: 34.2 GB
Status: Ready for training
```

### Model Configuration
```
Base Model: microsoft/phi-3-mini-4k-instruct
Parameters: 3.8B
Vocab Size: 32,011
Device: CUDA
```

### LoRA Configuration
```
Trainable Parameters: 25,165,824
Total Parameters: 3,846,245,376
Trainable Percentage: 0.65%
Target Modules: q_proj, v_proj
Rank: 8
Alpha: 16
```

### Dataset
```
Training Examples: 31
Format: JSON with function-docstring pairs
Tokenization: Complete
```

### Training Configuration
```
Batch Size: 2
Gradient Accumulation Steps: 4
Effective Batch Size: 8
Learning Rate: 0.0002
Max Steps: 300
Training Time: ~6 minutes
```

### Training Progress

| Step | Loss  | Epoch | Learning Rate |
|------|-------|-------|---------------|
| 0    | 0.760 | 0.25  | 0.0          |
| 20   | 0.647 | 1.25  | 0.00008      |
| 40   | 0.411 | 2.50  | 0.00018      |
| 60   | 0.162 | 3.75  | 0.000197     |
| 80   | 0.111 | 5.00  | 0.000194     |
| 100  | 0.074 | 6.25  | 0.000190     |
| 200  | 0.023 | 51.25 | 0.000066     |
| 300  | 0.023 | 75.00 | 0.000001     |

**Loss Reduction: 0.760 → 0.023 (97% improvement)**

### Training Results
```
Runtime: 351 seconds (~6 minutes)
Samples/Second: 6.84
Steps/Second: 0.86
Final Training Loss: 0.046
Model Saved: ./docstring-model-phi3
```

### Test Output

**Input Function:**
```python
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
```

**Generated Docstring:**
```python
def calculate_sum(numbers):
    """
    Calculate the sum of all numbers in a list.

    Args:
        numbers (list): List of numeric values

    Returns:
        int/float: The sum of all numbers in the list
    """
    total = 0
    for num in numbers:
        total += num
    return total
```

### Testing the Model
```bash
python test_model.py --model_dir ./docstring-model-phi3
```

---

**Status:** ✓ Training Complete | Model Successfully Generates Google-Style Docstrings