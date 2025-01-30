# FP8 Refactor Verification

A test that verify the correctness and performance of implementations across original(Deepseek Original files) and refactored(files i refactored).

## Requirements

- NVIDIA GPU with compute capability 7.0 (Volta) or higher
- CUDA toolkit installed
- Python 3.7+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fp8-refactor-verification.git
cd fp8-refactor-verification
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
fp8-refactor-verification/
├── requirements.txt
├── generate.py                    # Original text generation implementation
├── refactored_generate.py        # Refactored text generation implementation
├── fp8_cast_bf16.py             # Original FP8 operations implementation
├── refactored_fp8_cast_bf16.py  # Refactored FP8 operations implementation
├── kernel.py             # Original Kernel operations implementation
├── refactored_kernel.py  # Refactored Kernel operations implementation
├── verification_tests.py         #
└── README.md
```

The script will output:
- ✅/❌ Output match indicators
- Performance impact percentages
- Memory usage differences
- Numerical precision metrics (for FP8 operations)

## Requirements Details

```
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5
numpy>=1.24.0
pytest>=7.0.0
typing-extensions>=4.0.0
cuda-python>=12.0.0
```

## Hardware Requirements

- NVIDIA GPU (Volta/V100 or newer)
- Minimum 8GB GPU memory recommended
- CUDA toolkit compatible with PyTorch 2.4.1

## Note

This verification framework is designed to run on NVIDIA GPUs only. CPU execution is not supported due to CUDA-specific operations and FP8 requirements.
