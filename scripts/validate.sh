#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo " Validating Mistral Pixtral 12B Serving Runtime   "
echo "=================================================="

python3 -m py_compile pixtral_serving_runtime.py
python3 -m py_compile dynamic_patch_tokenizer.py

python3 pixtral_serving_runtime.py
python3 dynamic_patch_tokenizer.py

echo "Validation successful!"
