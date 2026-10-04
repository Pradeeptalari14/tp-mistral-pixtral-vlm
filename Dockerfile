FROM vllm/vllm-openai:v0.6.3.post1

LABEL maintainer="Pradeep Talari <pradeep@pradeeptalari.com>"
LABEL description="Mistral Pixtral 12B Vision-Language Multimodal Serving Runtime with Dynamic Patching & FP8 Quantization"

ENV MODEL_NAME="mistralai/Pixtral-12B-2409"
ENV QUANTIZATION="fp8"
ENV MAX_MODEL_LEN="128000"
ENV GPU_MEMORY_UTILIZATION="0.92"
ENV PORT="8000"

WORKDIR /app

COPY pixtral_serving_runtime.py /app/
COPY dynamic_patch_tokenizer.py /app/

EXPOSE 8000

CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "mistralai/Pixtral-12B-2409", \
     "--quantization", "fp8", \
     "--max-model-len", "128000", \
     "--tensor-parallel-size", "1", \
     "--gpu-memory-utilization", "0.92", \
     "--port", "8000", \
     "--host", "0.0.0.0"]
