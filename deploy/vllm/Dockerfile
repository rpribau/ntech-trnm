# Imagen de vLLM para servir Qwen3-Coder-30B-A3B en Cloud Run (GPU L4).
# Base: imagen oficial de vLLM con servidor compatible con la API de OpenAI.
#
# Fijada a v0.19.1 (CUDA 12.9) a propósito: desde v0.20.0 la imagen ":latest"
# se compila contra CUDA 13, que requiere driver NVIDIA >=580.x. Los nodos GPU
# L4 de Cloud Run corren driver 535.x (CUDA 12.2) y Google no permite
# cambiarlo. CUDA 12.9 sí es compatible con un driver 12.2 (minor-version
# compatibility dentro de la misma familia CUDA 12.x); el salto de major
# version 12->13 no lo es y falla con "NVIDIA driver ... too old".
FROM vllm/vllm-openai:v0.19.1-x86_64

# El modelo y sus parámetros se inyectan por variables de entorno (ver deploy.sh).
# MODEL_ID puede ser un repo de Hugging Face (se descarga a la caché montada en GCS)
# o una ruta local montada.
ENV MODEL_ID="stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ" \
    QUANTIZATION="" \
    MAX_MODEL_LEN="32768" \
    GPU_MEMORY_UTILIZATION="0.92" \
    SERVED_MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    HF_HOME="/root/.cache/huggingface"

# Cloud Run inyecta $PORT (8080 por defecto). vLLM debe escuchar ahí.
# Se usa un entrypoint de shell para expandir las variables en tiempo de arranque.
# QUANTIZATION vacío (default) = vLLM autodetecta el método leyendo el config.json
# del checkpoint (evita mismatches como "compressed-tensors" vs "awq_marlin" al
# cambiar de modelo). Solo se pasa --quantization si se fuerza explícitamente.
ENTRYPOINT []
CMD exec python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port "${PORT:-8080}" \
    --model "${MODEL_ID}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    ${QUANTIZATION:+--quantization "${QUANTIZATION}"} \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
