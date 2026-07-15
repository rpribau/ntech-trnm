# Serving del LLM en GCP (Cloud Run GPU L4 + vLLM)

Sirve **Qwen3-Coder-30B-A3B** con vLLM en Cloud Run con GPU **NVIDIA L4 (24 GB)**
y **scale-to-zero**: pagas la GPU (~$0.67/hr) solo mientras infiere.

## Pasos

1. **Cuota de GPU**: en la consola de GCP, *IAM & Admin → Quotas*, solicita
   `Total Nvidia L4 GPUs` ≥ 1 en tu región (p. ej. `us-central1`). Puede tardar.
2. **Autenticar** `gcloud` y fijar el proyecto:
   ```bash
   gcloud auth login
   gcloud config set project TU_PROYECTO
   ```
3. **Desplegar**:
   ```bash
   cd deploy && bash deploy.sh
   ```
   El script habilita APIs, crea Artifact Registry + bucket de caché, construye la
   imagen, despliega el servicio (privado) y crea la SA de invocación.
4. **Configurar el driver** (`.env` en la raíz):
   ```
   NTECH_CLOUDRUN_URL=<URL que imprime el script>
   GOOGLE_APPLICATION_CREDENTIALS=<ruta a ntech-invoker-sa.json>
   ```

## Seguridad

- El servicio se despliega con `--no-allow-unauthenticated` (privado). Solo
  identidades con `roles/run.invoker` pueden llamarlo.
- El driver local mintea un **ID token** con la service account `ntech-invoker` y
  lo envía en `Authorization: Bearer <token>` (lo hace `ntech_agent/llm.py`).
- **No** expongas el endpoint públicamente ni subas el JSON de la SA a git
  (está en `.gitignore`).

## Modelo y cuantización

- Default: `stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ` (4-bit vía llm-compressor
  de vLLM, ~15-17 GB de pesos, entra cómodo en 24 GB con margen para el KV-cache).
- **Ojo con FP8**: aunque Qwen publica `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`
  oficial, es MoE con **30B de parámetros totales** (no solo los ~3B activos por
  token) → en FP8 (1 byte/parámetro) son ~30 GB de pesos, **no cabe** en una L4
  de 24 GB. Por eso usamos AWQ (4-bit, ~15-17 GB) en vez de FP8.
- Alternativa AWQ si `stelterlab` no estuviera disponible:
  `nm-testing/Qwen3-Coder-30B-A3B-Instruct-W4A16-awq` (Neural Magic).
- Si migras a una GPU con más VRAM (p. ej. A100/H100 40GB+) o al modelo denso
  14B, cambia `MODEL_ID`, `QUANTIZATION` y `SERVED_MODEL_NAME` (env vars del
  contenedor / `deploy.sh`).
- `MAX_MODEL_LEN=32768` acota el KV-cache para caber en la L4; súbelo si hay margen.
- **Verifica siempre que el repo de Hugging Face exista y esté escrito
  exactamente igual** (`huggingface.co/<owner>/<repo>`) antes de desplegar — un
  nombre inventado o mal escrito falla con `401/Repository Not Found`, no con
  un error de permisos.
- **Tool calling**: la imagen arranca vLLM con `--enable-auto-tool-choice
  --tool-call-parser hermes` para que funcionen los *structured outputs*
  (routing, revisión, rerank). Si tu versión de vLLM tiene un parser específico
  `qwen3_coder`, úsalo en su lugar. El agente igual tiene *fallbacks* si falla.

## Cold starts

La primera petición tras escalar a cero arranca el contenedor y carga el modelo
(varios minutos la 1ª vez; luego lee los pesos desde el bucket GCS montado).
Para sesiones de trabajo intensas, sube `min-instances` a 1 temporalmente:

```bash
gcloud run services update ntech-vllm --region us-central1 --min-instances 1
# ... al terminar ...
gcloud run services update ntech-vllm --region us-central1 --min-instances 0
```

## Troubleshooting: "container failed to start and listen on the port"

Casi siempre es el **startup probe** por defecto de Cloud Run, cuya ventana es
demasiado corta para que vLLM descargue ~15-20 GB de pesos (AWQ) desde Hugging
Face y los cargue a la GPU en el primer arranque. `deploy.sh` ya incluye
`--startup-probe` con una ventana de ~40 min (`failureThreshold=160,
periodSeconds=15`); si desplegaste manualmente sin ese flag, aplícalo así:

```bash
gcloud run services update ntech-vllm --region "$REGION" \
  --startup-probe "tcpSocket.port=8080,initialDelaySeconds=0,periodSeconds=15,failureThreshold=160,timeoutSeconds=10"
```

Antes de asumir que es solo lentitud, revisa los logs por si es un error real
(memoria insuficiente, modelo gated, flag no soportado):

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ntech-vllm" \
  --project "$PROJECT_ID" --limit 100 --format "value(textPayload)"
```

Los arranques posteriores son más rápidos porque el bucket GCS ya tiene los
pesos cacheados.

## Troubleshooting: "The NVIDIA driver on your system is too old (found version 12020)"

Los nodos GPU L4 de Cloud Run corren un driver fijo **535.x (CUDA 12.2)** —no
lo puedes actualizar tú. Desde vLLM **v0.20.0**, la imagen `vllm/vllm-openai:latest`
se compila contra **CUDA 13**, que exige driver ≥580.x → choque de *major
version*, no arreglable con más tiempo de espera ni variables de entorno.
El [Dockerfile](vllm/Dockerfile) ya fija la base a `vllm/vllm-openai:v0.19.1-x86_64`
(CUDA 12.9, compatible con el driver 535.x por *minor-version compatibility*
dentro de la familia CUDA 12.x). Si tocas el Dockerfile, **requiere rebuild**:

```bash
gcloud builds submit deploy/vllm --tag us-central1-docker.pkg.dev/PROJECT_ID/ntech/vllm:latest
gcloud run deploy ntech-vllm --image us-central1-docker.pkg.dev/PROJECT_ID/ntech/vllm:latest --region us-central1 [...resto de flags del paso 6]
```

## Troubleshooting: "Quantization method ... does not match ..."

Cada checkpoint cuantizado declara su método en su propio `config.json`
(`awq`, `awq_marlin`, `gptq`, `compressed-tensors`, `fp8`, etc.). Si forzabas
`--quantization` con un valor que no coincide con el checkpoint, vLLM rechaza
arrancar. Desde esta versión, `QUANTIZATION` viene **vacío por defecto** y
vLLM autodetecta el método correcto leyendo el checkpoint — no debería volver
a pasar al cambiar de `MODEL_ID`. Si necesitas forzarlo, fija `QUANTIZATION`
al valor exacto que use ese checkpoint (revisa su `config.json` en Hugging
Face, campo `quantization_config.quant_method`).

## Probar el endpoint

```bash
TOKEN=$(gcloud auth print-identity-token)
curl -H "Authorization: Bearer $TOKEN" "$NTECH_CLOUDRUN_URL/v1/models"
```
