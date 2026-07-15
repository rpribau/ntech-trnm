#!/usr/bin/env bash
# ============================================================================
# Despliega vLLM (Qwen3-Coder-30B-A3B) en Cloud Run con GPU L4, scale-to-zero.
# Requisitos: gcloud CLI autenticado, proyecto con facturación y cuota de
# "Total Nvidia L4 GPUs" en la región elegida (solicítala en IAM & Admin > Quotas).
# ============================================================================
set -euo pipefail

# ---- Config (edita o exporta antes de correr) ----
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"                 # región con soporte de GPU en Cloud Run
SERVICE="${SERVICE:-ntech-vllm}"
IMAGE="${IMAGE:-${REGION}-docker.pkg.dev/${PROJECT_ID}/ntech/vllm:latest}"
REPO="${REPO:-ntech}"                            # Artifact Registry repo
HF_BUCKET="${HF_BUCKET:-${PROJECT_ID}-ntech-hf-cache}"  # caché de pesos HF (persistente)
INVOKER_SA="${INVOKER_SA:-ntech-invoker}"        # SA que el driver local usa para invocar

MODEL_ID="${MODEL_ID:-stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"

echo ">> Proyecto: ${PROJECT_ID} | Región: ${REGION} | Servicio: ${SERVICE}"

# ---- 1. Habilitar APIs ----
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com storage.googleapis.com --project "${PROJECT_ID}"

# ---- 2. Artifact Registry + bucket de caché ----
gcloud artifacts repositories describe "${REPO}" --location "${REGION}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "${REPO}" --repository-format=docker \
    --location "${REGION}" --project "${PROJECT_ID}"

gcloud storage buckets describe "gs://${HF_BUCKET}" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${HF_BUCKET}" --location "${REGION}" \
    --project "${PROJECT_ID}"

# ---- 3. Build de la imagen (Cloud Build) ----
echo ">> Construyendo imagen ${IMAGE} ..."
gcloud builds submit "$(dirname "$0")/vllm" --tag "${IMAGE}" --project "${PROJECT_ID}"

# ---- 4. Deploy en Cloud Run (GPU L4, scale-to-zero) ----
# La caché de HF se monta desde GCS: la 1a arrancada descarga los pesos y las
# siguientes los leen del bucket (cold-starts más rápidos).
echo ">> Desplegando en Cloud Run ..."
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --no-allow-unauthenticated \
  --gpu 1 --gpu-type nvidia-l4 \
  --cpu 8 --memory 32Gi \
  --no-cpu-throttling \
  --execution-environment gen2 \
  --min-instances 0 --max-instances 1 \
  --concurrency 8 \
  --timeout 3600 \
  --startup-probe "tcpSocket.port=8080,initialDelaySeconds=0,periodSeconds=15,failureThreshold=160,timeoutSeconds=10" \
  --add-volume "name=hf,type=cloud-storage,bucket=${HF_BUCKET}" \
  --add-volume-mount "volume=hf,mount-path=/root/.cache/huggingface" \
  --set-env-vars "MODEL_ID=${MODEL_ID},SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"

# ---- 5. Service account de invocación (para el driver local) ----
gcloud iam service-accounts describe \
  "${INVOKER_SA}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --project "${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${INVOKER_SA}" \
    --display-name "NTech Cloud Run invoker" --project "${PROJECT_ID}"

gcloud run services add-iam-policy-binding "${SERVICE}" \
  --region "${REGION}" --project "${PROJECT_ID}" \
  --member "serviceAccount:${INVOKER_SA}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role "roles/run.invoker"

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" \
  --project "${PROJECT_ID}" --format 'value(status.url)')"

echo ""
echo "============================================================"
echo " Despliegue listo."
echo " URL del servicio: ${URL}"
echo " Ponla en tu .env:  NTECH_CLOUDRUN_URL=${URL}"
echo ""
echo " Para la SA de invocación, crea una key y apúntala en .env:"
echo "   gcloud iam service-accounts keys create ntech-invoker-sa.json \\"
echo "     --iam-account ${INVOKER_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
echo "   GOOGLE_APPLICATION_CREDENTIALS=<ruta a ntech-invoker-sa.json>"
echo "============================================================"
