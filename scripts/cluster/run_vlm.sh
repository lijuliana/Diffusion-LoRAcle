#!/usr/bin/env bash
# Launch the VLM stages (Qwen2.5-VL benign-drafting + sensitive triage) on a Forge Beta p5 node.
# Follows WORKING_NORMS: confirm account, check FREE p5 capacity (don't evict), anonymized names,
# FSx for state, delete the pod when done. Secrets stay in env, never in the YAML/logs/repo.
#
# Prereqs (laptop): notes/.env sourced (HF_TOKEN, CIVITAI_API_KEY); ada creds fresh; kubeconfig set.
# Usage:  bash scripts/cluster/run_vlm.sh <you>      e.g.  bash scripts/cluster/run_vlm.sh juliaili
set -euo pipefail

YOU="${1:?usage: run_vlm.sh <you> (e.g. juliaili)}"
AZ="us-east-1b"
POD="${YOU}-devpod-${AZ}"
FSX_ROOT="/fsx/${YOU}/oss-vlm"            # one tree → one-line cleanup
REPO_REMOTE="https://github.com/lijuliana/Diffusion-LoRAcle.git"

echo "== 0. confirm account (must be 947810121487) =="
aws sts get-caller-identity --query Account --output text | grep -q 947810121487 || {
  echo "WRONG ACCOUNT — run: ada credentials update --provider=conduit --account=947810121487 --once \\
    --role=open-beta-nf-open-beta-role-open-beta-us-east-1"; exit 1; }

echo "== 1. free p5 capacity (be a good neighbour; need >=1 GPU) =="
kubectl get nodes -o json | python3 -c '
import sys,json,subprocess
ns=json.load(sys.stdin)["items"]
alloc={n["metadata"]["name"]:(int(n["status"]["allocatable"].get("nvidia.com/gpu",0)),n["metadata"]["labels"].get("node.kubernetes.io/instance-type","?")) for n in ns}
pods=json.loads(subprocess.run(["kubectl","get","pods","-A","--field-selector=status.phase=Running","-o","json"],capture_output=True,text=True).stdout)
used={}
for p in pods["items"]:
    nn=p["spec"].get("nodeName")
    for c in p["spec"]["containers"]:
        g=c.get("resources",{}).get("requests",{}).get("nvidia.com/gpu")
        if g and nn: used[nn]=used.get(nn,0)+int(g)
free=sum(max(0,a-used.get(n,0)) for n,(a,t) in alloc.items() if "p5" in t)
print(f"free p5 GPUs: {free}")
sys.exit(0 if free>=1 else 1)' || { echo "no free p5 GPU — wait, do not evict."; exit 1; }

echo "== 2. launch pod ${POD} =="
sed "s/REPLACE_PODNAME/${POD}/; s/REPLACE_YOU/${YOU}/g" scripts/cluster/vlm_pod.yaml | kubectl apply -f -
kubectl wait --for=condition=Ready "pod/${POD}" --timeout=900s   # cold image pull can take ~10 min

echo "== 3. set up repo + deps + secrets on the pod (secrets via stdin env, not files) =="
kubectl exec "${POD}" -- bash -lc "
  set -e
  mkdir -p ${FSX_ROOT} && cd ${FSX_ROOT}
  [ -d repo ] || git clone --depth 1 ${REPO_REMOTE} repo
  cd repo
  pip -q install 'transformers>=4.49' qwen-vl-utils accelerate pillow requests safetensors >/dev/null
"
echo "== 4. copy the corpus MANIFEST to the pod (it's gitignored, so not in the cloned repo) =="
# the manifest lists which adapters to process; images are downloaded ON the pod (no GPU, has CIVITAI key)
kubectl exec "${POD}" -- mkdir -p "${FSX_ROOT}/repo/assets/corpus"
kubectl cp assets/corpus/manifest_civitai_dl.json \
  "${POD}:${FSX_ROOT}/repo/assets/corpus/manifest_civitai_dl.json"

echo "== 4b. download images on the pod (no GPU; 12/adapter, PG-first) =="
kubectl exec "${POD}" -- bash -lc "
  cd ${FSX_ROOT}/repo
  export CIVITAI_API_KEY='${CIVITAI_API_KEY:-}' PYTHONPATH=.
  python -m ditloracle.data.download_images --manifest assets/corpus/manifest_civitai_dl.json --per-adapter 12
"

echo "== 5. sensitive routing (VLM triage) then benign drafting — both --backend qwen, GPU =="
kubectl exec "${POD}" -- bash -lc "
  cd ${FSX_ROOT}/repo
  export HF_TOKEN='${HF_TOKEN:-}' CIVITAI_API_KEY='${CIVITAI_API_KEY:-}' PYTHONPATH=.
  nohup bash -c '
    python scripts/route_sensitive.py --backend qwen --out assets/corpus/sensitive_routing.json &&
    python scripts/vlm_draft_benign.py --backend qwen --out assets/corpus/vlm_drafts.json
  ' > ${FSX_ROOT}/vlm_run.log 2>&1 &
  echo started; sleep 2; tail -5 ${FSX_ROOT}/vlm_run.log || true
"
echo "
Running in background on ${POD}. Monitor:  kubectl exec ${POD} -- tail -f ${FSX_ROOT}/vlm_run.log
Pull results back:  kubectl cp ${POD}:${FSX_ROOT}/repo/assets/corpus/vlm_drafts.json ./assets/corpus/vlm_drafts.json
                    kubectl cp ${POD}:${FSX_ROOT}/repo/assets/corpus/sensitive_routing.json ./assets/corpus/sensitive_routing.json
CLEAN UP when done (frees the node):  kubectl delete pod ${POD}  &&  # then on a sleeper pod: rm -rf ${FSX_ROOT}
"
