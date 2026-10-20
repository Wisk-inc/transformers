#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#  CorX5.5 -- CorxLabs' 34B HELIX model, built from scratch: the whole program as one startup script.
#
#  Paste this into the provider's "startup script / cloud-config" field (or run it once over SSH as root).
#  It mounts the big volume, installs the software, and starts a service that runs every stage in order and
#  survives reboots: data sources checked -> CorxLabs tokenizer -> Jamaican public records crawled -> SJE/Patois
#  dialogues synthesized -> 870B tokens of data prepared -> smoke run -> 1B pilot (HELIX against a transformer
#  twin) -> the 34B run, a year long, checkpointed every 30 minutes, pushed to the Hub every 12 hours with a
#  milestone copy every 100B tokens -> the coding-agent fine-tune with the safeguards -> benchmarks -> model card.
#
#  SETTINGS -- edit before pasting. Everything else is derived.
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
CORX_HF_TOKEN="${CORX_HF_TOKEN:-}"      # your Hugging Face WRITE token, or leave empty and later: echo hf_xxx > /root/hf_token
CORX_REPO="${CORX_REPO:-CorxLabs/CorX5.5}"   # the model repository: code, checkpoints, milestones, model card
CORX_PUBLIC="${CORX_PUBLIC:-0}"          # 1 makes the repository public (private repos over 100 GB need a paid Hub plan)
CORX_SIZE="${CORX_SIZE:-34b}"            # HELIX 34b (33.9B parameters). Others: 23b, 7b, 3b, 1b, t-34b, t-14b ...
CORX_TOKENS="${CORX_TOKENS:-870e9}"      # tokens to train on; 870B is one year on 8 x B300 at ~35% utilization
CORX_HOURS="${CORX_HOURS:-8760}"         # wall-clock budget for the main run (0 = until the tokens are done)
CORX_SEQ_LEN="${CORX_SEQ_LEN:-8192}"     # pretraining context; the fine-tune runs at 32k
CORX_PILOT="${CORX_PILOT:-1}"            # 1: run the 1B pilot (HELIX vs transformer twin, ~4 h) before the big run
CORX_CRAWL_HOURS="${CORX_CRAWL_HOURS:-12}"        # Jamaican public-record crawl, in the background (0 = skip)
CORX_SYNTH_COUNT="${CORX_SYNTH_COUNT:-300000}"    # code-switched dialogues to synthesize (0 = skip)
CORX_TEACHER="${CORX_TEACHER:-Qwen/Qwen3-32B}"    # Apache-2.0 teacher served locally with vLLM for the dialogues
CORX_NODES="${CORX_NODES:-1}"            # more than 1: every node runs this script with CORX_NODE_RANK and CORX_MASTER_ADDR
CORX_NODE_RANK="${CORX_NODE_RANK:-0}"    # (multi-node needs the data directory on shared storage)
CORX_MASTER_ADDR="${CORX_MASTER_ADDR:-127.0.0.1}"
CORX_WHEEL_URL="${CORX_WHEEL_URL:-https://huggingface.co/${CORX_REPO}/resolve/main/code/helix_lm-1.1.0-py3-none-any.whl}"
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
set -uo pipefail
exec > >(tee -a /root/corx_setup.log) 2>&1
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
export DEBIAN_FRONTEND=noninteractive

say "1/6 The GPUs"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "!! no NVIDIA GPU visible"; exit 1; }

say "2/6 The volume"
read -r ACTION TARGET < <(lsblk -J -b -o NAME,PATH,SIZE,TYPE,FSTYPE,MOUNTPOINTS | python3 -c '
import json, sys
blocks = json.load(sys.stdin)["blockdevices"]
def mounts(d):
    out = [m for m in (d.get("mountpoints") or []) if m]
    for c in d.get("children") or []:
        out += mounts(c)
    return out
big = 1000e9
used = [m for d in blocks if int(d["size"]) >= big for m in mounts(d)
        if m not in ("/", "[SWAP]") and not m.startswith(("/boot", "/snap"))]
disks = [d for d in blocks if d["type"] == "disk" and int(d["size"]) >= big and not mounts(d) and not d.get("children")]
parts = [c for d in blocks for c in d.get("children") or [] if int(c["size"]) >= big and c.get("fstype") and not mounts(c)]
if "/data" in used: print("use /data")
elif used: print("use", used[0])
elif parts: print("mount", parts[0]["path"])
elif disks: print("mount" if disks[0].get("fstype") else "format", disks[0]["path"])
else: print("none -")
')
case "$ACTION" in
  use)    ROOTDIR="$TARGET" ;;
  format) blkid -p "$TARGET" >/dev/null 2>&1 && { echo "!! $TARGET is not empty: not formatting it"; exit 1; }
          mkfs.ext4 -q -F -L corx "$TARGET" ;&
  mount)  mkdir -p /data && mount "$TARGET" /data
          UUID=$(blkid -s UUID -o value "$TARGET"); FS=$(blkid -s TYPE -o value "$TARGET")
          grep -q "$UUID" /etc/fstab || echo "UUID=$UUID /data $FS defaults,nofail 0 2" >> /etc/fstab
          ROOTDIR=/data ;;
  *)      ROOTDIR=/root; echo ".. no volume of 1 TB or more found: using the main disk (the data budget will be cut to fit)" ;;
esac
DATA_DIR="$ROOTDIR/corx"; mkdir -p "$DATA_DIR/logs" "$DATA_DIR/runs" "$DATA_DIR/tmp" "$DATA_DIR/hf"; echo "$DATA_DIR" > /root/.corx_data
df -h "$DATA_DIR" | tail -1

say "3/6 Settings and the token"
cat > /etc/corx.env <<EOF
CORX_REPO=$CORX_REPO
CORX_PUBLIC=$CORX_PUBLIC
CORX_SIZE=$CORX_SIZE
CORX_TOKENS=$CORX_TOKENS
CORX_HOURS=$CORX_HOURS
CORX_SEQ_LEN=$CORX_SEQ_LEN
CORX_PILOT=$CORX_PILOT
CORX_CRAWL_HOURS=$CORX_CRAWL_HOURS
CORX_SYNTH_COUNT=$CORX_SYNTH_COUNT
CORX_TEACHER=$CORX_TEACHER
CORX_NODES=$CORX_NODES
CORX_NODE_RANK=$CORX_NODE_RANK
CORX_MASTER_ADDR=$CORX_MASTER_ADDR
CORX_WHEEL_URL=$CORX_WHEEL_URL
DATA_DIR=$DATA_DIR
EOF
chmod 600 /etc/corx.env
if [ -n "$CORX_HF_TOKEN" ]; then (umask 077; echo "$CORX_HF_TOKEN" > /root/hf_token); fi
[ -s /root/hf_token ] && echo "token: present" || echo "token: NOT SET -- the program waits; run:  echo hf_xxxxxxxx > /root/hf_token"

say "4/6 Python, PyTorch, helix-lm, kernels, vLLM (several minutes)"
apt-get update -qq && apt-get install -y -qq python3-venv python3-pip curl tmux jq >/dev/null
[ -x /root/hx/bin/python ] || python3 -m venv /root/hx
/root/hx/bin/pip install -q --upgrade pip
/root/hx/bin/pip install -q --upgrade torch
check_torch() { /root/hx/bin/python -c 'import torch; assert torch.cuda.is_available(), "PyTorch cannot see the GPU"; print("PyTorch", torch.__version__, "on", torch.cuda.get_device_name(0), "| kernels:", " ".join(torch.cuda.get_arch_list()))'; }
check_torch || echo "!! PyTorch cannot use the GPU yet (driver?); the program keeps retrying the install"
cat > /root/corx_install.sh <<'CORX_INSTALL'
#!/usr/bin/env bash
# Installs helix-lm into /root/hx. The package travels with the model repository (code/<wheel>); the token
# unlocks a private repository. Falls back to PyPI. Exit 0 when `helix` works.
set -a; . /etc/corx.env 2>/dev/null; set +a
WHEEL="helix_lm-1.1.0-py3-none-any.whl"
URL="${CORX_WHEEL_URL:-https://huggingface.co/${CORX_REPO:-CorxLabs/CorX5.5}/resolve/main/code/$WHEEL}"
token=""; [ -s /root/hf_token ] && token=$(tr -d ' \n' < /root/hf_token)
fetch_wheel() { curl -fsSL -H "Authorization: Bearer $token" -o "/root/$WHEEL" "$URL"; }
if [ -s "/root/$WHEEL" ] || fetch_wheel; then
  /root/hx/bin/pip install -q --upgrade "/root/$WHEEL" || exit 1
else
  /root/hx/bin/pip install -q --upgrade "helix-lm>=1.1.0" || exit 1
fi
/root/hx/bin/pip install -q --upgrade "helix-lm[eval,jamaica]" >/dev/null 2>&1 || echo ".. the eval extras (lm_eval, evalplus) did not install; benchmarks are skipped until they do"
/root/hx/bin/helix --version
CORX_INSTALL
chmod +x /root/corx_install.sh
bash /root/corx_install.sh || echo "!! helix-lm is not installed yet: upload helix_lm-1.1.0-py3-none-any.whl to $CORX_REPO/code/ (or set CORX_WHEEL_URL in /etc/corx.env); the program retries every 5 minutes"
/root/hx/bin/pip install -q flash-linear-attention 2>/dev/null || echo ".. flash-linear-attention did not install; the reference recurrence will be used"
check_torch >/dev/null 2>&1 || { echo ".. a dependency replaced PyTorch; putting the GPU build back"; /root/hx/bin/pip install -q --upgrade --force-reinstall torch; }
if [ "$CORX_SYNTH_COUNT" != "0" ]; then  # vLLM pins its own PyTorch, so it lives in its own environment
  [ -x /root/vx/bin/python ] || python3 -m venv /root/vx
  /root/vx/bin/pip install -q --upgrade pip >/dev/null
  /root/vx/bin/pip install -q vllm 2>/dev/null || echo ".. vLLM did not install; the dialogue synthesis stage will be skipped"
fi

say "5/6 The program"
cat > /root/corx_program.sh <<'CORX_PROGRAM'
#!/usr/bin/env bash
# Every stage of CorX5.5, in order, resumable; runs under systemd and continues after a reboot.
set -uo pipefail
set -a; . /etc/corx.env; set +a
export PATH="/root/hx/bin:$PATH" PYTHONUNBUFFERED=1 TMPDIR="$DATA_DIR/tmp" HF_HOME="$DATA_DIR/hf" HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$TMPDIR" "$HF_HOME" "$DATA_DIR/logs" "$DATA_DIR/runs"
LOGS="$DATA_DIR/logs"; STOP="$LOGS/stop"; export HELIX_STOP_FILE="$STOP"
NGPU=$(nvidia-smi --list-gpus | wc -l)
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
done_marker() { grep -q "$2" "$LOGS/$1.log" 2>/dev/null; }
is_main_node() { [ "$CORX_NODE_RANK" = "0" ]; }
billions() { awk -v t="$1" 'BEGIN{printf "%.0f", t/1e9}'; }
torchrun_args() {
  if [ "$CORX_NODES" -gt 1 ]; then echo "--nnodes=$CORX_NODES --node_rank=$CORX_NODE_RANK --master_addr=$CORX_MASTER_ADDR --master_port=29500 --nproc_per_node=$NGPU";
  else echo "--standalone --nproc_per_node=$NGPU"; fi
}

# ---- the token and the software ---------------------------------------------------------------------------
until [ -s /root/hf_token ]; do log "waiting for the Hugging Face token: echo hf_xxxxxxxx > /root/hf_token"; sleep 60; done
export HF_TOKEN; HF_TOKEN=$(tr -d ' \n' < /root/hf_token)
until helix --version >/dev/null 2>&1; do
  log "helix-lm is not installed: upload the wheel to $CORX_REPO/code/ (or set CORX_WHEEL_URL in /etc/corx.env); retrying in 5 minutes"
  sleep 300; bash /root/corx_install.sh
done
PUBLIC_FLAG=""; [ "$CORX_PUBLIC" = "1" ] && PUBLIC_FLAG="--hub-public"
log "CorX5.5: HELIX $CORX_SIZE, $(billions "$CORX_TOKENS")B tokens, $NGPU GPU(s) x $CORX_NODES node(s), repo $CORX_REPO, data $DATA_DIR"
helix size "$CORX_SIZE" | tail -8
helix kernels || true

# ---- stage 0: the repository ----------------------------------------------------------------------------
# Creates CorxLabs/CorX5.5 if it does not exist (private unless CORX_PUBLIC=1), puts the code (the wheel) under
# code/ and a first model card, so the repository holds everything needed to rebuild this machine.
if is_main_node && ! done_marker repo "REPOSITORY READY"; then
  log "stage 0: the model repository $CORX_REPO"
  python3 - <<'PY' 2>&1 | tee -a "$LOGS/repo.log"
import glob, os
from huggingface_hub import HfApi

repo, private = os.environ["CORX_REPO"], os.environ.get("CORX_PUBLIC") != "1"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, exist_ok=True, private=private)
have = set(api.list_repo_files(repo))
for wheel in sorted(glob.glob("/root/helix_lm-*.whl")):
    name = os.path.basename(wheel)
    if f"code/{name}" not in have:
        api.upload_file(path_or_fileobj=wheel, path_in_repo=f"code/{name}", repo_id=repo, commit_message=f"code: {name}")
        print(f"[repo] uploaded code/{name}")
if "README.md" not in have:
    card = (
        "---\nlicense: apache-2.0\nlanguage: [en, jam]\ntags: [helix, corxlabs, jamaica, coding-agent, long-context]\n---\n"
        "# CorX5.5\n\nCorxLabs' 34B HELIX coding agent, trained from scratch in Kingston, Jamaica. Training is in progress: "
        "`code/` holds the software, `runs/` the smoke and pilot runs, `checkpoints/corx5.5` the latest export, "
        "`milestones/` a copy every 100B tokens. The full card is written when training finishes.\n"
    )
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, commit_message="CorX5.5: training in progress")
    print("[repo] first model card uploaded")
print(f"[repo] {'private' if private else 'public'} https://huggingface.co/{repo}")
print("REPOSITORY READY")
PY
fi

# ---- stage 1: sources -------------------------------------------------------------------------------------
if is_main_node && ! done_marker verify "source"; then
  log "stage 1: the data sources"; helix data verify 2>&1 | tee "$LOGS/verify.log" | grep -E "!!|GATED" || true
  grep -q "GATED" "$LOGS/verify.log" && log "!! some gated sources need their terms accepted on the Hub with this account (the-stack-dedup, starcoderdata, the-stack-github-issues, Caselaw_Access_Project); the run continues without them until then"
fi

# ---- stage 2: the tokenizer ------------------------------------------------------------------------------
if is_main_node && [ ! -f "$DATA_DIR/tokenizer/tokenizer.json" ]; then
  log "stage 2: the CorxLabs tokenizer (65,536 tokens)"
  [ -s "$DATA_DIR/tokenizer_sample.txt" ] || helix tokenizer sample --data "$DATA_DIR" --bytes-per-source 800e6 2>&1 | tee -a "$LOGS/tokenizer.log"
  helix tokenizer train --data "$DATA_DIR" --max-bytes 16e9 2>&1 | tee -a "$LOGS/tokenizer.log"
fi
until [ -f "$DATA_DIR/tokenizer/tokenizer.json" ]; do log "waiting for the tokenizer (made on node 0)"; sleep 60; done

# ---- stage 3: Jamaican public records, in the background ---------------------------------------------------
if is_main_node && [ ! -f "$DATA_DIR/local/jamaica_gov/summary.json" ] && [ "$CORX_CRAWL_HOURS" != "0" ]; then
  log "stage 3: crawling Jamaican public records for $CORX_CRAWL_HOURS h (background, log: $LOGS/crawl.log)"
  nohup helix jamaica crawl --data "$DATA_DIR" --hours "$CORX_CRAWL_HOURS" >> "$LOGS/crawl.log" 2>&1 &
fi

# ---- stage 4: code-switched dialogues through a local teacher ---------------------------------------------
VLLM=/root/vx/bin/vllm
if is_main_node && [ "$CORX_SYNTH_COUNT" != "0" ] && ! done_marker synth "SYNTHESIZED" && [ -x "$VLLM" ]; then
  log "stage 4: serving $CORX_TEACHER with vLLM and synthesizing $CORX_SYNTH_COUNT SJE/Patois dialogues (log: $LOGS/synth.log)"
  nohup "$VLLM" serve "$CORX_TEACHER" --tensor-parallel-size "$NGPU" --max-model-len 8192 --port 8000 --gpu-memory-utilization 0.85 >> "$LOGS/vllm.log" 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 1 120); do curl -fs http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 30; done
  if curl -fs http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    helix jamaica synth --data "$DATA_DIR" --count "$CORX_SYNTH_COUNT" --model "$CORX_TEACHER" --workers 64 2>&1 | tee -a "$LOGS/synth.log"
  else
    log "!! vLLM did not come up in an hour (see $LOGS/vllm.log); continuing without the synthetic dialogues"
  fi
  kill "$VLLM_PID" 2>/dev/null; sleep 20; pkill -f "vllm serve" 2>/dev/null; sleep 10
fi

# ---- stage 5: the data, in the background ------------------------------------------------------------------
BUDGET=$(awk -v t="$CORX_TOKENS" 'BEGIN{printf "%.0f", t * 1.08}')
if is_main_node && ! grep -q "PREPARED" "$LOGS/prepare.log" 2>/dev/null; then
  WORKERS=$(( $(nproc) * 3 / 4 )); [ "$WORKERS" -gt 48 ] && WORKERS=48; [ "$WORKERS" -lt 4 ] && WORKERS=4
  log "stage 5: preparing $(billions "$BUDGET")B tokens in the background with $WORKERS workers (log: $LOGS/prepare.log)"
  nohup helix data prepare --data "$DATA_DIR" --tokens "$BUDGET" --workers "$WORKERS" --min-free-gb 800 >> "$LOGS/prepare.log" 2>&1 &
fi
tokens_on_disk() { helix data status --data "$DATA_DIR" 2>/dev/null | awk '/^total/ {gsub("B","",$2); print $2}'; }
wait_for_tokens() {  # wait_for_tokens BILLIONS: until that much is on disk, or the preparation is finished
  local want="$1" have
  while :; do
    have=$(tokens_on_disk); have=${have:-0}
    grep -q "PREPARED" "$LOGS/prepare.log" 2>/dev/null && return 0
    awk -v h="$have" -v w="$want" 'BEGIN { exit !(h >= w) }' && return 0
    log "data: ${have}B tokens on disk, waiting for ${want}B"; sleep 300
  done
}

# ---- training helper: OOM back-off and kernel fallbacks --------------------------------------------------
RB="auto"; AB="auto"
train() {  # train NAME SIZE MICRO_BATCH ARGS...
  local name="$1" size="$2" mb="$3" code; shift 3
  for attempt in 1 2 3 4 5 6; do
    [ -f "$STOP" ] && return 0
    log "train $name ($size): micro-batch $mb, attempt $attempt"
    torchrun $(torchrun_args) -m helix_lm.train --size "$size" --data "$DATA_DIR" --out "$DATA_DIR/runs/$name" \
      --micro-batch "$mb" --seq-len "$CORX_SEQ_LEN" --recurrent-backend "$RB" --attention-backend "$AB" $PUBLIC_FLAG "$@" \
      2>&1 | tee -a "$LOGS/$name.log"
    code=${PIPESTATUS[0]}
    [ "$code" -eq 0 ] && return 0
    [ -f "$STOP" ] && return 0
    if [ "$code" -eq 3 ] && [ "$mb" -gt 1 ]; then mb=$((mb / 2)); log "out of memory: micro-batch $mb"; continue; fi
    if [ "$RB" = "auto" ] && tail -n 300 "$LOGS/$name.log" | grep -qE "flash_linear_attention|fla[./]ops"; then
      RB="torch"; log "the fused recurrence failed: using the reference recurrence"; continue; fi
    if [ "$AB" = "auto" ] && tail -n 300 "$LOGS/$name.log" | grep -qE "flex_attention|_flex_"; then
      AB="sdpa"; log "FlexAttention failed: using the reference attention"; continue; fi
    log "!! $name failed (exit $code); retrying in 10 minutes (see $LOGS/$name.log)"; sleep 600
  done
  return 1
}
projection() {  # how long the whole run takes at the speed the log shows
  local rate days
  rate=$(grep -oE "tok/s [0-9.]+M" "$LOGS/$1.log" | tail -20 | awk '{s+=$2; n++} END {if (n) printf "%.3f", s/n}')
  [ -z "$rate" ] && return 0
  days=$(awk -v r="$rate" -v t="$CORX_TOKENS" 'BEGIN { printf "%.1f", t / (r * 1e6) / 86400 }')
  log "[projection] ${rate}M tokens/s -> $(billions "$CORX_TOKENS")B tokens take $days days from now (about $(date -u -d "+${days%.*} days" +%Y-%m-%d))"
}
pilot_loss() {  # the mean held-out loss at the end of a run, from its log.jsonl
  python3 - "$DATA_DIR/runs/$1/log.jsonl" <<'PY' 2>/dev/null || echo "?"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
evals = [r for r in rows if r.get("event") == "eval"]
last = evals[-1]
values = [v for k, v in last.items() if k.startswith("eval/")]
print(f"{sum(values) / len(values):.4f}")
PY
}

# ---- stage 6: smoke, then the pilot ------------------------------------------------------------------------
wait_for_tokens 3
if ! done_marker smoke "\[train\] finished"; then
  log "stage 6: smoke run (HELIX 350m, 25 minutes): kernels, FSDP, checkpoint, export, push"
  train smoke 350m 32 --total-tokens 1e9 --hours 0.4 --global-batch-tokens 1048576 --lr 1e-3 --save-every-min 8 \
    --push-every-min 10 --hub-repo "$CORX_REPO" --hub-path "runs/smoke" || { log "!! the smoke run failed; fix it, then: systemctl restart corx"; exit 1; }
fi
if [ "$CORX_PILOT" = "1" ] && ! done_marker pilot-twin "\[train\] finished"; then
  wait_for_tokens 30
  log "stage 6b: the 1B pilot -- HELIX against a transformer twin on the same tokens (about 4 hours)"
  train pilot-helix 1b 16 --total-tokens 20e9 --hours 2 --global-batch-tokens 1048576 --lr 6e-4 --eval-every 300 \
    --hub-repo "$CORX_REPO" --hub-path "runs/pilot-helix" || true
  train pilot-twin t-1b 16 --total-tokens 20e9 --hours 2 --global-batch-tokens 1048576 --lr 6e-4 --eval-every 300 \
    --hub-repo "$CORX_REPO" --hub-path "runs/pilot-twin" || true
  log "PILOT: mean held-out loss HELIX $(pilot_loss pilot-helix) vs transformer $(pilot_loss pilot-twin) at equal tokens (within ~2%: healthy; clearly worse: read docs/CORX5.5.md before spending the year)"
fi

# ---- stage 7: CorX5.5 ---------------------------------------------------------------------------------------
wait_for_tokens "$(awk -v t="$CORX_TOKENS" 'BEGIN{printf "%.0f", t/1e9*0.15}')"
if ! done_marker corx "\[train\] finished.*reason=planned"; then
  log "stage 7: CorX5.5 -- HELIX $CORX_SIZE on $(billions "$CORX_TOKENS")B tokens"
  HOURS_ARGS=(); [ "$CORX_HOURS" != "0" ] && HOURS_ARGS=(--hours "$CORX_HOURS")
  ( sleep 5400; projection corx ) &
  train corx "$CORX_SIZE" 2 --total-tokens "$CORX_TOKENS" ${HOURS_ARGS[@]+"${HOURS_ARGS[@]}"} --global-batch-tokens 4194304 \
    --lr 2e-4 --warmup-steps 2000 --save-every-min 30 --push-every-min 720 --milestone-tokens 100e9 --keep-checkpoints 2 \
    --eval-every 2000 --hub-repo "$CORX_REPO" --hub-path "checkpoints/corx5.5" || { log "!! the main run stopped; fix it, then: systemctl restart corx"; exit 1; }
  projection corx
fi
[ -f "$STOP" ] && { log "stopped by request; run 'systemctl start corx' to continue"; exit 0; }

# ---- stage 8: the coding-agent fine-tune, with the safeguards --------------------------------------------------
BASE="$DATA_DIR/runs/corx/export"
if [ -f "$BASE/config.json" ] && ! done_marker corx-instruct "\[train\] finished"; then
  log "stage 8: CorX5.5-Instruct -- agent trajectories, domain chats, code-switched dialogues, safety, identity"
  if is_main_node && [ ! -f "$DATA_DIR/sft/mixture.json" ]; then
    helix sft prepare --data "$DATA_DIR" --seq-len 32768 2>&1 | tee -a "$LOGS/sft-prepare.log"
  fi
  until [ -f "$DATA_DIR/sft/mixture.json" ]; do log "waiting for the fine-tuning data (made on node 0)"; sleep 60; done
  torchrun $(torchrun_args) -m helix_lm.train --size "$CORX_SIZE" --data "$DATA_DIR/sft" --mixture "$DATA_DIR/sft/mixture.json" \
    --tokenizer "$DATA_DIR/sft/tokenizer" --out "$DATA_DIR/runs/corx-instruct" --init-from "$BASE" --seq-len 32768 \
    --micro-batch 1 --global-batch-tokens 2097152 --lr 2e-5 --warmup-steps 100 --total-tokens 4e9 --hours 36 \
    --recurrent-backend "$RB" --attention-backend "$AB" --hub-repo "$CORX_REPO" --hub-path "instruct" $PUBLIC_FLAG \
    2>&1 | tee -a "$LOGS/corx-instruct.log" || log "!! the fine-tune failed; the base model is published"
fi

# ---- stage 9: benchmarks, safeguards check, model card ----------------------------------------------------------
MODEL="$DATA_DIR/runs/corx-instruct/export"; [ -f "$MODEL/config.json" ] || MODEL="$BASE"
if is_main_node && [ -f "$MODEL/config.json" ] && [ ! -f "$DATA_DIR/runs/eval.json" ]; then
  log "stage 9: benchmarks (MMLU-Pro, GSM8K, HumanEval+, MBPP+, ARC, HellaSwag) and the refusal / over-refusal check"
  helix eval --model "$MODEL" --out "$DATA_DIR/runs/eval.json" 2>&1 | tee -a "$LOGS/eval.log" || true
  CORX_MODEL_DIR="$MODEL" python3 - <<'PY'
import json, os, time
from helix_lm.data.identity import IDENTITY
from helix_lm.hub import upload_text

data_dir, repo = os.environ["DATA_DIR"], os.environ["CORX_REPO"]
scores = {}
try:
    scores = json.load(open(f"{data_dir}/runs/eval.json"))["scores"]
except Exception:
    pass
rows = "\n".join(f"| {k} | {v} |" for k, v in scores.items()) or "| (pending) | |"
card = f"""---
license: apache-2.0
language: [en, jam]
tags: [helix, corxlabs, jamaica, coding-agent, long-context]
---
# CorX5.5

CorX5.5 is a {IDENTITY['parameters']}-parameter HELIX language model built from scratch by CorxLabs in Kingston,
Jamaica, trained on about {IDENTITY['training_tokens']} tokens of permissively licensed code, educational web text,
mathematics, legal, financial, health and education material, and Jamaican English and Patois: the public record
of Jamaica and synthetic code-switched dialogues for banking, legal, health, education and customer-service work.

## Results
| benchmark | score |
|---|---|
{rows}

## Safeguards
Trained with a policy of declining harmful requests, professional boundaries for legal, medical and financial
questions with Jamaican referrals, and agent conduct rules; measured with XSTest (over-refusal) and a harmful
set. Deploy behind a moderation model (for example Qwen3Guard-Gen-8B) and the system prompt in
`helix_lm.data.safety.POLICY`.

## Training
HELIX architecture (helix-lm 1.1), 8 x NVIDIA B300, FSDP2, bf16. Code: `code/` in this repository.
Checkpoints: `checkpoints/corx5.5` (latest), `milestones/` (every 100B tokens), `instruct/` (the fine-tune).
Built {time.strftime('%Y-%m-%d')} from {os.environ.get('CORX_MODEL_DIR', '')}.
"""
upload_text(repo, "README.md", card, "CorX5.5 model card", private=os.environ.get("CORX_PUBLIC") != "1")
print("[card] uploaded")
PY
fi
log "done. $CORX_REPO holds the code, the checkpoints, the milestones, the fine-tune and the card."
log "Nothing else runs until you ask: the service exits. Stop the instance in the dashboard if you are finished."
CORX_PROGRAM
chmod +x /root/corx_program.sh

cat > /root/corx_stop.sh <<'CORX_STOP'
#!/usr/bin/env bash
# Every CorX job saves a checkpoint (pushed to the Hub) and stops. Nothing is deleted. `systemctl start corx` continues.
DATA_DIR=$(cat /root/.corx_data); echo stop > "$DATA_DIR/logs/stop"
pkill -f "helix data prepare" 2>/dev/null; pkill -f "helix jamaica" 2>/dev/null
echo "stop requested: the training job saves and exits within a few minutes (watch: journalctl -fu corx)"
for _ in $(seq 1 90); do pgrep -f "helix_lm.train" >/dev/null || break; sleep 10; done
systemctl stop corx; rm -f "$DATA_DIR/logs/stop"; echo "stopped. Continue with: systemctl start corx"
CORX_STOP
chmod +x /root/corx_stop.sh

say "6/6 The service"
cat > /etc/systemd/system/corx.service <<'UNIT'
[Unit]
Description=CorX5.5 training program
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/bin/bash -c '/root/corx_program.sh 2>&1 | tee -a $(cat /root/.corx_data)/run.log'
Restart=on-failure
RestartSec=300
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable corx >/dev/null 2>&1
systemctl restart corx
cat > /etc/motd <<EOF

  CorX5.5 is running as a service.   log: journalctl -fu corx     or: tail -f $DATA_DIR/run.log
  token missing?  echo hf_xxxxxxxx > /root/hf_token     stop safely: bash /root/corx_stop.sh    continue: systemctl start corx

EOF
echo; cat /etc/motd
