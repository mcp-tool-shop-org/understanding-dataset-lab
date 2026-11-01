#!/usr/bin/env bash
# Per-window Google Colab launcher for the understanding-dataset-lab scenario writer.
# Follows the colab skill: shared open lock, trap-guaranteed session stop on every
# exit path, neighbour-session preservation, token scrubbing, strict result tar
# allow-list, dual rig/VM wall caps, and Publisher outcome.draft.json output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${UDLAB_MODE:-gpu}"
RESUME_FLAG=0
CUSTOM_GRANT_ID="${UDLAB_GRANT_ID:-}"
OUT_DIR="${UDLAB_OUT_DIR:-}"
RUN_ID="${UDLAB_RUN_ID:-udlab-writer-$(date -u +%Y%m%dT%H%M%SZ)}"

usage() {
  cat <<'EOF'
Usage: scripts/colab/launch.sh [--stub] [--resume] [--grant-id ID] [--out-dir DIR] [--run-id ID]

Options:
  --stub          Run the CPU stub rung (CPU, wall 7200s, cap 0.16, --writer mock)
  --resume        Resume from an existing checkpoint in --out-dir
  --grant-id ID   Grant ID for GPU writer runs (or set UDLAB_GRANT_ID)
  --out-dir DIR   Rig output directory for logs, result.tar.gz, and outcome.draft.json
  --run-id ID     Window identifier recorded in outcome.draft.json
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stub)
      MODE="stub"
      shift
      ;;
    --resume)
      RESUME_FLAG=1
      shift
      ;;
    --grant-id)
      CUSTOM_GRANT_ID="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# --- Mode-specific configuration (stub vs GPU diff block) ---
if [[ "$MODE" == "stub" ]]; then
  CARD="CPU"
  COLAB_NEW_FLAGS=()
  WALL_SECONDS="${UDLAB_WALL_SECONDS:-7200}"
  RATE_UNITS_PER_HOUR="0.08"
  CAP_UNITS="${UDLAB_CAP_UNITS:-0.16}"
  WRITER="mock"
  GRANT_ID="${CUSTOM_GRANT_ID:-cpu-stub-none-no-gpu}"
  VRAM_HALT_MIB="${UDLAB_VRAM_HALT_MIB:-0}"
else
  CARD="A100"
  COLAB_NEW_FLAGS=(--gpu A100)
  WALL_SECONDS="${UDLAB_WALL_SECONDS:-5400}"
  RATE_UNITS_PER_HOUR="5.3"
  CAP_UNITS="${UDLAB_CAP_UNITS:-7.95}"
  WRITER="ollama"
  GRANT_ID="${CUSTOM_GRANT_ID:?GPU mode requires --grant-id or UDLAB_GRANT_ID}"
  VRAM_HALT_MIB="${UDLAB_VRAM_HALT_MIB:-26000}"
fi

COUNT="${UDLAB_COUNT:-3}"
SEED="${UDLAB_SEED:-5}"
OPEN_LOCK="${UDLAB_OPEN_LOCK:-/tmp/colab-window-open.lock}"
PYTHON_BIN="${UDLAB_PYTHON:-python3}"

if [[ -z "${UDLAB_CAP_UNITS:-}" ]]; then
  CAP_UNITS="$("$PYTHON_BIN" -c "import sys; w=float(sys.argv[1]); r=float(sys.argv[2]); print(f'{round(w * r / 3600.0, 4):g}')" "$WALL_SECONDS" "$RATE_UNITS_PER_HOUR")"
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$REPO_ROOT/runs/$RUN_ID"
fi

if [[ "$OUT_DIR" == *"/content/drive"* ]] || [[ "$OUT_DIR" == *"/gdrive"* ]]; then
  echo "FORBIDDEN_STORAGE_PATH: Google Drive paths are forbidden: $OUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
CLI_LOG="$OUT_DIR/colab-cli.log"
OUTCOME_FILE="$OUT_DIR/outcome.draft.json"
touch "$CLI_LOG"

scrub_stream() {
  sed -E 's/colab-runtime-proxy-token=[^ "'"'"'&;]+/colab-runtime-proxy-token=[REDACTED]/g'
}

log_msg() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | scrub_stream | tee -a "$CLI_LOG"
}

run_colab() {
  local out err_file err rc=0
  err_file="$(mktemp)"
  out="$(colab "$@" 2>"$err_file")" || rc=$?
  err="$(cat "$err_file" 2>/dev/null || true)"
  rm -f "$err_file"
  if [[ -n "$err" ]]; then
    printf '%s\n' "$err" | scrub_stream >> "$CLI_LOG"
  fi
  if [[ -n "$out" ]]; then
    printf '%s\n' "$out" | scrub_stream >> "$CLI_LOG"
  fi
  printf '%s\n' "$out" | scrub_stream
  return $rc
}

S=""
FETCHED=0
STOPPED=0
WATCHDOG_PID=""
STAGE_DIR=""
RUN_OUTCOME="fail"
RUN_WHY="window exited before completion"
START_EPOCH="$(date +%s)"

cleanup_stage() {
  if [[ -n "$STAGE_DIR" ]] && [[ -d "$STAGE_DIR" ]]; then
    rm -rf "$STAGE_DIR" 2>/dev/null || true
  fi
}

fetch_results_once() {
  local fetch_timeout="${1:-0}"
  if [[ "$FETCHED" -eq 1 ]] || [[ -z "$S" ]] || [[ -f "$OUT_DIR/.rig_fetched" ]]; then
    return 0
  fi
  FETCHED=1
  touch "$OUT_DIR/.rig_fetched"
  log_msg "FETCH_START session=$S"
  if [[ "$fetch_timeout" -gt 0 ]]; then
    timeout --signal=TERM --kill-after=5 "${fetch_timeout}s" bash -c '
      colab cp -s "$1" "/content/result.tar.gz" "$2/result.tar.gz" >/dev/null 2>&1 || true
      colab cp -s "$1" "/content/udlab_run/vm_receipt.json" "$2/vm_receipt.json" >/dev/null 2>&1 || true
    ' _ "$S" "$OUT_DIR" || true
  else
    run_colab cp -s "$S" "/content/result.tar.gz" "$OUT_DIR/result.tar.gz" >/dev/null 2>&1 || true
    run_colab cp -s "$S" "/content/udlab_run/vm_receipt.json" "$OUT_DIR/vm_receipt.json" >/dev/null 2>&1 || true
  fi
  log_msg "FETCH_DONE session=$S"
}

write_outcome_draft() {
  local end_epoch runtime_s
  end_epoch="$(date +%s)"
  runtime_s=$(( end_epoch - START_EPOCH ))
  if [[ "$runtime_s" -lt 0 ]]; then
    runtime_s=0
  fi
  "$PYTHON_BIN" - "$OUTCOME_FILE" "$RUN_ID" "$CARD" "$CAP_UNITS" "$RATE_UNITS_PER_HOUR" "$runtime_s" "$RUN_OUTCOME" "$RUN_WHY" <<'PY'
import datetime
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
run_id = sys.argv[2]
hardware = sys.argv[3]
cap = float(sys.argv[4])
rate = float(sys.argv[5])
runtime_seconds = int(sys.argv[6])
outcome = sys.argv[7]
why = sys.argv[8]
cost = round((runtime_seconds / 3600.0) * rate, 4)

receipt_path = out_path.parent / "vm_receipt.json"
if receipt_path.is_file():
    try:
        receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_hw = receipt_data.get("hardware")
        if isinstance(receipt_hw, str) and receipt_hw.strip():
            hardware = receipt_hw.strip()
    except Exception:
        pass

payload = {
    "id": run_id,
    "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
    "venue": "colab",
    "hardware": hardware,
    "cap": cap,
    "cost": cost,
    "cost_unit": "colab_units",
    "cost_basis": "rate_x_runtime",
    "runtime_seconds": runtime_seconds,
    "outcome": outcome,
    "why": why,
}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
PY
}

stop_all() {
  local exit_code=$?
  set +e
  cleanup_stage
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""
  fi
  if [[ -n "$S" ]] && [[ "$STOPPED" -eq 0 ]] && [[ ! -f "$OUT_DIR/.rig_stopped" ]]; then
    fetch_results_once
    log_msg "STOP_SESSION session=$S"
    run_colab stop -s "$S" >/dev/null 2>&1 || true
    STOPPED=1
    touch "$OUT_DIR/.rig_stopped"
  fi
  rm -f "$OUT_DIR/.rig_fetched" "$OUT_DIR/.rig_stopped" 2>/dev/null || true
  "$PYTHON_BIN" -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts' / 'colab')); import vm_udlab; vm_udlab.scrub_log_file_in_place(Path(sys.argv[2]))" "$REPO_ROOT" "$CLI_LOG" >/dev/null 2>&1 || true
  write_outcome_draft
  exit "$exit_code"
}

trap stop_all EXIT
trap 'RUN_WHY="interrupted by signal"; exit 143' TERM INT

log_msg "WINDOW_START id=$RUN_ID mode=$MODE card=$CARD wall_s=$WALL_SECONDS rate_uh=$RATE_UNITS_PER_HOUR cap_units=$CAP_UNITS writer=$WRITER count=$COUNT seed=$SEED grant_id=$GRANT_ID vram_halt_mib=$VRAM_HALT_MIB"

PROMPTS_DIR="${UDLAB_EXACT_CTX_PROMPTS:-}"
if [[ "$PROMPTS_DIR" == *"/content/drive"* ]] || [[ "$PROMPTS_DIR" == *"/gdrive"* ]]; then
  RUN_WHY="Google Drive paths are forbidden for UDLAB_EXACT_CTX_PROMPTS"
  echo "FORBIDDEN_STORAGE_PATH: Google Drive paths are forbidden: $PROMPTS_DIR" >&2
  exit 2
fi
if [[ -z "$PROMPTS_DIR" ]] || [[ ! -d "$PROMPTS_DIR" ]]; then
  RUN_WHY="UDLAB_EXACT_CTX_PROMPTS directory is not set or missing"
  echo "EXACT_COUNT_PROMPT_MISSING: UDLAB_EXACT_CTX_PROMPTS must point to the out-of-repo exact-ctx prompts directory" >&2
  exit 2
fi
INDEX_JSON="${UDLAB_EXACT_CTX_INDEX:-$REPO_ROOT/tests/fixtures/exact_ctx/index.json}"

STAGE_DIR="$(mktemp -d)"
PROMPTS_TAR="$STAGE_DIR/exact_ctx_prompts.tar.gz"
REPO_TAR="$STAGE_DIR/repo_bundle.tar.gz"

if ! "$PYTHON_BIN" - "$INDEX_JSON" "$PROMPTS_DIR" "$PROMPTS_TAR" <<'PY'
import hashlib
import json
import sys
import tarfile
from pathlib import Path

index_path = Path(sys.argv[1])
prompts_dir = Path(sys.argv[2])
tar_out = Path(sys.argv[3])

entries = json.loads(index_path.read_text(encoding="utf-8"))
if len(entries) != 4:
    raise SystemExit(f"EXACT_COUNT_INDEX_MISMATCH: expected 4 entries, got {len(entries)}")

with tarfile.open(tar_out, "w:gz") as tf:
    for entry in entries:
        fname = entry.get("file") or entry.get("prompt_file")
        fpath = prompts_dir / fname
        if not fpath.is_file():
            raise SystemExit(f"EXACT_COUNT_PROMPT_MISSING: missing {fname} in {prompts_dir}")
        content = fpath.read_bytes().decode("utf-8")
        messages = [{"role": "user", "content": content}]
        actual_sha = hashlib.sha256(
            json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if actual_sha.lower() != entry["prompt_sha256"].lower():
            raise SystemExit(f"EXACT_COUNT_SHA256_MISMATCH: {fname} sha256 mismatch on rig")
        tf.add(fpath, arcname=fname, recursive=False)
PY
then
  RUN_WHY="exact-count prompt verification failed on rig before upload"
  exit 2
fi

"$PYTHON_BIN" - "$REPO_ROOT" "$REPO_TAR" "$INDEX_JSON" <<'PY'
import sys
import tarfile
from pathlib import Path

repo_root = Path(sys.argv[1])
tar_out = Path(sys.argv[2])
index_override = Path(sys.argv[3])
include_paths = [
    "pyproject.toml",
    ".python-version",
    "uv.lock",
    "README.md",
    "src",
    "scripts",
]

with tarfile.open(tar_out, "w:gz") as tf:
    for rel in include_paths:
        target = repo_root / rel
        if not target.exists():
            continue
        if target.is_file():
            tf.add(target, arcname=rel, recursive=False)
        else:
            for child in sorted(target.rglob("*")):
                if "__pycache__" in child.parts or child.suffix == ".pyc":
                    continue
                if child.is_file():
                    tf.add(child, arcname=child.relative_to(repo_root).as_posix(), recursive=False)
    tf.add(index_override, arcname="tests/fixtures/exact_ctx/index.json", recursive=False)
PY

mkdir -p "$(dirname "$OPEN_LOCK")"
exec 9>"$OPEN_LOCK"
flock -w 300 9 || {
  RUN_WHY="failed to acquire open lock $OPEN_LOCK"
  exit 1
}

log_msg "LOCK_ACQUIRED lock=$OPEN_LOCK"
PRE_LS="$(run_colab ls --json || echo "[]")"
NEW_OUT="$(run_colab new "${COLAB_NEW_FLAGS[@]+"${COLAB_NEW_FLAGS[@]}"}" --json)" || {
  flock -u 9 || true
  RUN_WHY="colab new failed"
  exit 1
}

S="$("$PYTHON_BIN" -c 'import json, sys; data = json.loads(sys.argv[1]); print(data.get("session_id") or data.get("id") or "")' "$NEW_OUT")"
if [[ -z "$S" ]]; then
  flock -u 9 || true
  RUN_WHY="colab new returned empty session id"
  exit 1
fi
log_msg "SESSION_OPENED session=$S pre_sessions=$(printf '%s' "$PRE_LS" | tr -d '\n ')"

GATE_OUT="$(run_colab exec -s "$S" -- "echo VM_GATE_OK && uname -a")" || {
  flock -u 9 || true
  RUN_WHY="VM liveness gate failed on session $S"
  exit 1
}
flock -u 9 || true
log_msg "LOCK_RELEASED session=$S"

# Start independent rig-side deadline watchdog (fires at WALL - FETCH_MARGIN, fetches, then stops by WALL)
FETCH_MARGIN="${UDLAB_FETCH_MARGIN_SECONDS:-180}"
WATCHDOG_FETCH_TIMEOUT="${UDLAB_WATCHDOG_FETCH_TIMEOUT_SECONDS:-$(( FETCH_MARGIN > 10 ? FETCH_MARGIN - 5 : (FETCH_MARGIN > 0 ? FETCH_MARGIN : 1) ))}"
if [[ "$WALL_SECONDS" -gt "$FETCH_MARGIN" ]]; then
  WATCHDOG_SLEEP_SECONDS=$(( WALL_SECONDS - FETCH_MARGIN ))
else
  WATCHDOG_SLEEP_SECONDS=0
fi

DEADLINE_MARKER="$OUT_DIR/.rig_deadline_exceeded"
rm -f "$DEADLINE_MARKER" "$OUT_DIR/.rig_fetched" "$OUT_DIR/.rig_stopped"
(
  sleep "$WATCHDOG_SLEEP_SECONDS"
  touch "$DEADLINE_MARKER"
  printf '%s RIG_DEADLINE_EXCEEDED session=%s wall_s=%s fetch_margin_s=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$S" "$WALL_SECONDS" "$FETCH_MARGIN" >> "$CLI_LOG"
  fetch_results_once "$WATCHDOG_FETCH_TIMEOUT"
  printf '%s WATCHDOG_STOP_SESSION session=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$S" >> "$CLI_LOG"
  colab stop -s "$S" >/dev/null 2>&1 || true
  touch "$OUT_DIR/.rig_stopped"
) &
WATCHDOG_PID="$!"

run_colab cp -s "$S" "$REPO_TAR" "/content/repo_bundle.tar.gz" >/dev/null
run_colab cp -s "$S" "$PROMPTS_TAR" "/content/exact_ctx_prompts.tar.gz" >/dev/null
cleanup_stage

run_colab exec -s "$S" -- "mkdir -p /content/udlab /content/udlab_run && tar -xzf /content/repo_bundle.tar.gz -C /content/udlab && rm -f /content/repo_bundle.tar.gz" >/dev/null

if [[ "$RESUME_FLAG" -eq 1 ]] || [[ -f "$OUT_DIR/checkpoint.json" ]]; then
  RESUME_FLAG=1
  if [[ -f "$OUT_DIR/checkpoint.json" ]]; then
    RESUME_TAR="$(mktemp -u).tar.gz"
    "$PYTHON_BIN" - "$OUT_DIR" "$RESUME_TAR" <<'PY'
import sys
import tarfile
from pathlib import Path

out_dir = Path(sys.argv[1])
tar_path = Path(sys.argv[2])
candidates = [
    "checkpoint.json",
    "generations.jsonl",
    "build_summary.json",
    "sealed_store/keys.sealed.jsonl",
    "sealed_store/sealing.sealed.json",
    "sealed_store/checkpoint.sealed.json",
    "sealed_store/responses.sealed.jsonl",
    "part_public/items.jsonl",
    "part_public/part.json",
]
with tarfile.open(tar_path, "w:gz") as tf:
    for rel in candidates:
        p = out_dir / rel
        if p.is_file():
            tf.add(p, arcname=rel, recursive=False)
PY
    run_colab cp -s "$S" "$RESUME_TAR" "/content/resume_state.tar.gz" >/dev/null
    rm -f "$RESUME_TAR"
    run_colab exec -s "$S" -- "tar -xzf /content/resume_state.tar.gz -C /content/udlab_run && rm -f /content/resume_state.tar.gz" >/dev/null
  fi
fi

VM_ARGS=(
  "--mode" "$MODE"
  "--writer" "$WRITER"
  "--count" "$COUNT"
  "--seed" "$SEED"
  "--grant-id" "$GRANT_ID"
  "--vram-halt-mib" "$VRAM_HALT_MIB"
  "--wall-seconds" "$WALL_SECONDS"
  "--repo-dir" "/content/udlab"
  "--prompts-dir" "/content/udlab_prompts"
  "--prompts-tar" "/content/exact_ctx_prompts.tar.gz"
  "--run-dir" "/content/udlab_run"
  "--result-tar" "/content/result.tar.gz"
)
if [[ "$RESUME_FLAG" -eq 1 ]]; then
  VM_ARGS+=("--resume")
fi
if [[ "${VM_DRYRUN:-0}" == "1" ]]; then
  VM_ARGS+=("--dryrun")
fi

RIG_GRACE_SECONDS="${UDLAB_RIG_GRACE_SECONDS:-5}"
RIG_TIMEOUT=$(( WALL_SECONDS + RIG_GRACE_SECONDS ))
EXEC_RC=0
timeout --signal=TERM --kill-after=15 "${RIG_TIMEOUT}s" \
  colab exec -s "$S" -- "timeout --signal=TERM --kill-after=30 ${WALL_SECONDS}s python3 /content/udlab/scripts/colab/vm_udlab.py ${VM_ARGS[*]}" 2>&1 | scrub_stream >> "$CLI_LOG" || EXEC_RC=$?

if [[ -f "$DEADLINE_MARKER" ]]; then
  if [[ -n "$WATCHDOG_PID" ]]; then
    wait "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""
  fi
  rm -f "$DEADLINE_MARKER"
  EXEC_RC=124
fi

fetch_results_once

if [[ "$EXEC_RC" -ne 0 ]]; then
  if [[ "$EXEC_RC" -eq 124 ]] || [[ "$EXEC_RC" -eq 137 ]]; then
    RUN_WHY="wall-clock timeout exceeded (${WALL_SECONDS}s)"
  else
    RUN_WHY="vm_udlab.py exited non-zero ($EXEC_RC)"
  fi
  exit "$EXEC_RC"
fi

if [[ ! -f "$OUT_DIR/result.tar.gz" ]]; then
  RUN_WHY="result.tar.gz missing after VM run"
  exit 1
fi

EXTRACT_DIR="$OUT_DIR/extracted"
mkdir -p "$EXTRACT_DIR"
if ! "$PYTHON_BIN" - "$REPO_ROOT" "$OUT_DIR/result.tar.gz" "$EXTRACT_DIR" "$CLI_LOG" <<'PY'
import json
import sys
import tarfile
from pathlib import Path

repo_root = Path(sys.argv[1])
tar_path = Path(sys.argv[2])
extract_dir = Path(sys.argv[3])
cli_log = Path(sys.argv[4])

sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_udlab

vm_udlab.validate_tar_file(tar_path)
with tarfile.open(tar_path, "r:*") as tf:
    tf.extractall(extract_dir)

prompt_line_hashes: list[str] = []
receipt_file = extract_dir / "vm_receipt.json"
if receipt_file.is_file():
    try:
        rdata = json.loads(receipt_file.read_text(encoding="utf-8"))
        prompt_line_hashes = list(rdata.get("prompt_line_sha256s") or [])
    except Exception:
        pass

vm_udlab.scrub_log_file_in_place(cli_log)
log_files = list((extract_dir / "logs").glob("*.log")) + [cli_log]
vm_udlab.verify_logs_clean(log_files, prompt_line_hashes=prompt_line_hashes)
PY
then
  RUN_WHY="result tar allow-list or log policy check failed on rig"
  exit 2
fi

RUN_OUTCOME="ok"
RUN_WHY="completed ${COUNT} scenario(s) in ${MODE} mode (${WRITER} writer)"
log_msg "WINDOW_OK id=$RUN_ID session=$S"
exit 0
