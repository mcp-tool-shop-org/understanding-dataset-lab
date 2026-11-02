#!/usr/bin/env bash
# Per-window Google Colab launcher for the understanding-dataset-lab scenario writer.
# Follows the colab skill: shared open lock, reconciler lock file convention,
# trap-guaranteed session stop on every exit path, neighbour-session preservation,
# token scrubbing, strict result tar allow-list, dual rig/VM wall caps, and
# Publisher outcome.draft.json output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${UDLAB_MODE:-gpu}"
RESUME_FLAG=0
DRY_FLAG="${UDLAB_DRY:-0}"
SESSION_SMOKE_FLAG=0
CUSTOM_GRANT_ID="${UDLAB_GRANT_ID:-}"
OUT_DIR="${UDLAB_OUT_DIR:-}"
RUN_ID="${UDLAB_RUN_ID:-writer-$(date -u +%Y%m%dT%H%M%SZ)}"

usage() {
  cat <<'EOF'
Usage: scripts/colab/launch.sh [--stub] [--session-smoke] [--dry] [--resume] [--grant-id ID] [--out-dir DIR] [--run-id ID]

Options:
  --stub            Run the CPU stub rung (CPU, wall 7200s, cap 0.16, --writer mock)
  --session-smoke   Run the first-rung CPU session smoke test only (no upload, no pull)
  --dry             Print every CLI command line with real values in order and exit
  --resume          Resume from an existing checkpoint in --out-dir
  --grant-id ID     Grant ID for GPU writer runs (or set UDLAB_GRANT_ID)
  --out-dir DIR     Rig output directory for logs, result.tar.gz, and outcome.draft.json
  --run-id ID       Window identifier recorded in outcome.draft.json (session is udlab-<run-id>)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stub)
      MODE="stub"
      shift
      ;;
    --session-smoke)
      SESSION_SMOKE_FLAG=1
      MODE="stub"
      shift
      ;;
    --dry|--dry-run)
      DRY_FLAG=1
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
LOCK_DIR="${COLAB_LOCK_DIR:-/tmp/colab-locks}"
RECONCILE_SH="${COLAB_RECONCILE_SH:-/mnt/e/AI/rnd-local/projects/colab-lanes/reconcile/reconcile.sh}"
MAIN=$$
PYTHON_BIN="${UDLAB_PYTHON:-python3}"
GATE_TIMEOUT_SECONDS="${UDLAB_GATE_TIMEOUT_SECONDS:-180}"
UPLOAD_TIMEOUT_SECONDS="${UDLAB_UPLOAD_TIMEOUT_SECONDS:-600}"
DOWNLOAD_TIMEOUT_SECONDS="${UDLAB_DOWNLOAD_TIMEOUT_SECONDS:-300}"

# --- Ordered timeout hierarchy derived from WALL_SECONDS in one place ---
# Order: VM_WALL_SECONDS < EXEC_TIMEOUT_SECONDS < WATCHDOG_SLEEP_SECONDS < RIG_TIMEOUT
#   1. WATCHDOG_SLEEP_SECONDS = WALL_SECONDS - FETCH_MARGIN (margin 180s before WALL_SECONDS for fetch + stop)
#   2. EXEC_TIMEOUT_SECONDS   = WATCHDOG_SLEEP_SECONDS - EXEC_MARGIN (margin 60s before rig watchdog)
#   3. VM_WALL_SECONDS        = EXEC_TIMEOUT_SECONDS - VM_SELF_STOP_MARGIN (margin 60s before colab exec --timeout)
#   4. RIG_TIMEOUT            = WALL_SECONDS + RIG_GRACE_SECONDS (margin 185s after rig watchdog / 5s after WALL_SECONDS)
FETCH_MARGIN="${UDLAB_FETCH_MARGIN_SECONDS:-180}"
EXEC_MARGIN="${UDLAB_EXEC_MARGIN_SECONDS:-60}"
VM_SELF_STOP_MARGIN="${UDLAB_VM_SELF_STOP_MARGIN_SECONDS:-60}"
RIG_GRACE_SECONDS="${UDLAB_RIG_GRACE_SECONDS:-5}"
WATCHDOG_FETCH_TIMEOUT="${UDLAB_WATCHDOG_FETCH_TIMEOUT_SECONDS:-$(( FETCH_MARGIN > 10 ? FETCH_MARGIN - 5 : (FETCH_MARGIN > 0 ? FETCH_MARGIN : 1) ))}"

if [[ "$WALL_SECONDS" -gt "$FETCH_MARGIN" ]]; then
  WATCHDOG_SLEEP_SECONDS=$(( WALL_SECONDS - FETCH_MARGIN ))
else
  WATCHDOG_SLEEP_SECONDS=1
fi
if [[ "$WATCHDOG_SLEEP_SECONDS" -gt "$EXEC_MARGIN" ]]; then
  EXEC_TIMEOUT_SECONDS=$(( WATCHDOG_SLEEP_SECONDS - EXEC_MARGIN ))
else
  EXEC_TIMEOUT_SECONDS=$(( WATCHDOG_SLEEP_SECONDS > 1 ? WATCHDOG_SLEEP_SECONDS - 1 : 1 ))
fi
if [[ "$EXEC_TIMEOUT_SECONDS" -gt "$VM_SELF_STOP_MARGIN" ]]; then
  VM_WALL_SECONDS=$(( EXEC_TIMEOUT_SECONDS - VM_SELF_STOP_MARGIN ))
else
  VM_WALL_SECONDS=$(( EXEC_TIMEOUT_SECONDS > 1 ? EXEC_TIMEOUT_SECONDS - 1 : 1 ))
fi
RIG_TIMEOUT=$(( WALL_SECONDS + RIG_GRACE_SECONDS ))
if [[ "$RIG_TIMEOUT" -le "$WATCHDOG_SLEEP_SECONDS" ]]; then
  RIG_TIMEOUT=$(( WATCHDOG_SLEEP_SECONDS + 1 ))
fi

if [[ "$RUN_ID" == udlab-* ]]; then
  SESSION_NAME="$RUN_ID"
else
  SESSION_NAME="udlab-$RUN_ID"
fi

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

scrub_stream() {
  sed -E 's/colab-runtime-proxy-token=[^ "'"'"'&;]+/colab-runtime-proxy-token=[REDACTED]/g'
}

if [[ "$DRY_FLAG" == "1" ]]; then
  if [[ "$SESSION_SMOKE_FLAG" -eq 1 ]]; then
    printf 'colab sessions\n'
    printf 'colab new -s %s\n' "$SESSION_NAME"
    printf 'colab sessions\n'
    printf 'colab stop -s %s\n' "$SESSION_NAME"
    printf 'colab sessions\n'
    exit 0
  fi

  DRY_STAGE_DIR="${OUT_DIR}/stage"
  DRY_REPO_TAR="${DRY_STAGE_DIR}/repo_bundle.tar.gz"
  DRY_PROMPTS_TAR="${DRY_STAGE_DIR}/exact_ctx_prompts.tar.gz"
  DRY_RESUME_TAR="${DRY_STAGE_DIR}/resume_state.tar.gz"
  DRY_NEW_OPTS=""
  if [[ "${#COLAB_NEW_FLAGS[@]}" -gt 0 ]]; then
    DRY_NEW_OPTS=" ${COLAB_NEW_FLAGS[*]}"
  fi
  DRY_RESUME_VAL=0
  if [[ "$RESUME_FLAG" -eq 1 ]] || [[ -f "$OUT_DIR/checkpoint.json" ]]; then
    DRY_RESUME_VAL=1
  fi
  DRY_ENV_STR="--env UDLAB_STEP=run --env UDLAB_MODE=${MODE} --env UDLAB_WRITER=${WRITER} --env UDLAB_COUNT=${COUNT} --env UDLAB_SEED=${SEED} --env UDLAB_GRANT_ID=${GRANT_ID} --env UDLAB_VRAM_HALT_MIB=${VRAM_HALT_MIB} --env UDLAB_WALL_SECONDS=${VM_WALL_SECONDS} --env UDLAB_REPO_DIR=/content/udlab --env UDLAB_REPO_TAR=/content/repo_bundle.tar.gz --env UDLAB_PROMPTS_DIR=/content/udlab_prompts --env UDLAB_PROMPTS_TAR=/content/exact_ctx_prompts.tar.gz --env UDLAB_RUN_DIR=/content/udlab_run --env UDLAB_RESULT_TAR=/content/result.tar.gz --env UDLAB_STATUS_FILE=/content/vm_status.json --env UDLAB_RESUME=${DRY_RESUME_VAL}"
  if [[ "$DRY_RESUME_VAL" -eq 1 ]] && [[ -f "$OUT_DIR/checkpoint.json" ]]; then
    DRY_ENV_STR="${DRY_ENV_STR} --env UDLAB_RESUME_TAR=/content/resume_state.tar.gz"
  fi
  if [[ "${VM_DRYRUN:-0}" == "1" ]]; then
    DRY_ENV_STR="${DRY_ENV_STR} --env VM_DRYRUN=1"
  fi

  {
    printf 'colab sessions\n'
    printf 'colab new%s -s %s\n' "$DRY_NEW_OPTS" "$SESSION_NAME"
    printf 'colab exec -s %s --env UDLAB_STEP=gate -f %s --timeout %s\n' "$SESSION_NAME" "$SCRIPT_DIR/vm_udlab.py" "$GATE_TIMEOUT_SECONDS"
    printf 'colab upload -s %s %s /content/repo_bundle.tar.gz\n' "$SESSION_NAME" "$DRY_REPO_TAR"
    printf 'colab upload -s %s %s /content/exact_ctx_prompts.tar.gz\n' "$SESSION_NAME" "$DRY_PROMPTS_TAR"
    if [[ "$DRY_RESUME_VAL" -eq 1 ]] && [[ -f "$OUT_DIR/checkpoint.json" ]]; then
      printf 'colab upload -s %s %s /content/resume_state.tar.gz\n' "$SESSION_NAME" "$DRY_RESUME_TAR"
    fi
    printf 'colab exec -s %s %s -f %s --timeout %s\n' "$SESSION_NAME" "$DRY_ENV_STR" "$SCRIPT_DIR/vm_udlab.py" "$EXEC_TIMEOUT_SECONDS"
    printf 'colab download -s %s /content/result.tar.gz %s/result.tar.gz\n' "$SESSION_NAME" "$OUT_DIR"
    printf 'colab download -s %s /content/vm_status.json %s/vm_status.json\n' "$SESSION_NAME" "$OUT_DIR"
    printf 'colab download -s %s /content/udlab_run/vm_receipt.json %s/vm_receipt.json\n' "$SESSION_NAME" "$OUT_DIR"
    printf 'colab stop -s %s\n' "$SESSION_NAME"
  } | scrub_stream
  exit 0
fi

mkdir -p "$OUT_DIR"
CLI_LOG="$OUT_DIR/colab-cli.log"
OUTCOME_FILE="$OUT_DIR/outcome.draft.json"
FETCH_CLAIM_DIR="$OUT_DIR/.rig_fetch_claim"
FETCH_DONE_MARKER="$OUT_DIR/.rig_fetched"
DEADLINE_MARKER="$OUT_DIR/.rig_deadline_exceeded"
touch "$CLI_LOG"

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
NEW_SUCCEEDED=0
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

wait_for_fetch_done() {
  local wait_timeout="${1:-${WATCHDOG_FETCH_TIMEOUT:-175}}"
  if [[ "$wait_timeout" -le 0 ]]; then
    wait_timeout="${WATCHDOG_FETCH_TIMEOUT:-175}"
  fi
  local deadline=$(( $(date +%s) + wait_timeout ))
  while [[ ! -f "$FETCH_DONE_MARKER" ]]; do
    if [[ "$(date +%s)" -ge "$deadline" ]]; then
      break
    fi
    sleep 0.05
  done
}

fetch_results_once() {
  local fetch_timeout="${1:-0}"
  local wait_timeout="$fetch_timeout"
  if [[ "$wait_timeout" -le 0 ]]; then
    wait_timeout="${WATCHDOG_FETCH_TIMEOUT:-175}"
  fi
  if [[ "$NEW_SUCCEEDED" -ne 1 ]] || [[ -z "$S" ]] || [[ "$SESSION_SMOKE_FLAG" -eq 1 ]]; then
    return 0
  fi
  if [[ -f "$FETCH_DONE_MARKER" ]]; then
    FETCHED=1
    return 0
  fi
  if ! mkdir "$FETCH_CLAIM_DIR" 2>/dev/null; then
    wait_for_fetch_done "$wait_timeout"
    FETCHED=1
    return 0
  fi
  FETCHED=1
  log_msg "FETCH_START session=$S"
  if [[ "$fetch_timeout" -gt 0 ]]; then
    timeout --signal=TERM --kill-after=5 "${fetch_timeout}s" bash -c '
      colab download -s "$1" "/content/result.tar.gz" "$2/result.tar.gz" >/dev/null 2>&1 || true
      colab download -s "$1" "/content/vm_status.json" "$2/vm_status.json" >/dev/null 2>&1 || true
      colab download -s "$1" "/content/udlab_run/vm_receipt.json" "$2/vm_receipt.json" >/dev/null 2>&1 || true
    ' _ "$S" "$OUT_DIR" || true
  else
    timeout "$DOWNLOAD_TIMEOUT_SECONDS" colab download -s "$S" "/content/result.tar.gz" "$OUT_DIR/result.tar.gz" >/dev/null 2>&1 || true
    timeout "$DOWNLOAD_TIMEOUT_SECONDS" colab download -s "$S" "/content/vm_status.json" "$OUT_DIR/vm_status.json" >/dev/null 2>&1 || true
    timeout "$DOWNLOAD_TIMEOUT_SECONDS" colab download -s "$S" "/content/udlab_run/vm_receipt.json" "$OUT_DIR/vm_receipt.json" >/dev/null 2>&1 || true
  fi
  log_msg "FETCH_DONE session=$S"
  touch "$FETCH_DONE_MARKER"
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
  if [[ "$NEW_SUCCEEDED" -eq 1 ]] && [[ -n "$S" ]] && [[ "$STOPPED" -eq 0 ]] && [[ ! -f "$OUT_DIR/.rig_stopped" ]]; then
    fetch_results_once "$WATCHDOG_FETCH_TIMEOUT"
  fi
  if [[ -n "$WATCHDOG_PID" ]]; then
    if [[ -f "$DEADLINE_MARKER" ]]; then
      wait "$WATCHDOG_PID" 2>/dev/null || true
    else
      pkill -P "$WATCHDOG_PID" 2>/dev/null || true
      kill "$WATCHDOG_PID" 2>/dev/null || true
      wait "$WATCHDOG_PID" 2>/dev/null || true
    fi
    WATCHDOG_PID=""
  fi
  if [[ "$NEW_SUCCEEDED" -eq 1 ]] && [[ -n "$S" ]] && [[ "$STOPPED" -eq 0 ]] && [[ ! -f "$OUT_DIR/.rig_stopped" ]]; then
    log_msg "STOP_SESSION session=$S"
    run_colab stop -s "$S" >/dev/null 2>&1 || true
    rm -f "$LOCK_DIR/$S.lock"
    STOPPED=1
    touch "$OUT_DIR/.rig_stopped"
    if [[ "$SESSION_SMOKE_FLAG" -eq 1 ]]; then
      POST_SESSIONS="$(run_colab sessions || true)"
      log_msg "SESSIONS_AFTER_STOP session=$S sessions=$(printf '%s' "$POST_SESSIONS" | tr '\n' ' ')"
      if printf '%s\n' "$POST_SESSIONS" | grep -q "^\[$S\]"; then
        RUN_OUTCOME="fail"
        RUN_WHY="session $S still present after colab stop"
        exit_code=1
      fi
    fi
  elif [[ "$NEW_SUCCEEDED" -eq 1 ]] && [[ -n "$S" ]]; then
    rm -f "$LOCK_DIR/$S.lock"
  fi
  rm -rf "$DEADLINE_MARKER" "$FETCH_CLAIM_DIR" "$FETCH_DONE_MARKER" "$OUT_DIR/.rig_stopped" 2>/dev/null || true
  "$PYTHON_BIN" -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts' / 'colab')); import vm_udlab; vm_udlab.scrub_log_file_in_place(Path(sys.argv[2]))" "$REPO_ROOT" "$CLI_LOG" >/dev/null 2>&1 || true
  write_outcome_draft
  exit "$exit_code"
}

trap stop_all EXIT
trap 'RUN_WHY="interrupted by signal"; exit 143' TERM INT

if [[ "$SESSION_SMOKE_FLAG" -eq 1 ]]; then
  log_msg "SESSION_SMOKE_START id=$RUN_ID session=$SESSION_NAME card=$CARD"
  mkdir -p "$(dirname "$OPEN_LOCK")"
  exec 9>"$OPEN_LOCK"
  flock -w 300 9 || {
    RUN_WHY="failed to acquire open lock $OPEN_LOCK"
    exit 1
  }
  log_msg "LOCK_ACQUIRED lock=$OPEN_LOCK"
  PRE_SESSIONS="$(run_colab sessions || true)"
  if printf '%s\n' "$PRE_SESSIONS" | grep -Fq "[$SESSION_NAME]"; then
    flock -u 9 || true
    RUN_WHY="session $SESSION_NAME already exists in colab sessions; refusing to reuse name"
    log_msg "SESSION_NAME_IN_USE session=$SESSION_NAME"
    echo "SESSION_NAME_IN_USE: session [$SESSION_NAME] already exists in colab sessions" >&2
    exit 1
  fi
  if ! run_colab new -s "$SESSION_NAME" >/dev/null; then
    flock -u 9 || true
    RUN_WHY="colab new failed for $SESSION_NAME"
    exit 1
  fi
  S="$SESSION_NAME"
  NEW_SUCCEEDED=1
  mkdir -p "$LOCK_DIR"
  printf '%s\n' "$MAIN" > "$LOCK_DIR/$S.lock"
  log_msg "SESSION_LOCK_WRITTEN lock_file=$LOCK_DIR/$S.lock pid=$MAIN"
  MID_SESSIONS="$(run_colab sessions || true)"
  log_msg "SESSION_OPENED session=$S pre_sessions=$(printf '%s' "$PRE_SESSIONS" | tr -d '\n ') mid_sessions=$(printf '%s' "$MID_SESSIONS" | tr '\n' ' ')"
  if ! printf '%s\n' "$MID_SESSIONS" | grep -q "^\[$S\]"; then
    flock -u 9 || true
    RUN_WHY="colab sessions did not list [$S] after colab new"
    exit 1
  fi
  flock -u 9 || true
  log_msg "LOCK_RELEASED session=$S"
  RUN_OUTCOME="ok"
  RUN_WHY="completed CPU session-smoke check for $S"
  log_msg "SESSION_SMOKE_OK id=$RUN_ID session=$S"
  exit 0
fi

log_msg "WINDOW_START id=$RUN_ID mode=$MODE card=$CARD wall_s=$WALL_SECONDS vm_wall_s=$VM_WALL_SECONDS exec_timeout_s=$EXEC_TIMEOUT_SECONDS watchdog_sleep_s=$WATCHDOG_SLEEP_SECONDS rig_timeout_s=$RIG_TIMEOUT rate_uh=$RATE_UNITS_PER_HOUR cap_units=$CAP_UNITS writer=$WRITER count=$COUNT seed=$SEED grant_id=$GRANT_ID vram_halt_mib=$VRAM_HALT_MIB"

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
if [[ -x "$RECONCILE_SH" ]]; then
  COLAB_LOCK_DIR="$LOCK_DIR" "$RECONCILE_SH" --live | scrub_stream >> "$CLI_LOG" || true
fi
PRE_SESSIONS="$(run_colab sessions || true)"
if printf '%s\n' "$PRE_SESSIONS" | grep -Fq "[$SESSION_NAME]"; then
  flock -u 9 || true
  RUN_WHY="session $SESSION_NAME already exists in colab sessions; refusing to reuse name"
  log_msg "SESSION_NAME_IN_USE session=$SESSION_NAME"
  echo "SESSION_NAME_IN_USE: session [$SESSION_NAME] already exists in colab sessions" >&2
  exit 1
fi
if ! run_colab new "${COLAB_NEW_FLAGS[@]+"${COLAB_NEW_FLAGS[@]}"}" -s "$SESSION_NAME" >/dev/null; then
  flock -u 9 || true
  RUN_WHY="colab new failed for $SESSION_NAME"
  exit 1
fi
S="$SESSION_NAME"
NEW_SUCCEEDED=1

mkdir -p "$LOCK_DIR"
printf '%s\n' "$MAIN" > "$LOCK_DIR/$S.lock"
log_msg "SESSION_OPENED session=$S pre_sessions=$(printf '%s' "$PRE_SESSIONS" | tr -d '\n ')"

GATE_OUT="$(timeout "$GATE_TIMEOUT_SECONDS" colab exec -s "$S" --env UDLAB_STEP=gate -f "$SCRIPT_DIR/vm_udlab.py" --timeout "$GATE_TIMEOUT_SECONDS" 2>&1 | scrub_stream | tee -a "$CLI_LOG")" || {
  flock -u 9 || true
  RUN_WHY="VM liveness gate failed on session $S"
  exit 1
}
if ! printf '%s\n' "$GATE_OUT" | grep -q '^VM_GATE_OK'; then
  flock -u 9 || true
  RUN_WHY="VM liveness gate did not print VM_GATE_OK on session $S"
  exit 1
fi
flock -u 9 || true
log_msg "LOCK_RELEASED session=$S"

# Start independent rig-side deadline watchdog (sleeps WATCHDOG_SLEEP_SECONDS, fetches, then stops by WALL_SECONDS)
rm -rf "$DEADLINE_MARKER" "$FETCH_CLAIM_DIR" "$FETCH_DONE_MARKER" "$OUT_DIR/.rig_stopped" "$OUT_DIR/vm_status.json"
(
  SLEEP_PID=""
  trap 'if [[ -n "$SLEEP_PID" ]]; then kill "$SLEEP_PID" 2>/dev/null || true; fi; exit 0' TERM INT
  sleep "$WATCHDOG_SLEEP_SECONDS" &
  SLEEP_PID="$!"
  wait "$SLEEP_PID" 2>/dev/null || exit 0
  SLEEP_PID=""
  touch "$DEADLINE_MARKER"
  printf '%s RIG_DEADLINE_EXCEEDED session=%s wall_s=%s fetch_margin_s=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$S" "$WALL_SECONDS" "$FETCH_MARGIN" >> "$CLI_LOG"
  fetch_results_once "$WATCHDOG_FETCH_TIMEOUT"
  printf '%s WATCHDOG_STOP_SESSION session=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$S" >> "$CLI_LOG"
  colab stop -s "$S" >/dev/null 2>&1 || true
  rm -f "$LOCK_DIR/$S.lock"
  touch "$OUT_DIR/.rig_stopped"
) </dev/null >/dev/null 2>&1 9>&- &
WATCHDOG_PID="$!"

timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$REPO_TAR" "/content/repo_bundle.tar.gz" 2>&1 | scrub_stream >> "$CLI_LOG"
timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$PROMPTS_TAR" "/content/exact_ctx_prompts.tar.gz" 2>&1 | scrub_stream >> "$CLI_LOG"
cleanup_stage

UPLOADED_RESUME_TAR=""
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
    timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$RESUME_TAR" "/content/resume_state.tar.gz" 2>&1 | scrub_stream >> "$CLI_LOG"
    rm -f "$RESUME_TAR"
    UPLOADED_RESUME_TAR="/content/resume_state.tar.gz"
  fi
fi

EXEC_ENV_ARGS=(
  --env "UDLAB_STEP=run"
  --env "UDLAB_MODE=$MODE"
  --env "UDLAB_WRITER=$WRITER"
  --env "UDLAB_COUNT=$COUNT"
  --env "UDLAB_SEED=$SEED"
  --env "UDLAB_GRANT_ID=$GRANT_ID"
  --env "UDLAB_VRAM_HALT_MIB=$VRAM_HALT_MIB"
  --env "UDLAB_WALL_SECONDS=$VM_WALL_SECONDS"
  --env "UDLAB_REPO_DIR=/content/udlab"
  --env "UDLAB_REPO_TAR=/content/repo_bundle.tar.gz"
  --env "UDLAB_PROMPTS_DIR=/content/udlab_prompts"
  --env "UDLAB_PROMPTS_TAR=/content/exact_ctx_prompts.tar.gz"
  --env "UDLAB_RUN_DIR=/content/udlab_run"
  --env "UDLAB_RESULT_TAR=/content/result.tar.gz"
  --env "UDLAB_STATUS_FILE=/content/vm_status.json"
  --env "UDLAB_RESUME=$RESUME_FLAG"
)
if [[ -n "$UPLOADED_RESUME_TAR" ]]; then
  EXEC_ENV_ARGS+=(--env "UDLAB_RESUME_TAR=$UPLOADED_RESUME_TAR")
fi
if [[ "${VM_DRYRUN:-0}" == "1" ]]; then
  EXEC_ENV_ARGS+=(--env "VM_DRYRUN=1")
fi
if [[ -n "${UDLAB_FAKE_API_PS_JSON:-}" ]]; then
  EXEC_ENV_ARGS+=(--env "UDLAB_FAKE_API_PS_JSON=$UDLAB_FAKE_API_PS_JSON")
fi

EXEC_RC=0
timeout --signal=TERM --kill-after=15 "${RIG_TIMEOUT}s" \
  colab exec -s "$S" "${EXEC_ENV_ARGS[@]}" -f "$SCRIPT_DIR/vm_udlab.py" --timeout="$EXEC_TIMEOUT_SECONDS" 2>&1 | scrub_stream >> "$CLI_LOG" || EXEC_RC=$?

if [[ -f "$DEADLINE_MARKER" ]]; then
  if [[ -n "$WATCHDOG_PID" ]]; then
    wait "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""
  fi
  rm -f "$DEADLINE_MARKER"
  EXEC_RC=124
fi

fetch_results_once

if [[ "$EXEC_RC" -eq 124 ]] || [[ "$EXEC_RC" -eq 137 ]]; then
  RUN_WHY="wall-clock timeout exceeded (${WALL_SECONDS}s)"
  exit "$EXEC_RC"
fi

if [[ ! -f "$OUT_DIR/vm_status.json" ]]; then
  RUN_WHY="vm_status.json missing after VM run (exec_rc=$EXEC_RC)"
  if [[ "$EXEC_RC" -ne 0 ]]; then
    exit "$EXEC_RC"
  fi
  exit 1
fi

STATUS_CHECK_OUT=""
STATUS_CHECK_RC=0
STATUS_CHECK_OUT="$("$PYTHON_BIN" - "$OUT_DIR/vm_status.json" <<'PY'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
try:
    data = json.loads(status_path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"invalid_json|{type(exc).__name__}")
    raise SystemExit(2)

if not isinstance(data, dict):
    print("invalid_payload|not_a_dict")
    raise SystemExit(2)

step = str(data.get("step") or "unknown")
rc = data.get("rc")
reason = str(data.get("reason") or "")
if not isinstance(rc, int) or isinstance(rc, bool):
    print(f"{step}|invalid_rc|{reason}")
    raise SystemExit(2)

print(f"{step}|{rc}|{reason}")
if rc != 0:
    raise SystemExit(rc if 1 <= rc <= 255 else 2)
PY
)" || STATUS_CHECK_RC=$?

if [[ "$STATUS_CHECK_RC" -ne 0 ]]; then
  RUN_WHY="VM status file reported failure (${STATUS_CHECK_OUT})"
  exit "$STATUS_CHECK_RC"
fi

if [[ "$EXEC_RC" -ne 0 ]]; then
  RUN_WHY="colab exec exited non-zero ($EXEC_RC)"
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
if not receipt_file.is_file():
    raise SystemExit("vm_receipt.json missing inside result.tar.gz")
rdata = json.loads(receipt_file.read_text(encoding="utf-8"))
if rdata.get("status") != "ok":
    raise SystemExit(f"vm_receipt.json status is {rdata.get('status')!r}, expected 'ok'")
prompt_line_hashes = list(rdata.get("prompt_line_sha256s") or [])

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
