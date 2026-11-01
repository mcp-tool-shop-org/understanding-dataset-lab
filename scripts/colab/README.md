# Colab Scenario Writer Launcher (`scripts/colab`)

Per-window Google Colab launcher (`launch.sh`) and VM runner (`vm_udlab.py`) for the `understanding-dataset-lab` scenario writer.

## Usage

- **First-rung CPU session smoke** (opens `udlab-<run-id>`, writes reconciler lock, checks `colab sessions`, stops session in trap, verifies removal; no upload or model pull):
  ```bash
  bash scripts/colab/launch.sh --session-smoke --out-dir runs/smoke-window --run-id smoke-01
  ```
- **Dry command listing** (prints every `colab` command line in order with resolved session name, files, and timeouts):
  ```bash
  bash scripts/colab/launch.sh --stub --dry --out-dir runs/stub-window --run-id stub-window-01
  ```
- **CPU stub rung** (`--writer mock`, 7200s wall cap, 0.16 Colab compute units):
  ```bash
  UDLAB_EXACT_CTX_PROMPTS=/path/to/exact_ctx_prompts \
    bash scripts/colab/launch.sh --stub --out-dir runs/stub-window --run-id stub-window-01
  ```
- **GPU writer rung** (`--gpu A100`, `--writer ollama`, 5400s wall cap, 7.95 Colab compute units):
  ```bash
  UDLAB_EXACT_CTX_PROMPTS=/path/to/exact_ctx_prompts \
    bash scripts/colab/launch.sh --grant-id <grant-id> --out-dir runs/gpu-window --run-id gpu-window-01
  ```

## Limits

- **(c)** The log check matches prompt lines of 40 or more characters, after whitespace is collapsed. Shorter lines, and a prompt line split across two log lines, are not detected.
- **(d)** The prompts tar is on the VM's local disk from upload until the exact-count gate. If the wall timeout hard-kills the job in that span, the tar stays on the VM until the session is stopped, which the trap and watchdog always do.
- **(e)** Stub-rung disk behaviour: the CPU stub pulls the full model, and below 40 GiB free it refuses with `DISK_TOO_SMALL` and records the free bytes in the receipt (`free_disk_bytes` in `vm_receipt.json`).
