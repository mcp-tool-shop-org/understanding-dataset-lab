from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_COLAB_DIR = REPO_ROOT / "scripts" / "colab"
LAUNCH_SH = SCRIPTS_COLAB_DIR / "launch.sh"
VM_UDLAB_PY = SCRIPTS_COLAB_DIR / "vm_udlab.py"
CLI_CONTRACT_JSON = REPO_ROOT / "tests" / "fixtures" / "colab-cli-contract.json"
CLI_CONTRACT_SHA256 = "39704f50455ef401106938e9df78f4443acd12256e96c5cdf856e0d098a2250f"

if str(SCRIPTS_COLAB_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_COLAB_DIR))

import vm_udlab  # noqa: E402


def _to_wsl_path(path: Path) -> str:
    """Convert a Windows or POSIX path into a POSIX path usable inside bash."""
    raw = str(path.resolve()).replace("\\", "/")
    if len(raw) >= 2 and raw[1] == ":":
        drive = raw[0].lower()
        rest = raw[2:].lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return raw


def _make_synthetic_prompts_dir(target_dir: Path) -> dict[str, int]:
    """Create 4 synthetic prompt files and a matching index.json in target_dir."""
    prompts_dir = target_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    counts = vm_udlab.EXPECTED_EXACT_COUNTS
    entries = []
    sha_to_count: dict[str, int] = {}
    for idx, expected_count in enumerate(counts):
        fname = f"synthetic-{idx}.prompt.txt"
        body_text = f"synthetic prompt #{idx} for count {expected_count}\n"
        sha = vm_udlab.compute_prompt_messages_sha256(body_text)
        (prompts_dir / fname).write_text(body_text, encoding="utf-8", newline="\n")
        entries.append(
            {
                "run": "synthetic-run",
                "call_id": f"scen_000{idx}:r0:call_0001",
                "file": fname,
                "prompt_sha256": sha,
                "prompt_eval_count": expected_count,
                "think": False,
            }
        )
        sha_to_count[sha] = expected_count

    index_path = target_dir / "index.json"
    index_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8", newline="\n")
    return sha_to_count


def _make_synthetic_prompts_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create a synthetic 4-prompt directory and index.json for fake-CLI tests."""
    fixture_dir = tmp_path / "synthetic_exact_ctx"
    _make_synthetic_prompts_dir(fixture_dir)
    return fixture_dir / "prompts", fixture_dir / "index.json"


def test_scripts_utf8_lf_zero_cr_and_bash_syntax() -> None:
    for script in (LAUNCH_SH, VM_UDLAB_PY):
        raw = script.read_bytes()
        assert b"\r" not in raw, f"{script.name} must have 0 CR bytes"
        raw.decode("utf-8")

    proc = subprocess.run(
        ["bash", "-n", _to_wsl_path(LAUNCH_SH)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_ollama_archive_and_version_refusal(tmp_path: Path) -> None:
    archive = tmp_path / "ollama-linux-amd64.tar.zst"
    archive.write_bytes(b"fake-ollama-archive-bytes")
    actual_sha = vm_udlab.sha256_file(archive)

    with pytest.raises(vm_udlab.RefusalError) as exc_info:
        vm_udlab.verify_ollama_archive_and_version(
            archive_path=archive,
            version_output="ollama version is 0.40.13",
            expected_sha256="0" * 64,
            expected_version="0.40.13",
        )
    assert exc_info.value.failure_class == "OLLAMA_ARCHIVE_SHA256_MISMATCH"

    with pytest.raises(vm_udlab.RefusalError) as exc_info2:
        vm_udlab.verify_ollama_archive_and_version(
            archive_path=archive,
            version_output="ollama version is 0.40.12",
            expected_sha256=actual_sha,
            expected_version="0.40.13",
        )
    assert exc_info2.value.failure_class == "OLLAMA_VERSION_MISMATCH"

    sha_ok, ver_ok = vm_udlab.verify_ollama_archive_and_version(
        archive_path=archive,
        version_output="ollama version is 0.40.13\n",
        expected_sha256=actual_sha,
        expected_version="0.40.13",
    )
    assert sha_ok == actual_sha
    assert "0.40.13" in ver_ok


def test_writer_model_digest_mismatch_refusal(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    manifest_dir = (
        models_root / "manifests" / "registry.ollama.ai" / "library" / "qwen3.6"
    )
    blobs_dir = models_root / "blobs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir.mkdir(parents=True, exist_ok=True)

    blob_bytes = b"synthetic-gguf-layer-bytes"
    blob_hex = hashlib.sha256(blob_bytes).hexdigest()
    blob_name = f"sha256-{blob_hex}"
    blob_path = blobs_dir / blob_name
    blob_path.write_bytes(blob_bytes)

    manifest_obj = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": f"sha256:{blob_hex}",
                "size": len(blob_bytes),
            }
        ],
    }
    manifest_bytes = json.dumps(manifest_obj).encode("utf-8")
    manifest_hex = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_path = manifest_dir / "35b"
    manifest_path.write_bytes(manifest_bytes)

    # 1. Wrong expected manifest digest -> MANIFEST_DIGEST_MISMATCH
    with pytest.raises(vm_udlab.RefusalError) as exc_manifest:
        vm_udlab.verify_writer_model_digests(
            models_root=models_root,
            model="qwen3.6:35b",
            expected_manifest_digest="f" * 64,
            expected_layer_blob=blob_name,
        )
    assert exc_manifest.value.failure_class == "MANIFEST_DIGEST_MISMATCH"

    # 2. Wrong expected model layer blob -> MODEL_LAYER_DIGEST_MISMATCH
    with pytest.raises(vm_udlab.RefusalError) as exc_layer:
        vm_udlab.verify_writer_model_digests(
            models_root=models_root,
            model="qwen3.6:35b",
            expected_manifest_digest=manifest_hex,
            expected_layer_blob="sha256-" + ("a" * 64),
        )
    assert exc_layer.value.failure_class == "MODEL_LAYER_DIGEST_MISMATCH"

    # 3. Corrupted blob contents -> MODEL_LAYER_DIGEST_MISMATCH
    blob_path.write_bytes(b"tampered-gguf-bytes")
    with pytest.raises(vm_udlab.RefusalError) as exc_corrupt:
        vm_udlab.verify_writer_model_digests(
            models_root=models_root,
            model="qwen3.6:35b",
            expected_manifest_digest=manifest_hex,
            expected_layer_blob=blob_name,
            verify_blob_sha256=True,
        )
    assert exc_corrupt.value.failure_class == "MODEL_LAYER_DIGEST_MISMATCH"

    # 4. Exact match passes
    blob_path.write_bytes(blob_bytes)
    got_manifest, got_blob, got_path = vm_udlab.verify_writer_model_digests(
        models_root=models_root,
        model="qwen3.6:35b",
        expected_manifest_digest=manifest_hex,
        expected_layer_blob=blob_name,
        verify_blob_sha256=True,
    )
    assert got_manifest == manifest_hex
    assert got_blob == blob_name
    assert got_path == blob_path


def test_exact_count_gate_pass_and_wrong_count_refusal(tmp_path: Path) -> None:
    # Case 1: exact match passes and deletes all .prompt.txt files
    case1 = tmp_path / "case1"
    sha_to_count = _make_synthetic_prompts_dir(case1)
    prompts_dir1 = case1 / "prompts"
    assert len(list(prompts_dir1.glob("*.prompt.txt"))) == 4

    verified = vm_udlab.run_exact_count_gate(
        index_path=case1 / "index.json",
        prompts_dir=prompts_dir1,
        count_fn=lambda txt: sha_to_count[vm_udlab.compute_prompt_messages_sha256(txt)],
    )
    assert [v["actual_count"] for v in verified] == [3124, 3423, 3393, 3394]
    assert not list(prompts_dir1.glob("*.prompt.txt"))

    # Case 2: wrong token count on entry #2 (3393 -> 3390) refuses and still deletes prompts
    case2 = tmp_path / "case2"
    sha_to_count2 = _make_synthetic_prompts_dir(case2)
    prompts_dir2 = case2 / "prompts"

    def _wrong_counter(txt: str) -> int:
        sha = vm_udlab.compute_prompt_messages_sha256(txt)
        val = sha_to_count2[sha]
        return 3390 if val == 3393 else val

    with pytest.raises(vm_udlab.RefusalError) as exc_cnt:
        vm_udlab.run_exact_count_gate(
            index_path=case2 / "index.json",
            prompts_dir=prompts_dir2,
            count_fn=_wrong_counter,
        )
    assert exc_cnt.value.failure_class == "EXACT_COUNT_TOKEN_MISMATCH"
    assert not list(prompts_dir2.glob("*.prompt.txt"))

    # Case 3: prompt sha256 mismatch refuses and still deletes prompts
    case3 = tmp_path / "case3"
    sha_to_count3 = _make_synthetic_prompts_dir(case3)
    prompts_dir3 = case3 / "prompts"
    (prompts_dir3 / "synthetic-1.prompt.txt").write_bytes(b"corrupted prompt bytes")

    with pytest.raises(vm_udlab.RefusalError) as exc_sha:
        vm_udlab.run_exact_count_gate(
            index_path=case3 / "index.json",
            prompts_dir=prompts_dir3,
            count_fn=lambda txt: sha_to_count3.get(
                vm_udlab.compute_prompt_messages_sha256(txt), 3124
            ),
        )
    assert exc_sha.value.failure_class == "EXACT_COUNT_SHA256_MISMATCH"
    assert not list(prompts_dir3.glob("*.prompt.txt"))


def test_tar_allowlist_accepts_valid_and_refuses_outside_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dir"
    (run_dir / "sealed_store").mkdir(parents=True)
    (run_dir / "part_public").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)

    for rel in vm_udlab.TAR_ALLOWED_EXACT_FILES:
        p = run_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8", newline="\n")
    (run_dir / "logs" / "vm.log").write_text("ok\n", encoding="utf-8", newline="\n")
    (run_dir / "logs" / "udlab-gen.log").write_text("ok\n", encoding="utf-8", newline="\n")

    tar_ok = tmp_path / "result_ok.tar.gz"
    packed = vm_udlab.build_allowlisted_result_tar(run_dir, tar_ok)
    assert "sealed_store/responses.sealed.jsonl" in packed
    assert "generations.jsonl" in packed
    assert "vm_receipt.json" in packed

    # Adding a disallowed file inside run_dir must be refused
    disallowed = run_dir / "unsealed_prompt.prompt.txt"
    disallowed.write_text("forbidden", encoding="utf-8", newline="\n")
    with pytest.raises(vm_udlab.RefusalError) as exc_tar:
        vm_udlab.build_allowlisted_result_tar(run_dir, tmp_path / "result_bad.tar.gz")
    assert exc_tar.value.failure_class == "TAR_ALLOWLIST_VIOLATION"
    disallowed.unlink()

    # A crafted tar archive containing a file outside the allow-list must be refused
    crafted_tar = tmp_path / "crafted_bad.tar.gz"
    extra_file = tmp_path / "leaked_draft.json"
    extra_file.write_text("{}", encoding="utf-8", newline="\n")
    with tarfile.open(crafted_tar, "w:gz") as tf:
        tf.add(run_dir / "generations.jsonl", arcname="generations.jsonl")
        tf.add(extra_file, arcname="sealed_store/unexpected_leak.json")

    with pytest.raises(vm_udlab.RefusalError) as exc_crafted:
        vm_udlab.validate_tar_file(crafted_tar)
    assert exc_crafted.value.failure_class == "TAR_ALLOWLIST_VIOLATION"


def test_log_scrubber_and_policy_verifier(tmp_path: Path) -> None:
    raw_line = (
        "Connecting to https://localhost:8080/?colab-runtime-proxy-token=sec_ABC123xyz&foo=1\n"
        "Header colab-runtime-proxy-token=anotherSecretToken999; done\n"
    )
    scrubbed = vm_udlab.scrub_log_text(raw_line)
    assert "sec_ABC123xyz" not in scrubbed
    assert "anotherSecretToken999" not in scrubbed
    assert scrubbed.count("colab-runtime-proxy-token=[REDACTED]") == 2

    clean_log = tmp_path / "clean.log"
    clean_log.write_text(scrubbed, encoding="utf-8", newline="\n")
    vm_udlab.verify_logs_clean([clean_log])

    # 1. Line longer than 2000 chars fails the window
    long_log = tmp_path / "long.log"
    long_log.write_text("x" * 2001 + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(vm_udlab.RefusalError) as exc_long:
        vm_udlab.verify_logs_clean([long_log])
    assert exc_long.value.failure_class == "LOG_POLICY_VIOLATION"

    # 2. Key response_text fails the window
    resp_log = tmp_path / "resp.log"
    resp_log.write_text('{"item": 0, "response_text": "secret"}\n', encoding="utf-8", newline="\n")
    with pytest.raises(vm_udlab.RefusalError) as exc_resp:
        vm_udlab.verify_logs_clean([resp_log])
    assert exc_resp.value.failure_class == "LOG_POLICY_VIOLATION"

    # 3. Unredacted colab-runtime-proxy-token= fails the window
    token_log = tmp_path / "token.log"
    token_log.write_text("url?colab-runtime-proxy-token=leaked123\n", encoding="utf-8", newline="\n")
    with pytest.raises(vm_udlab.RefusalError) as exc_tok:
        vm_udlab.verify_logs_clean([token_log])
    assert exc_tok.value.failure_class == "LOG_POLICY_VIOLATION"


def _setup_fake_colab_env(tmp_path: Path) -> dict[str, Path]:
    """Create a front-of-PATH fake `colab` CLI enforcing colab-cli-contract.json and a later-on-PATH real-colab sentinel."""
    fake_bin = tmp_path / "fake_bin"
    sentinel_bin = tmp_path / "sentinel_real_bin"
    state_dir = tmp_path / "fake_colab_state"
    colab_locks_dir = tmp_path / "colab_locks"
    fake_bin.mkdir(parents=True, exist_ok=True)
    sentinel_bin.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    colab_locks_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate a neighbour session that must never be stopped
    sessions_file = state_dir / "sessions.json"
    sessions_file.write_text(
        json.dumps([{"session_id": "sess-neighbour-keep", "gpu": "T4"}]),
        encoding="utf-8",
        newline="\n",
    )

    sentinel_called = state_dir / "REAL_COLAB_CALLED"
    sentinel_script = sentinel_bin / "colab"
    sentinel_script.write_text(
        f"#!/usr/bin/env bash\ntouch '{_to_wsl_path(sentinel_called)}'\nexit 99\n",
        encoding="utf-8",
        newline="\n",
    )
    sentinel_script.chmod(sentinel_script.stat().st_mode | stat.S_IEXEC)

    contract_path_posix = _to_wsl_path(CLI_CONTRACT_JSON)
    fake_driver_py = state_dir / "fake_colab_driver.py"
    fake_driver_py.write_text(
        f"""#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONTRACT_PATH = Path({contract_path_posix!r})
contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
commands_contract = contract["commands"]
global_options = set(contract.get("global_options", []))

state_dir = Path(os.environ["FAKE_COLAB_STATE_DIR"])
vm_fs = state_dir / "vm_fs"
vm_fs.mkdir(parents=True, exist_ok=True)
calls_log = state_dir / "calls.jsonl"
sessions_file = state_dir / "sessions.json"
lock_file = Path(os.environ.get("UDLAB_OPEN_LOCK", "/tmp/colab-window-open.lock"))
reconcile_lock_dir = Path(os.environ.get("COLAB_LOCK_DIR", "/tmp/colab-locks"))

raw_args = sys.argv[1:]

# Parse optional global flags before subcommand
idx = 0
GLOBAL_VALUE_FLAGS = {{"--auth", "--client-oauth-config", "-c", "--config"}}
top_help_requested = False
while idx < len(raw_args) and raw_args[idx].startswith("-"):
    gopt = raw_args[idx].split("=", 1)[0]
    if gopt not in global_options:
        sys.stderr.write(f"No such global option: {{raw_args[idx]}}\\n")
        raise SystemExit(2)
    if gopt in ("-h", "--help"):
        top_help_requested = True
    if gopt in GLOBAL_VALUE_FLAGS and "=" not in raw_args[idx]:
        idx += 2
    else:
        idx += 1

if top_help_requested and idx >= len(raw_args):
    print("Usage: colab [OPTIONS] COMMAND [ARGS]...")
    print(f"Options: {{' '.join(sorted(global_options))}}")
    print(f"Commands: {{' '.join(sorted(commands_contract.keys()))}}")
    raise SystemExit(0)

if idx >= len(raw_args):
    sys.stderr.write("Missing command.\\n")
    raise SystemExit(2)

subcmd = raw_args[idx]
args = raw_args[idx + 1:]

if subcmd == "version":
    print(f"Version: {{contract['version'].lstrip('v')}}")
    raise SystemExit(0)

if subcmd not in commands_contract:
    sys.stderr.write(f"No such command '{{subcmd}}' in {{contract['cli']}} {{contract['version']}}.\\n")
    raise SystemExit(2)

allowed_opts = set(commands_contract[subcmd]["options"])
VALUE_OPTS_BY_CMD = {{
    "new": {{"-s", "--session", "--gpu", "--tpu"}},
    "exec": {{"-s", "--session", "-f", "--file", "--timeout", "--env", "--output-image"}},
    "upload": {{"-s", "--session"}},
    "download": {{"-s", "--session"}},
    "stop": {{"-s", "--session"}},
    "sessions": set(),
    "usage": set(),
    "status": {{"-s", "--session"}},
}}
value_opts = VALUE_OPTS_BY_CMD.get(subcmd, set())

parsed_opts: dict[str, list[str]] = {{}}
positionals: list[str] = []
i = 0
while i < len(args):
    tok = args[i]
    if tok == "--":
        sys.stderr.write(f"colab {{subcmd}} --: not an option of {{contract['cli']}} {{contract['version']}} (trailing shell commands are rejected)\\n")
        raise SystemExit(2)
    if tok.startswith("-") and len(tok) > 1:
        if "=" in tok:
            opt_name, opt_val = tok.split("=", 1)
            if opt_name not in allowed_opts:
                sys.stderr.write(f"colab {{subcmd}} {{opt_name}}: not an option of {{contract['cli']}} {{contract['version']}} (allowed: {{' '.join(sorted(allowed_opts))}})\\n")
                raise SystemExit(2)
            parsed_opts.setdefault(opt_name, []).append(opt_val)
            i += 1
        else:
            opt_name = tok
            if opt_name not in allowed_opts:
                sys.stderr.write(f"colab {{subcmd}} {{opt_name}}: not an option of {{contract['cli']}} {{contract['version']}} (allowed: {{' '.join(sorted(allowed_opts))}})\\n")
                raise SystemExit(2)
            if opt_name in value_opts:
                if i + 1 >= len(args):
                    sys.stderr.write(f"Option '{{opt_name}}' requires an argument.\\n")
                    raise SystemExit(2)
                parsed_opts.setdefault(opt_name, []).append(args[i + 1])
                i += 2
            else:
                parsed_opts.setdefault(opt_name, []).append("true")
                i += 1
    else:
        positionals.append(tok)
        i += 1

if "-h" in parsed_opts or "--help" in parsed_opts:
    print(f"Usage: colab {{subcmd}} [OPTIONS]")
    print(f"Options: {{' '.join(sorted(allowed_opts))}}")
    raise SystemExit(0)

def _get_opt(*names: str) -> str | None:
    for n in names:
        vals = parsed_opts.get(n)
        if vals:
            return vals[-1]
    return None

# Check whether the open lock is currently held by the caller
lock_held = False
if lock_file.exists():
    try:
        res = subprocess.run(["flock", "-n", str(lock_file), "true"], check=False)
        lock_held = (res.returncode != 0)
    except Exception:
        lock_held = False

reconcile_locks_present = sorted(p.name for p in reconcile_lock_dir.glob("*.lock")) if reconcile_lock_dir.is_dir() else []

try:
    stdin_is_devnull = os.path.samefile("/proc/self/fd/0", "/dev/null")
except Exception:
    try:
        s0 = os.fstat(0)
        sn = os.stat("/dev/null")
        stdin_is_devnull = (s0.st_rdev == sn.st_rdev and s0.st_rdev != 0)
    except Exception:
        stdin_is_devnull = False

if os.environ.get("FAKE_COLAB_READ_STDIN") == "1":
    _ = sys.stdin.read()

record = {{
    "subcmd": subcmd,
    "args": args,
    "lock_held": lock_held,
    "reconcile_locks": reconcile_locks_present,
    "stdin_is_devnull": stdin_is_devnull,
    "ts": time.time(),
}}
with calls_log.open("a", encoding="utf-8", newline="\\n") as fh:
    fh.write(json.dumps(record) + "\\n")

# Always emit a proxy token in stderr to verify launch.sh scrubs CLI output
sys.stderr.write("debug proxy https://colab.research.google.com/?colab-runtime-proxy-token=FAKE_SECRET_TOKEN_777\\n")

sessions = json.loads(sessions_file.read_text(encoding="utf-8"))

if subcmd == "sessions":
    if positionals:
        sys.stderr.write(f"Got unexpected extra arguments ({{' '.join(positionals)}})\\n")
        raise SystemExit(2)
    override_out = os.environ.get("FAKE_COLAB_SESSIONS_OVERRIDE")
    if override_out is not None:
        print(override_out)
        raise SystemExit(0)
    prefix_line = os.environ.get("FAKE_COLAB_SESSIONS_PREFIX_LINE")
    if not sessions:
        print(prefix_line if prefix_line is not None else "[colab] No active sessions found on server.")
        raise SystemExit(0)
    if prefix_line is not None:
        print(prefix_line)
    row_tpl = os.environ.get("FAKE_COLAB_SESSIONS_TEMPLATE", "[{{sid}}] m-s-fake-{{sid}} | {{gpu}}")
    for s in sessions:
        sid = s.get("session_id", "?")
        gpu = s.get("gpu", "CPU")
        print(row_tpl.format(sid=sid, gpu=gpu))
    raise SystemExit(0)

if subcmd == "new":
    if positionals:
        sys.stderr.write(f"Got unexpected extra arguments ({{' '.join(positionals)}})\\n")
        raise SystemExit(2)
    sid = _get_opt("-s", "--session") or "sess-udlab-test-001"
    if os.environ.get("FAKE_COLAB_FAIL_NEW") == "1":
        sys.stderr.write(f"Failed to create session [{{sid}}]\\n")
        raise SystemExit(1)
    if any(s.get("session_id") == sid for s in sessions):
        sys.stderr.write(f"Session [{{sid}}] already exists\\n")
        raise SystemExit(1)
    gpu = _get_opt("--gpu") or "CPU"
    sessions.append({{"session_id": sid, "gpu": gpu}})
    sessions_file.write_text(json.dumps(sessions), encoding="utf-8", newline="\\n")
    print(f"Created session [{{sid}}] ({{gpu}})")
    raise SystemExit(0)

if subcmd == "upload":
    sid = _get_opt("-s", "--session")
    if not sid or len(positionals) != 2:
        sys.stderr.write("Usage: colab upload -s SESSION <local_path> <remote_path>\\n")
        raise SystemExit(2)
    src, dst = positionals[0], positionals[1]
    if not any(s.get("session_id") == sid for s in sessions):
        sys.stderr.write(f"Session {{sid}} not found\\n")
        raise SystemExit(1)
    if not dst.startswith("/content/"):
        sys.stderr.write(f"Remote path must be under /content/: {{dst}}\\n")
        raise SystemExit(1)
    vm_dst = vm_fs / dst.removeprefix("/content/")
    vm_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, vm_dst)
    print(f"Uploaded {{src}} -> {{dst}}")
    raise SystemExit(0)

if subcmd == "download":
    sid = _get_opt("-s", "--session")
    if not sid or len(positionals) != 2:
        sys.stderr.write("Usage: colab download -s SESSION <remote_path> <local_path>\\n")
        raise SystemExit(2)
    src, dst = positionals[0], positionals[1]
    if not any(s.get("session_id") == sid for s in sessions):
        sys.stderr.write(f"Session {{sid}} not found\\n")
        raise SystemExit(1)
    if not src.startswith("/content/"):
        sys.stderr.write(f"Remote path must be under /content/: {{src}}\\n")
        raise SystemExit(1)
    vm_src = vm_fs / src.removeprefix("/content/")
    fetch_sleep_s = float(os.environ.get("FAKE_COLAB_FETCH_SLEEP", "0"))
    if fetch_sleep_s > 0 and src == "/content/result.tar.gz":
        if vm_src.is_file():
            raw_bytes = vm_src.read_bytes()
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            Path(dst).write_bytes(raw_bytes[: max(1, len(raw_bytes) // 2)])
        time.sleep(fetch_sleep_s)
        sessions_now = json.loads(sessions_file.read_text(encoding="utf-8"))
        if not any(s.get("session_id") == sid for s in sessions_now):
            raise SystemExit(1)
    if not vm_src.is_file():
        sys.stderr.write(f"Remote file not found: {{src}}\\n")
        raise SystemExit(1)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vm_src, dst)
    print(f"Downloaded {{src}} -> {{dst}}")
    raise SystemExit(0)

if subcmd == "exec":
    if positionals:
        sys.stderr.write(f"colab exec does not take positional shell arguments ({{' '.join(positionals)}})\\n")
        raise SystemExit(2)
    sid = _get_opt("-s", "--session")
    py_file = _get_opt("-f", "--file")
    if not sid or not py_file:
        sys.stderr.write("colab exec requires -s/--session and -f/--file\\n")
        raise SystemExit(2)
    if not any(s.get("session_id") == sid for s in sessions):
        sys.stderr.write(f"Session {{sid}} not found\\n")
        raise SystemExit(1)
    py_path = Path(py_file)
    if not py_path.is_file() or py_path.suffix != ".py":
        sys.stderr.write(f"colab exec -f must point to an existing Python (.py) file: {{py_file}}\\n")
        raise SystemExit(2)
    timeout_raw = _get_opt("--timeout")
    timeout_s = float(timeout_raw) if timeout_raw is not None else 30.0
    if timeout_s <= 0:
        sys.stderr.write(f"Invalid --timeout={{timeout_raw}}\\n")
        raise SystemExit(2)

    exec_env = os.environ.copy()
    for kv in parsed_opts.get("--env", []):
        if "=" not in kv:
            sys.stderr.write(f"Invalid --env value (expected KEY=VALUE): {{kv}}\\n")
            raise SystemExit(2)
        k, v = kv.split("=", 1)
        if v.startswith("/content/"):
            v = str(vm_fs / v.removeprefix("/content/"))
        elif v == "/content":
            v = str(vm_fs)
        exec_env[k] = v

    sleep_s = float(os.environ.get("FAKE_COLAB_EXEC_SLEEP", "0"))
    step = exec_env.get("UDLAB_STEP", "run")
    try:
        if step == "run" and sleep_s > 0:
            time.sleep(sleep_s)
            sys.stderr.write(f"Execution timed out after {{timeout_s}}s\\n")
            raise SystemExit(124)
        kernel_code = (
            "import sys,runpy; "
            "sys.argv=['colab_kernel_launcher.py','-f','/tmp/kernel-fake.json']; "
            f"runpy.run_path({{str(py_path)!r}}, run_name='__main__')"
        )
        proc = subprocess.run(
            ["python3", "-c", kernel_code],
            stdin=subprocess.DEVNULL,
            env=exec_env,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"Execution timed out after {{timeout_s}}s\\n")
        raise SystemExit(124)
    if os.environ.get("FAKE_COLAB_DELETE_STATUS_FILE") == "1":
        status_env = exec_env.get("UDLAB_STATUS_FILE")
        if status_env and Path(status_env).is_file():
            Path(status_env).unlink()
    raise SystemExit(proc.returncode)

if subcmd == "stop":
    if positionals:
        sys.stderr.write(f"Got unexpected extra arguments ({{' '.join(positionals)}})\\n")
        raise SystemExit(2)
    sid = _get_opt("-s", "--session")
    if not sid:
        sys.stderr.write("colab stop requires -s/--session\\n")
        raise SystemExit(2)
    stopped_file = state_dir / "stopped_sessions.jsonl"
    with stopped_file.open("a", encoding="utf-8", newline="\\n") as fh:
        fh.write(json.dumps({{"session_id": sid, "ts": time.time()}}) + "\\n")
    sessions = [s for s in sessions if s.get("session_id") != sid]
    sessions_file.write_text(json.dumps(sessions), encoding="utf-8", newline="\\n")
    print(f"Stopped session [{{sid}}]")
    raise SystemExit(0)

if subcmd == "status":
    sid = _get_opt("-s", "--session")
    print(f"Session [{{sid or '?'}}] Status: IDLE")
    raise SystemExit(0)

if subcmd == "usage":
    print("Compute units balance: 100.0 | Usage rate: 0.08/hr | Active assignments: 1")
    raise SystemExit(0)

raise SystemExit(2)
""",
        encoding="utf-8",
        newline="\n",
    )

    fake_colab_sh = fake_bin / "colab"
    fake_colab_sh.write_text(
        f"#!/usr/bin/env bash\nexport UDLAB_COLAB_IS_FAKE=1\nexec python3 '{_to_wsl_path(fake_driver_py)}' \"$@\"\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_colab_sh.chmod(fake_colab_sh.stat().st_mode | stat.S_IEXEC)

    os.environ["COLAB_RECONCILE_SH"] = _to_wsl_path(state_dir / "nonexistent_reconcile.sh")
    os.environ["UDLAB_COLAB_IS_FAKE"] = "1"
    os.environ["WSLENV"] = "COLAB_RECONCILE_SH/u:UDLAB_COLAB_IS_FAKE"

    return {
        "fake_bin": fake_bin,
        "sentinel_bin": sentinel_bin,
        "state_dir": state_dir,
        "colab_locks_dir": colab_locks_dir,
        "sentinel_called": sentinel_called,
        "calls_log": state_dir / "calls.jsonl",
        "sessions_file": sessions_file,
        "stopped_file": state_dir / "stopped_sessions.jsonl",
    }


def test_launch_sh_stub_trap_ordering_and_outcome_with_fake_colab(tmp_path: Path) -> None:
    env_paths = _setup_fake_colab_env(tmp_path)
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)
    out_dir = tmp_path / "run_stub_ok"
    lock_file = tmp_path / "colab-open.lock"

    wsl_fake_bin = _to_wsl_path(env_paths["fake_bin"])
    wsl_sentinel_bin = _to_wsl_path(env_paths["sentinel_bin"])
    wsl_state_dir = _to_wsl_path(env_paths["state_dir"])
    wsl_locks_dir = _to_wsl_path(env_paths["colab_locks_dir"])
    wsl_prompts_dir = _to_wsl_path(prompts_dir)
    wsl_index_json = _to_wsl_path(index_json)
    wsl_out_dir = _to_wsl_path(out_dir)
    wsl_lock_file = _to_wsl_path(lock_file)
    wsl_launch = _to_wsl_path(LAUNCH_SH)

    cmd = (
        f"export PATH='{wsl_fake_bin}:{wsl_sentinel_bin}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{wsl_state_dir}'; "
        f"export COLAB_LOCK_DIR='{wsl_locks_dir}'; "
        f"export UDLAB_OPEN_LOCK='{wsl_lock_file}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{wsl_prompts_dir}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{wsl_index_json}'; "
        f"export VM_DRYRUN=1; "
        f"bash '{wsl_launch}' --stub --out-dir '{wsl_out_dir}' --run-id test-stub-window"
    )
    proc = subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    # Real colab sentinel was never called
    assert not env_paths["sentinel_called"].exists()

    # Check call ordering in fake colab
    calls = [
        json.loads(line)
        for line in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()
    ]
    subcmds = [c["subcmd"] for c in calls]
    assert "new" in subcmds
    assert "stop" in subcmds

    # Lock was held during `new` and initial VM gate `exec`, then released before main run
    new_call = next(c for c in calls if c["subcmd"] == "new")
    assert new_call["lock_held"] is True

    # Reconciler lock was present during gate exec and removed after stop
    gate_call = next(c for c in calls if c["subcmd"] == "exec")
    assert "udlab-test-stub-window.lock" in gate_call["reconcile_locks"]
    assert not (env_paths["colab_locks_dir"] / "udlab-test-stub-window.lock").exists()

    # Result fetch (`download` from /content/result.tar.gz) happened BEFORE `stop`
    stop_idx = next(i for i, c in enumerate(calls) if c["subcmd"] == "stop")
    fetch_idx = next(
        i
        for i, c in enumerate(calls)
        if c["subcmd"] == "download" and "/content/result.tar.gz" in c["args"]
    )
    assert fetch_idx < stop_idx

    # Only own session was stopped; neighbour session remains active
    stopped = [
        json.loads(line)["session_id"]
        for line in env_paths["stopped_file"].read_text(encoding="utf-8").splitlines()
    ]
    assert stopped == ["udlab-test-stub-window"]
    remaining_sessions = json.loads(env_paths["sessions_file"].read_text(encoding="utf-8"))
    assert [s["session_id"] for s in remaining_sessions] == ["sess-neighbour-keep"]

    # Proxy token was scrubbed from colab-cli.log
    cli_log_text = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "FAKE_SECRET_TOKEN_777" not in cli_log_text
    assert "colab-runtime-proxy-token=[REDACTED]" in cli_log_text

    # Prompt files on VM were deleted by exact-count gate before writer call
    vm_prompts_dir = env_paths["state_dir"] / "vm_fs" / "udlab_prompts"
    assert not list(vm_prompts_dir.glob("*.prompt.txt"))

    # Verify outcome.draft.json fields required by Publisher
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    for key in ("id", "venue", "hardware", "cap", "runtime_seconds", "outcome", "why"):
        assert key in outcome
    assert outcome["id"] == "test-stub-window"
    assert outcome["venue"] == "colab"
    assert outcome["hardware"] == "CPU"
    assert outcome["cap"] == 0.16
    assert outcome["outcome"] == "ok"


def test_launch_sh_trap_stops_session_on_refusal_failure(tmp_path: Path) -> None:
    env_paths = _setup_fake_colab_env(tmp_path)
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)
    out_dir = tmp_path / "run_gpu_fail"
    lock_file = tmp_path / "colab-open-fail.lock"

    wsl_fake_bin = _to_wsl_path(env_paths["fake_bin"])
    wsl_sentinel_bin = _to_wsl_path(env_paths["sentinel_bin"])
    wsl_state_dir = _to_wsl_path(env_paths["state_dir"])
    wsl_prompts_dir = _to_wsl_path(prompts_dir)
    wsl_index_json = _to_wsl_path(index_json)
    wsl_out_dir = _to_wsl_path(out_dir)
    wsl_lock_file = _to_wsl_path(lock_file)
    wsl_launch = _to_wsl_path(LAUNCH_SH)

    cmd = (
        f"export PATH='{wsl_fake_bin}:{wsl_sentinel_bin}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{wsl_state_dir}'; "
        f"export UDLAB_OPEN_LOCK='{wsl_lock_file}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{wsl_prompts_dir}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{wsl_index_json}'; "
        f"export VM_DRYRUN=1; "
        f"export UDLAB_FAKE_API_PS_JSON='{{\"models\":[]}}'; "
        f"bash '{wsl_launch}' --grant-id test-grant-01 --out-dir '{wsl_out_dir}' --run-id test-gpu-refusal"
    )
    proc = subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode != 0
    assert not env_paths["sentinel_called"].exists()

    # Session was still stopped in trap, and neighbour session was untouched
    stopped = [
        json.loads(line)["session_id"]
        for line in env_paths["stopped_file"].read_text(encoding="utf-8").splitlines()
    ]
    assert stopped == ["udlab-test-gpu-refusal"]
    remaining_sessions = json.loads(env_paths["sessions_file"].read_text(encoding="utf-8"))
    assert [s["session_id"] for s in remaining_sessions] == ["sess-neighbour-keep"]

    # GPU defaults (A100, cap 7.95) and fail outcome recorded in outcome.draft.json
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["id"] == "test-gpu-refusal"
    assert outcome["venue"] == "colab"
    assert outcome["hardware"] == "A100"
    assert outcome["cap"] == 7.95
    assert outcome["outcome"] == "fail"


def test_launch_sh_wall_timeout_stops_own_session_and_writes_outcome(tmp_path: Path) -> None:
    env_paths = _setup_fake_colab_env(tmp_path)
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)
    out_dir = tmp_path / "run_wall_timeout"
    lock_file = tmp_path / "colab-open-timeout.lock"

    wsl_fake_bin = _to_wsl_path(env_paths["fake_bin"])
    wsl_sentinel_bin = _to_wsl_path(env_paths["sentinel_bin"])
    wsl_state_dir = _to_wsl_path(env_paths["state_dir"])
    wsl_prompts_dir = _to_wsl_path(prompts_dir)
    wsl_index_json = _to_wsl_path(index_json)
    wsl_out_dir = _to_wsl_path(out_dir)
    wsl_lock_file = _to_wsl_path(lock_file)
    wsl_launch = _to_wsl_path(LAUNCH_SH)

    cmd = (
        f"export PATH='{wsl_fake_bin}:{wsl_sentinel_bin}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{wsl_state_dir}'; "
        f"export UDLAB_OPEN_LOCK='{wsl_lock_file}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{wsl_prompts_dir}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{wsl_index_json}'; "
        f"export VM_DRYRUN=1; "
        f"export UDLAB_WALL_SECONDS=1; "
        f"export FAKE_COLAB_EXEC_SLEEP=5; "
        f"bash '{wsl_launch}' --stub --out-dir '{wsl_out_dir}' --run-id test-wall-timeout"
    )
    proc = subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode != 0
    assert not env_paths["sentinel_called"].exists()

    stopped = [
        json.loads(line)["session_id"]
        for line in env_paths["stopped_file"].read_text(encoding="utf-8").splitlines()
    ]
    assert "udlab-test-wall-timeout" in stopped
    remaining_sessions = json.loads(env_paths["sessions_file"].read_text(encoding="utf-8"))
    assert [s["session_id"] for s in remaining_sessions] == ["sess-neighbour-keep"]

    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["id"] == "test-wall-timeout"
    assert outcome["outcome"] == "fail"
    assert "timeout" in outcome["why"]


def test_stub_vs_gpu_diff_only_card_wall_cap_writer_grant_vram() -> None:
    text = LAUNCH_SH.read_text(encoding="utf-8")
    marker = "# --- Mode-specific configuration (stub vs GPU diff block) ---"
    assert marker in text
    block = text.split(marker, 1)[1].split("fi\n", 1)[0]
    stub_part, gpu_part = block.split("\nelse\n", 1)
    stub_lines = [
        line.strip()
        for line in stub_part.splitlines()
        if line.strip() and not line.strip().startswith("if ")
    ]
    gpu_lines = [line.strip() for line in gpu_part.splitlines() if line.strip()]
    stub_vars = [line.split("=", 1)[0] for line in stub_lines]
    gpu_vars = [line.split("=", 1)[0] for line in gpu_lines]
    expected_vars = [
        "CARD",
        "COLAB_NEW_FLAGS",
        "WALL_SECONDS",
        "RATE_UNITS_PER_HOUR",
        "CAP_UNITS",
        "WRITER",
        "GRANT_ID",
        "VRAM_HALT_MIB",
    ]
    assert stub_vars == expected_vars
    assert gpu_vars == expected_vars


def test_uv_archive_sha_mismatch_and_version_refusal(tmp_path: Path) -> None:
    archive = tmp_path / f"uv-x86_64-unknown-linux-gnu-v{vm_udlab.UV_VERSION}.tar.gz"
    archive.write_bytes(b"fake-uv-archive-bytes")
    actual_sha = vm_udlab.sha256_file(archive)

    with pytest.raises(vm_udlab.RefusalError) as exc_sha:
        vm_udlab.verify_uv_archive_and_version(
            archive_path=archive,
            version_output=f"uv {vm_udlab.UV_VERSION} (7456b60bc 2026-10-20)",
            expected_sha256="0" * 64,
            expected_version=vm_udlab.UV_VERSION,
        )
    assert exc_sha.value.failure_class == "UV_ARCHIVE_SHA256_MISMATCH"

    with pytest.raises(vm_udlab.RefusalError) as exc_ver:
        vm_udlab.verify_uv_archive_and_version(
            archive_path=archive,
            version_output="uv 0.12.0",
            expected_sha256=actual_sha,
            expected_version=vm_udlab.UV_VERSION,
        )
    assert exc_ver.value.failure_class == "UV_VERSION_MISMATCH"

    sha_ok, ver_ok = vm_udlab.verify_uv_archive_and_version(
        archive_path=archive,
        version_output=f"uv {vm_udlab.UV_VERSION}\n",
        expected_sha256=actual_sha,
        expected_version=vm_udlab.UV_VERSION,
    )
    assert sha_ok == actual_sha
    assert vm_udlab.UV_VERSION in ver_ok

    # Also verify install_pinned_uv_and_sync_repo refuses on SHA-256 mismatch before extraction
    log_path = tmp_path / "uv_install.log"
    with log_path.open("w", encoding="utf-8", newline="\n") as log_fh:
        with pytest.raises(vm_udlab.RefusalError) as exc_install:
            vm_udlab.install_pinned_uv_and_sync_repo(
                work_dir=tmp_path,
                repo_dir=tmp_path,
                log_fh=log_fh,
                install_dir=tmp_path / "bin",
            )
    assert exc_install.value.failure_class == "UV_ARCHIVE_SHA256_MISMATCH"


def test_receipt_and_outcome_carry_gpu_name_with_fake_nvidia_smi(tmp_path: Path) -> None:
    env_paths = _setup_fake_colab_env(tmp_path)
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)
    out_dir = tmp_path / "run_gpu_receipt_ok"
    lock_file = tmp_path / "colab-open-gpu-ok.lock"

    fake_nvidia_smi = env_paths["fake_bin"] / "nvidia-smi"
    fake_gpu_line = "NVIDIA A100-SXM4-40GB, 40960 MiB, 550.54.15"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *"--query-gpu=name,memory.total,driver_version"* ]] && [[ "$*" == *"--format=csv,noheader"* ]]; then\n'
        f'  echo "{fake_gpu_line}"\n'
        "  exit 0\n"
        "fi\n"
        "exit 2\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_nvidia_smi.chmod(fake_nvidia_smi.stat().st_mode | stat.S_IEXEC)

    wsl_fake_bin = _to_wsl_path(env_paths["fake_bin"])
    wsl_sentinel_bin = _to_wsl_path(env_paths["sentinel_bin"])
    wsl_state_dir = _to_wsl_path(env_paths["state_dir"])
    wsl_prompts_dir = _to_wsl_path(prompts_dir)
    wsl_index_json = _to_wsl_path(index_json)
    wsl_out_dir = _to_wsl_path(out_dir)
    wsl_lock_file = _to_wsl_path(lock_file)
    wsl_launch = _to_wsl_path(LAUNCH_SH)

    cmd = (
        f"export PATH='{wsl_fake_bin}:{wsl_sentinel_bin}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{wsl_state_dir}'; "
        f"export UDLAB_OPEN_LOCK='{wsl_lock_file}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{wsl_prompts_dir}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{wsl_index_json}'; "
        f"export VM_DRYRUN=1; "
        f"bash '{wsl_launch}' --grant-id test-grant-gpu-01 --out-dir '{wsl_out_dir}' --run-id test-gpu-receipt"
    )
    proc = subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert not env_paths["sentinel_called"].exists()

    receipt = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert receipt["gpu_present"] is True
    assert receipt["gpu_name"] == "NVIDIA A100-SXM4-40GB"
    assert receipt["gpu_memory_total"] == "40960 MiB"
    assert receipt["gpu_driver_version"] == "550.54.15"
    assert receipt["nvidia_smi"] == fake_gpu_line
    assert receipt["hardware"] == fake_gpu_line
    assert vm_udlab.UV_VERSION in receipt["uv_version"]
    assert vm_udlab.REQUIRED_PYTHON_VERSION in receipt["python_version"]
    assert receipt["gpu_offload_check"]["checked"] is True
    assert receipt["gpu_offload_check"]["size_vram"] == receipt["gpu_offload_check"]["size"]

    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["outcome"] == "ok"
    assert outcome["hardware"] == fake_gpu_line


def test_stub_mode_records_cpu_and_absence_of_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Direct unit check with empty PATH (no nvidia-smi present): must not fail
    orig_path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", str(tmp_path))
    gpu_info = vm_udlab.probe_vm_gpu("stub")
    monkeypatch.setenv("PATH", orig_path)
    assert gpu_info["hardware"] == "CPU"
    assert gpu_info["gpu_name"] == "CPU"
    assert gpu_info["gpu_present"] is False
    assert gpu_info["gpu_memory_total"] is None
    assert gpu_info["gpu_driver_version"] is None
    assert gpu_info["nvidia_smi"] is None

    # 2. End-to-end stub run check via launch.sh
    env_paths = _setup_fake_colab_env(tmp_path)
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)
    out_dir = tmp_path / "run_stub_cpu_receipt"
    lock_file = tmp_path / "colab-open-stub-cpu.lock"

    wsl_fake_bin = _to_wsl_path(env_paths["fake_bin"])
    wsl_sentinel_bin = _to_wsl_path(env_paths["sentinel_bin"])
    wsl_state_dir = _to_wsl_path(env_paths["state_dir"])
    wsl_prompts_dir = _to_wsl_path(prompts_dir)
    wsl_index_json = _to_wsl_path(index_json)
    wsl_out_dir = _to_wsl_path(out_dir)
    wsl_lock_file = _to_wsl_path(lock_file)
    wsl_launch = _to_wsl_path(LAUNCH_SH)

    cmd = (
        f"export PATH='{wsl_fake_bin}:{wsl_sentinel_bin}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{wsl_state_dir}'; "
        f"export UDLAB_OPEN_LOCK='{wsl_lock_file}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{wsl_prompts_dir}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{wsl_index_json}'; "
        f"export VM_DRYRUN=1; "
        f"bash '{wsl_launch}' --stub --out-dir '{wsl_out_dir}' --run-id test-stub-cpu-receipt"
    )
    proc = subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert not env_paths["sentinel_called"].exists()

    receipt = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert receipt["hardware"] == "CPU"
    assert receipt["gpu_name"] == "CPU"
    assert receipt["gpu_present"] is False
    assert receipt["nvidia_smi"] is None
    assert receipt["gpu_offload_check"] == {"checked": False, "reason": "stub_mode"}

    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["hardware"] == "CPU"
    assert outcome["outcome"] == "ok"


def test_partial_offload_refusal_on_fake_api_ps(tmp_path: Path) -> None:
    import http.server
    import threading

    # 1. Direct verification of /api/ps payload with size_vram < size
    partial_ps = {
        "models": [
            {
                "name": "qwen3.6:35b",
                "model": "qwen3.6:35b",
                "size": 23456789012,
                "size_vram": 18000000000,
            }
        ]
    }
    with pytest.raises(vm_udlab.RefusalError) as exc_partial:
        vm_udlab.verify_model_full_gpu_offload(partial_ps, model="qwen3.6:35b")
    assert exc_partial.value.failure_class == "MODEL_PARTIAL_OFFLOAD"

    full_ps = {
        "models": [
            {
                "name": "qwen3.6:35b",
                "model": "qwen3.6:35b",
                "size": 23456789012,
                "size_vram": 23456789012,
            }
        ]
    }
    res_ok = vm_udlab.verify_model_full_gpu_offload(full_ps, model="qwen3.6:35b")
    assert res_ok["checked"] is True
    assert res_ok["size_vram"] == res_ok["size"] == 23456789012

    # 2. HTTP warm-up + /api/ps + unload lifecycle test against a fake /api/ps server
    requests_seen: list[tuple[str, str, dict[str, object]]] = []

    class FakeOllamaHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            requests_seen.append(("POST", self.path, body))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"done":true}')

        def do_GET(self) -> None:
            requests_seen.append(("GET", self.path, {}))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(partial_ps).encode("utf-8"))

        def log_message(self, format: str, *args: object) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(vm_udlab.RefusalError) as exc_srv:
            vm_udlab.check_ollama_server_full_gpu_offload(
                base_url=f"http://127.0.0.1:{port}",
                model="qwen3.6:35b",
            )
        assert exc_srv.value.failure_class == "MODEL_PARTIAL_OFFLOAD"
    finally:
        server.shutdown()
        server.server_close()

    # Verify warm-up load, /api/ps query, and keep_alive=0 unload were all sent
    assert [r[1] for r in requests_seen] == ["/api/generate", "/api/ps", "/api/generate"]
    assert requests_seen[-1][2].get("keep_alive") == 0

    # 3. End-to-end launch.sh GPU refusal when /api/ps reports size_vram < size
    env_paths = _setup_fake_colab_env(tmp_path)
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)
    out_dir = tmp_path / "run_gpu_partial_offload"
    lock_file = tmp_path / "colab-open-partial.lock"

    wsl_fake_bin = _to_wsl_path(env_paths["fake_bin"])
    wsl_sentinel_bin = _to_wsl_path(env_paths["sentinel_bin"])
    wsl_state_dir = _to_wsl_path(env_paths["state_dir"])
    wsl_prompts_dir = _to_wsl_path(prompts_dir)
    wsl_index_json = _to_wsl_path(index_json)
    wsl_out_dir = _to_wsl_path(out_dir)
    wsl_lock_file = _to_wsl_path(lock_file)
    wsl_launch = _to_wsl_path(LAUNCH_SH)
    partial_ps_json = json.dumps(partial_ps)

    cmd = (
        f"export PATH='{wsl_fake_bin}:{wsl_sentinel_bin}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{wsl_state_dir}'; "
        f"export UDLAB_OPEN_LOCK='{wsl_lock_file}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{wsl_prompts_dir}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{wsl_index_json}'; "
        f"export VM_DRYRUN=1; "
        f"export UDLAB_FAKE_API_PS_JSON='{partial_ps_json}'; "
        f"bash '{wsl_launch}' --grant-id test-grant-partial --out-dir '{wsl_out_dir}' --run-id test-partial-offload"
    )
    proc = subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode != 0
    assert not env_paths["sentinel_called"].exists()

    receipt = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "refused"
    assert receipt["failure_class"] == "MODEL_PARTIAL_OFFLOAD"


def test_finding1_verify_logs_clean_prompt_line_hashes_refusal_and_pass(tmp_path: Path) -> None:
    synthetic_prompt = (
        "short header line\n"
        "The quick brown fox jumps over the lazy dog near the riverbank every single morning\n"
        "Another distinct synthetic prompt line with plenty of characters to exceed forty easily\n"
    )
    line_hashes = vm_udlab.collect_prompt_line_sha256s(synthetic_prompt)
    assert len(line_hashes) == 2
    for h in line_hashes:
        assert len(h) == 64

    # 1. Log that does NOT contain any prompt line passes
    clean_log = tmp_path / "clean_no_prompt.log"
    clean_log.write_text(
        "2026-11-01T00:00:00Z normal status message with unrelated forty plus characters here\n",
        encoding="utf-8",
        newline="\n",
    )
    vm_udlab.verify_logs_clean([clean_log], prompt_line_hashes=line_hashes)

    # 2. Log containing the whole prompt line (even with extra whitespace) is refused
    leaked_full_log = tmp_path / "leaked_full.log"
    leaked_full_log.write_text(
        "  The   quick brown fox jumps over the lazy dog near the riverbank every single morning  \n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(vm_udlab.RefusalError) as exc_full:
        vm_udlab.verify_logs_clean([leaked_full_log], prompt_line_hashes=line_hashes)
    assert exc_full.value.failure_class == "LOG_POLICY_VIOLATION"

    # 3. Log containing a 40+ char prompt line as a segment split at sentence punctuation or tab is refused
    leaked_seg_log = tmp_path / "leaked_segment.log"
    leaked_seg_log.write_text(
        "prefix=ok\tAnother distinct synthetic prompt line with plenty of characters to exceed forty easily; suffix=done\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(vm_udlab.RefusalError) as exc_seg:
        vm_udlab.verify_logs_clean([leaked_seg_log], prompt_line_hashes=line_hashes)
    assert exc_seg.value.failure_class == "LOG_POLICY_VIOLATION"


def test_finding2_receipt_records_raw_sha256_deleted_and_absent_verified(tmp_path: Path) -> None:
    env_paths = _setup_fake_colab_env(tmp_path)
    fixture_dir = tmp_path / "f2_exact_ctx"
    prompts_dir = fixture_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    counts = vm_udlab.EXPECTED_EXACT_COUNTS
    entries = []
    expected_raw_shas: dict[str, str] = {}
    for idx, expected_count in enumerate(counts):
        fname = f"synthetic-{idx}.prompt.txt"
        body_text = (
            f"synthetic prompt #{idx} for count {expected_count} with a forty-plus character line for hashing check #{idx}\n"
        )
        raw_bytes = body_text.encode("utf-8")
        expected_raw_shas[fname] = hashlib.sha256(raw_bytes).hexdigest()
        sha = vm_udlab.compute_prompt_messages_sha256(body_text)
        (prompts_dir / fname).write_bytes(raw_bytes)
        entries.append(
            {
                "run": "synthetic-run",
                "call_id": f"scen_000{idx}:r0:call_0001",
                "file": fname,
                "prompt_sha256": sha,
                "prompt_eval_count": expected_count,
                "think": False,
            }
        )
    index_json = fixture_dir / "index.json"
    index_json.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8", newline="\n")

    out_dir = tmp_path / "run_f2_receipt"
    lock_file = tmp_path / "colab-open-f2.lock"
    cmd = (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_file)}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{_to_wsl_path(prompts_dir)}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{_to_wsl_path(index_json)}'; "
        f"export VM_DRYRUN=1; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id test-f2-receipt"
    )
    proc = subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert not env_paths["sentinel_called"].exists()

    receipt = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    gate_entries = receipt["exact_count_gate"]
    assert len(gate_entries) == 4
    for entry in gate_entries:
        fname = entry["file"]
        assert entry["raw_file_sha256"] == expected_raw_shas[fname]
        assert len(entry["prompt_sha256"]) == 64
        assert entry["raw_file_sha256"] != entry["prompt_sha256"]
        assert entry["deleted"] is True
        assert entry["absent_verified"] is True
    assert len(receipt["prompt_line_sha256s"]) == 4


def test_finding3a_prompts_extracted_just_before_gate_and_deleted_on_early_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)
    prompts_tar = tmp_path / "exact_ctx_prompts.tar.gz"
    with tarfile.open(prompts_tar, "w:gz") as tf:
        for p in sorted(prompts_dir.glob("*.prompt.txt")):
            tf.add(p, arcname=p.name, recursive=False)

    vm_prompts_dir = tmp_path / "vm_prompts"
    run_dir = tmp_path / "vm_run"
    work_dir = tmp_path / "vm_work"
    models_root = tmp_path / "vm_models"
    result_tar = tmp_path / "result.tar.gz"

    # Simulate an early non-refusal failure in install_pinned_ollama before the gate:
    # prompts_dir must never have .prompt.txt files extracted during install, and
    # prompts_tar + prompts_dir must be cleaned up by the top-level finally block.
    prompts_seen_during_install: list[Path] = []

    def _boom_install(w_dir: Path, log_fh: object) -> tuple[str, str]:
        if vm_prompts_dir.exists():
            prompts_seen_during_install.extend(vm_prompts_dir.glob("*.prompt.txt"))
        raise subprocess.CalledProcessError(100, ["apt-get", "update"])

    monkeypatch.setattr(vm_udlab, "install_pinned_ollama", _boom_install)

    rc = vm_udlab.main(
        [
            "--mode",
            "stub",
            "--writer",
            "mock",
            "--grant-id",
            "test-grant",
            "--vram-halt-mib",
            "0",
            "--wall-seconds",
            "60",
            "--repo-dir",
            str(REPO_ROOT),
            "--index-path",
            str(index_json),
            "--prompts-dir",
            str(vm_prompts_dir),
            "--prompts-tar",
            str(prompts_tar),
            "--run-dir",
            str(run_dir),
            "--work-dir",
            str(work_dir),
            "--models-root",
            str(models_root),
            "--result-tar",
            str(result_tar),
        ]
    )
    assert rc == 2
    assert prompts_seen_during_install == []
    assert not prompts_tar.exists()
    assert not list(vm_prompts_dir.glob("*.prompt.txt")) if vm_prompts_dir.exists() else True


def test_finding3b_unexpected_exception_writes_refusal_receipt_and_result_tar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_leak_msg = "secret_path=/root/private_token_xyz colab-runtime-proxy-token=LEAK999"

    # 1. Direct monkeypatched step in vm_udlab.main (no production failure-injection switch)
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path / "direct")
    prompts_tar = tmp_path / "direct_prompts.tar.gz"
    with tarfile.open(prompts_tar, "w:gz") as tf:
        for p in sorted(prompts_dir.glob("*.prompt.txt")):
            tf.add(p, arcname=p.name, recursive=False)

    vm_prompts_dir = tmp_path / "direct_vm_prompts"
    vm_prompts_dir.mkdir(parents=True, exist_ok=True)
    for p in prompts_dir.glob("*.prompt.txt"):
        (vm_prompts_dir / p.name).write_bytes(p.read_bytes())

    direct_run_dir = tmp_path / "direct_run"
    direct_result_tar = tmp_path / "direct_result.tar.gz"

    def _boom_dryrun_step(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError(secret_leak_msg)

    monkeypatch.setattr(vm_udlab, "run_dryrun_pipeline", _boom_dryrun_step)

    rc = vm_udlab.main(
        [
            "--mode",
            "stub",
            "--writer",
            "mock",
            "--grant-id",
            "test-grant",
            "--vram-halt-mib",
            "0",
            "--wall-seconds",
            "60",
            "--repo-dir",
            str(REPO_ROOT),
            "--index-path",
            str(index_json),
            "--prompts-dir",
            str(vm_prompts_dir),
            "--prompts-tar",
            str(prompts_tar),
            "--run-dir",
            str(direct_run_dir),
            "--result-tar",
            str(direct_result_tar),
            "--dryrun",
        ]
    )
    assert rc == 2
    assert (direct_run_dir / "vm_receipt.json").is_file()
    assert direct_result_tar.is_file()
    direct_receipt = json.loads((direct_run_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert direct_receipt["status"] == "refused"
    assert direct_receipt["failure_class"] == "UNEXPECTED_ERROR"
    assert direct_receipt["step"] == "dryrun_pipeline"
    assert direct_receipt["exception_type"] == "RuntimeError"
    direct_raw = (direct_run_dir / "vm_receipt.json").read_text(encoding="utf-8")
    assert "private_token_xyz" not in direct_raw
    assert "LEAK999" not in direct_raw
    assert not prompts_tar.exists()
    assert not list(vm_prompts_dir.glob("*.prompt.txt")) if vm_prompts_dir.exists() else True

    # 2. End-to-end launch.sh check via test-only sitecustomize monkeypatch on PYTHONPATH
    env_paths = _setup_fake_colab_env(tmp_path)
    out_dir = tmp_path / "run_f3b_unexpected"
    lock_file = tmp_path / "colab-open-f3b.lock"
    py_inject_dir = tmp_path / "py_inject"
    py_inject_dir.mkdir(parents=True, exist_ok=True)
    (py_inject_dir / "sitecustomize.py").write_text(
        "import os, shutil\n"
        "if os.environ.get('UDLAB_STEP') == 'run':\n"
        f"    shutil.disk_usage = lambda _p: (_ for _ in ()).throw(RuntimeError({secret_leak_msg!r}))\n",
        encoding="utf-8",
        newline="\n",
    )

    cmd = (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"export PYTHONPATH='{_to_wsl_path(py_inject_dir)}'; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_file)}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{_to_wsl_path(prompts_dir)}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{_to_wsl_path(index_json)}'; "
        f"export VM_DRYRUN=1; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id test-f3b-unexpected"
    )
    proc = subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 2
    assert not env_paths["sentinel_called"].exists()

    # Receipt and result.tar.gz must both be fetched back to out_dir
    assert (out_dir / "vm_receipt.json").is_file()
    assert (out_dir / "result.tar.gz").is_file()
    receipt = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "refused"
    assert receipt["failure_class"] == "UNEXPECTED_ERROR"
    assert receipt["step"] == "dryrun_pipeline"
    assert receipt["exception_type"] == "RuntimeError"
    # Exception message text must NOT be recorded in receipt or logs
    receipt_raw = (out_dir / "vm_receipt.json").read_text(encoding="utf-8")
    assert "private_token_xyz" not in receipt_raw
    assert "LEAK999" not in receipt_raw

    tar_members = vm_udlab.validate_tar_file(out_dir / "result.tar.gz")
    assert "vm_receipt.json" in tar_members
    assert "logs/vm.log" in tar_members

    # Prompts on the fake VM were cleaned up
    vm_fs = env_paths["state_dir"] / "vm_fs"
    assert not (vm_fs / "exact_ctx_prompts.tar.gz").exists()
    assert not list((vm_fs / "udlab_prompts").glob("*.prompt.txt")) if (vm_fs / "udlab_prompts").exists() else True


def test_finding4_watchdog_fetches_before_stop_and_bounds_hanging_fetch(tmp_path: Path) -> None:
    # Part A: Watchdog fires at WALL - FETCH_MARGIN, runs fetch_results_once, then colab stop
    env_paths = _setup_fake_colab_env(tmp_path)
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)
    out_dir = tmp_path / "run_f4_order"
    lock_file = tmp_path / "colab-open-f4-order.lock"

    cmd = (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_file)}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{_to_wsl_path(prompts_dir)}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{_to_wsl_path(index_json)}'; "
        f"export VM_DRYRUN=1; "
        f"export UDLAB_WALL_SECONDS=2; "
        f"export UDLAB_FETCH_MARGIN_SECONDS=1; "
        f"export UDLAB_WATCHDOG_FETCH_TIMEOUT_SECONDS=1; "
        f"export FAKE_COLAB_EXEC_SLEEP=4; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id test-f4-order"
    )
    proc = subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode != 0
    assert not env_paths["sentinel_called"].exists()

    calls = [
        json.loads(line)
        for line in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()
    ]
    fetch_idx = next(
        i
        for i, c in enumerate(calls)
        if c["subcmd"] == "download" and "/content/result.tar.gz" in c["args"]
    )
    stop_idx = next(i for i, c in enumerate(calls) if c["subcmd"] == "stop")
    assert fetch_idx < stop_idx, f"Expected fetch before stop in watchdog, got calls={calls}"

    # Part B: Hanging fake fetch is killed by watchdog fetch timeout and session is still stopped
    env_paths_hang = _setup_fake_colab_env(tmp_path / "hang_case")
    out_dir_hang = tmp_path / "run_f4_hang"
    lock_file_hang = tmp_path / "colab-open-f4-hang.lock"

    cmd_hang = (
        f"export PATH='{_to_wsl_path(env_paths_hang['fake_bin'])}:{_to_wsl_path(env_paths_hang['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths_hang['state_dir'])}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_file_hang)}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{_to_wsl_path(prompts_dir)}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{_to_wsl_path(index_json)}'; "
        f"export VM_DRYRUN=1; "
        f"export UDLAB_WALL_SECONDS=3; "
        f"export UDLAB_FETCH_MARGIN_SECONDS=2; "
        f"export UDLAB_WATCHDOG_FETCH_TIMEOUT_SECONDS=1; "
        f"export FAKE_COLAB_EXEC_SLEEP=5; "
        f"export FAKE_COLAB_FETCH_SLEEP=30; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --stub --out-dir '{_to_wsl_path(out_dir_hang)}' --run-id test-f4-hang"
    )
    proc_hang = subprocess.run(["bash", "-c", cmd_hang], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc_hang.returncode != 0
    assert not env_paths_hang["sentinel_called"].exists()

    calls_hang = [
        json.loads(line)
        for line in env_paths_hang["calls_log"].read_text(encoding="utf-8").splitlines()
    ]
    fetch_hang_idx = next(
        i
        for i, c in enumerate(calls_hang)
        if c["subcmd"] == "download" and "/content/result.tar.gz" in c["args"]
    )
    stop_hang_idx = next(i for i, c in enumerate(calls_hang) if c["subcmd"] == "stop")
    assert fetch_hang_idx < stop_hang_idx
    stopped_hang = [
        json.loads(line)["session_id"]
        for line in env_paths_hang["stopped_file"].read_text(encoding="utf-8").splitlines()
    ]
    assert "udlab-test-f4-hang" in stopped_hang


def test_finding5_disk_space_check_refuses_below_40gib_and_records_in_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections import namedtuple

    FakeUsage = namedtuple("usage", ["total", "used", "free"])

    # 1. Below 40 GiB refuses with DISK_TOO_SMALL
    small_free = vm_udlab.MIN_FREE_DISK_BYTES - 1024
    monkeypatch.setattr(
        vm_udlab.shutil,
        "disk_usage",
        lambda _p: FakeUsage(100 * 1024**3, 61 * 1024**3, small_free),
    )
    with pytest.raises(vm_udlab.RefusalError) as exc_disk:
        vm_udlab.check_model_disk_space(tmp_path / "models")
    assert exc_disk.value.failure_class == "DISK_TOO_SMALL"

    # 2. At or above 40 GiB passes and is recorded in vm_receipt.json
    ample_free = 55 * 1024 * 1024 * 1024
    monkeypatch.setattr(
        vm_udlab.shutil,
        "disk_usage",
        lambda _p: FakeUsage(100 * 1024**3, 45 * 1024**3, ample_free),
    )
    assert vm_udlab.check_model_disk_space(tmp_path / "models") == ample_free

    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)
    run_dir = tmp_path / "run_disk_ok"
    result_tar = tmp_path / "result_disk_ok.tar.gz"
    rc = vm_udlab.main(
        [
            "--mode",
            "stub",
            "--writer",
            "mock",
            "--grant-id",
            "test-grant",
            "--vram-halt-mib",
            "0",
            "--wall-seconds",
            "60",
            "--repo-dir",
            str(REPO_ROOT),
            "--index-path",
            str(index_json),
            "--prompts-dir",
            str(prompts_dir),
            "--run-dir",
            str(run_dir),
            "--models-root",
            str(tmp_path / "models"),
            "--result-tar",
            str(result_tar),
            "--dryrun",
        ]
    )
    assert rc == 0
    receipt = json.loads((run_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert receipt["free_disk_bytes"] == ample_free


def test_finding6_launch_sh_refuses_drive_path_for_exact_ctx_prompts(tmp_path: Path) -> None:
    env_paths = _setup_fake_colab_env(tmp_path)
    out_dir = tmp_path / "run_f6_drive_prompts"
    lock_file = tmp_path / "colab-open-f6.lock"

    for bad_prompts_path in ("/content/drive/MyDrive/prompts", "/gdrive/MyDrive/prompts"):
        cmd = (
            f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
            f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
            f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_file)}'; "
            f"export UDLAB_EXACT_CTX_PROMPTS='{bad_prompts_path}'; "
            f"export VM_DRYRUN=1; "
            f"bash '{_to_wsl_path(LAUNCH_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id test-f6-drive"
        )
        proc = subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
        assert proc.returncode == 2
        assert "FORBIDDEN_STORAGE_PATH" in proc.stderr
        assert not env_paths["sentinel_called"].exists()


def test_fetch_race_watchdog_waits_for_in_progress_main_fetch_before_stop(
    tmp_path: Path,
) -> None:
    env_paths = _setup_fake_colab_env(tmp_path)
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)
    out_dir = tmp_path / "run_fetch_race"
    lock_file = tmp_path / "colab-open-fetch-race.lock"

    # Main exec completes (~1.2s) and starts main fetch, which takes 2.5s.
    # Watchdog fires at WALL(4) - MARGIN(2) = 2s while main fetch is in progress.
    cmd = (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_file)}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{_to_wsl_path(prompts_dir)}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{_to_wsl_path(index_json)}'; "
        f"export VM_DRYRUN=1; "
        f"export UDLAB_WALL_SECONDS=4; "
        f"export UDLAB_FETCH_MARGIN_SECONDS=2; "
        f"export UDLAB_WATCHDOG_FETCH_TIMEOUT_SECONDS=5; "
        f"export FAKE_COLAB_EXEC_SLEEP=0; "
        f"export FAKE_COLAB_FETCH_SLEEP=2.5; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id test-fetch-race"
    )
    proc = subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert not env_paths["sentinel_called"].exists()

    calls = [
        json.loads(line)
        for line in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()
    ]
    result_tar_dl_calls = [
        c for c in calls if c["subcmd"] == "download" and "/content/result.tar.gz" in c["args"]
    ]
    # Exactly one fetch of result.tar.gz runs
    assert len(result_tar_dl_calls) == 1

    # Watchdog fired while main fetch was in progress, waited for FETCH_DONE, then stopped
    cli_log_text = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "FETCH_START" in cli_log_text
    assert "RIG_DEADLINE_EXCEEDED" in cli_log_text
    assert "FETCH_DONE" in cli_log_text
    assert "WATCHDOG_STOP_SESSION" in cli_log_text
    assert cli_log_text.index("FETCH_START") < cli_log_text.index("RIG_DEADLINE_EXCEEDED")
    assert cli_log_text.index("FETCH_DONE") < cli_log_text.index("WATCHDOG_STOP_SESSION")

    # Result tar is whole and valid
    assert (out_dir / "result.tar.gz").is_file()
    tar_members = vm_udlab.validate_tar_file(out_dir / "result.tar.gz")
    assert "vm_receipt.json" in tar_members
    assert "generations.jsonl" in tar_members
    assert proc.returncode == 0


_CONTRACT_STOP_TOKENS = {"|", "||", "&&", ";", ">", ">>", "2>&1", "<", "</dev/null", "0</dev/null", ")", "&"}
_STDIN_CMD_BOUNDARY_TOKENS = {"|", "||", "&&", ";", ")", "&"}
_CONTRACT_VALUE_OPTS_BY_CMD: dict[str, set[str]] = {
    "new": {"-s", "--session", "--gpu", "--tpu"},
    "exec": {"-s", "--session", "-f", "--file", "--timeout", "--env", "--output-image"},
    "upload": {"-s", "--session"},
    "download": {"-s", "--session"},
    "stop": {"-s", "--session"},
    "sessions": set(),
    "usage": set(),
    "status": {"-s", "--session"},
}


def _shlex_tokens(line: str) -> list[str]:
    import re
    import shlex

    # Strip outer assignment quotes around $(...) so inner `colab`/`run_colab` calls are tokenized
    unwrapped = re.sub(r'([A-Za-z_][A-Za-z0-9_]*=)"\$\((.*)\)"', r"\1 \2", line)
    unwrapped = re.sub(r"\$\(([^()]*)\)", r" \1 ", unwrapped)
    lex = shlex.shlex(unwrapped, posix=True, punctuation_chars="|&;<>()")
    lex.whitespace_split = True
    try:
        return list(lex)
    except ValueError:
        return unwrapped.split()


def extract_and_check_colab_invocations(
    script_text: str,
    contract_data: dict[str, object] | None = None,
    script_name: str = "launch.sh",
) -> tuple[list[tuple[int, str, list[str]]], list[str]]:
    """Extract every `colab ...` or `run_colab ...` invocation from a bash script and check against contract."""
    import re

    if contract_data is None:
        contract_data = json.loads(CLI_CONTRACT_JSON.read_text(encoding="utf-8"))
    commands = contract_data["commands"]  # type: ignore[index]
    cli_name = contract_data["cli"]  # type: ignore[index]
    cli_ver = contract_data["version"]  # type: ignore[index]

    array_flags: set[str] = set()
    for m in re.finditer(r"COLAB_NEW_FLAGS=\(([^)]*)\)", script_text):
        array_flags.update(t for t in m.group(1).split() if t.startswith("-"))
    for m in re.finditer(r"EXEC_ENV_ARGS\+?=\(([^)]*)\)", script_text, flags=re.S):
        array_flags.update(
            t.split("=", 1)[0]
            for t in _shlex_tokens(m.group(1))
            if t.startswith("-") and len(t) > 1
        )

    extracted: list[tuple[int, str, list[str]]] = []
    errors: list[str] = []

    joined = re.sub(r"\\\n", " ", script_text)
    for lineno, raw_line in enumerate(joined.splitlines(), start=1):
        stripped = raw_line.lstrip()
        if stripped.startswith("#") or stripped.startswith("printf ") or stripped.startswith("echo "):
            continue
        code = raw_line.split("#", 1)[0]
        toks = _shlex_tokens(code)
        for i, tok in enumerate(toks):
            if tok not in ("colab", "run_colab", "cli"):
                continue
            if i + 1 >= len(toks):
                continue
            nxt = toks[i + 1]
            if nxt in ("()", "$@", '"$@"') or (i > 0 and toks[i - 1] == "function"):
                continue
            if not re.fullmatch(r"[a-z][a-z0-9-]*", nxt):
                continue
            subcmd = nxt
            rest = toks[i + 2:]
            opts: list[str] = []
            value_opts = _CONTRACT_VALUE_OPTS_BY_CMD.get(subcmd, set())
            skip_next = False
            for r in rest:
                if r in _CONTRACT_STOP_TOKENS:
                    break
                if skip_next:
                    skip_next = False
                    continue
                if "COLAB_NEW_FLAGS" in r or "EXEC_ENV_ARGS" in r or r == "$NEW_ARGS":
                    for af in sorted(array_flags):
                        if subcmd == "new" and af in ("--gpu", "--high-mem", "--tpu", "-s", "--session"):
                            opts.append(af)
                        elif subcmd == "exec" and af in ("--env", "-f", "--file", "--timeout", "-s", "--session"):
                            opts.append(af)
                    continue
                if r == "--":
                    opts.append("--")
                    break
                if r.startswith("-") and len(r) > 1:
                    opt_name = r.split("=", 1)[0]
                    opts.append(opt_name)
                    if "=" not in r and opt_name in value_opts:
                        skip_next = True
                elif (
                    re.fullmatch(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", r)
                    and subcmd not in ("upload", "download", "cp")
                    and r not in ("$live", "$S", "$SESSION_NAME", "$1")
                ):
                    errors.append(
                        f"{script_name}:{lineno}: colab {subcmd} {r}: unchecked variable in option position"
                    )
            extracted.append((lineno, subcmd, opts))
            if subcmd not in commands:
                errors.append(f"{script_name}:{lineno}: colab {subcmd} is not in the contract")
                continue
            allowed = set(commands[subcmd]["options"])
            for o in opts:
                if o not in allowed:
                    errors.append(
                        f"{script_name}:{lineno}: colab {subcmd} {o}: not an option of {cli_name} {cli_ver} "
                        f"(allowed: {' '.join(sorted(allowed))})"
                    )
    return extracted, errors


def check_colab_invocations_have_stdin_devnull(
    script_text: str,
    script_name: str = "launch.sh",
) -> tuple[list[tuple[int, str, str]], list[str]]:
    """Parse a bash script and fail if any `colab` or `run_colab` invocation lacks `</dev/null`."""
    import re

    checked: list[tuple[int, str, str]] = []
    errors: list[str] = []

    joined = re.sub(r"\\\n", " ", script_text)
    for lineno, raw_line in enumerate(joined.splitlines(), start=1):
        stripped = raw_line.lstrip()
        if stripped.startswith("#") or stripped.startswith("printf ") or stripped.startswith("echo "):
            continue
        code = raw_line.split("#", 1)[0]
        toks = _shlex_tokens(code)
        for i, tok in enumerate(toks):
            if tok not in ("colab", "run_colab", "cli"):
                continue
            if i + 1 >= len(toks):
                continue
            nxt = toks[i + 1]
            if nxt == "()" or (i > 0 and toks[i - 1] == "function"):
                continue
            is_subcmd = bool(re.fullmatch(r"[a-z][a-z0-9-]*", nxt))
            is_passthrough = nxt in ("$@", '"$@"')
            if not (is_subcmd or is_passthrough):
                continue
            rest = toks[i + 2 :]
            cmd_slice: list[str] = []
            for r in rest:
                if r in _STDIN_CMD_BOUNDARY_TOKENS:
                    break
                cmd_slice.append(r)
            has_devnull = False
            for k, r in enumerate(cmd_slice):
                if r in ("</dev/null", "0</dev/null"):
                    has_devnull = True
                    break
                if r in ("<", "0<") and k + 1 < len(cmd_slice) and cmd_slice[k + 1] == "/dev/null":
                    has_devnull = True
                    break
            checked.append((lineno, tok, nxt))
            if not has_devnull:
                errors.append(
                    f"{script_name}:{lineno}: {tok} {nxt}: missing </dev/null stdin redirect"
                )
    return checked, errors


def test_colab_cli_contract_file_sha256_and_static_invocation_check() -> None:
    assert CLI_CONTRACT_JSON.is_file()
    assert vm_udlab.sha256_file(CLI_CONTRACT_JSON) == CLI_CONTRACT_SHA256

    # 1. Verify current launch.sh passes static contract check with zero errors
    script_text = LAUNCH_SH.read_text(encoding="utf-8")
    extracted, errors = extract_and_check_colab_invocations(script_text)
    assert errors == [], f"Static CLI contract errors in launch.sh: {errors}"
    subcmds_used = {sub for _, sub, _ in extracted}
    assert {"sessions", "new", "exec", "upload", "download", "stop"}.issubset(subcmds_used)
    assert "cp" not in subcmds_used
    assert "ls" not in subcmds_used

    # 2. Verify the static checker fails on fix-3's invalid calls: cp, ls --json, new --json, exec --
    legacy_snippet = "\n".join(
        [
            'colab cp -s "$1" "/content/result.tar.gz" "$2/result.tar.gz"',
            'run_colab cp -s "$S" "/content/result.tar.gz" "$OUT_DIR/result.tar.gz"',
            'PRE_LS="$(run_colab ls --json || echo "[]")"',
            'NEW_OUT="$(run_colab new "${COLAB_NEW_FLAGS[@]+"${COLAB_NEW_FLAGS[@]}"}" --json)"',
            'GATE_OUT="$(run_colab exec -s "$S" -- "echo VM_GATE_OK && uname -a")"',
            'colab exec -s "$S" -- "timeout --signal=TERM --kill-after=30 ${WALL_SECONDS}s python3 /content/udlab/scripts/colab/vm_udlab.py"',
        ]
    )
    _, legacy_errors = extract_and_check_colab_invocations(legacy_snippet, script_name="legacy_launch.sh")
    assert any("colab cp is not in the contract" in e for e in legacy_errors)
    assert any("colab ls is not in the contract" in e for e in legacy_errors)
    assert any("colab new --json: not an option" in e for e in legacy_errors)
    assert any("colab exec --: not an option" in e for e in legacy_errors)


def test_fake_colab_enforces_contract_and_rejects_unknown_commands_and_flags(tmp_path: Path) -> None:
    env_paths = _setup_fake_colab_env(tmp_path)
    wsl_fake_colab = _to_wsl_path(env_paths["fake_bin"] / "colab")
    wsl_state_dir = _to_wsl_path(env_paths["state_dir"])
    wsl_lock_file = _to_wsl_path(tmp_path / "open.lock")

    def _run_fake(args_str: str) -> subprocess.CompletedProcess[str]:
        cmd = (
            f"export FAKE_COLAB_STATE_DIR='{wsl_state_dir}'; "
            f"export UDLAB_OPEN_LOCK='{wsl_lock_file}'; "
            f"'{wsl_fake_colab}' {args_str}"
        )
        return subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)

    # Rejects unknown subcommand `cp`
    res_cp = _run_fake("cp -s udlab-1 /content/a /tmp/b")
    assert res_cp.returncode != 0
    assert "No such command 'cp'" in res_cp.stderr

    # Rejects `ls --json`
    res_ls = _run_fake("ls --json")
    assert res_ls.returncode != 0
    assert "No such command 'ls'" in res_ls.stderr

    # Rejects `new --json`
    res_new_json = _run_fake("new -s udlab-1 --json")
    assert res_new_json.returncode != 0
    assert "colab new --json: not an option" in res_new_json.stderr

    # Open a valid session so we can test `exec` rejections and timeout semantics
    res_new_ok = _run_fake("new -s udlab-1")
    assert res_new_ok.returncode == 0

    # Rejects `exec` with `--` trailing shell command
    res_exec_dash = _run_fake("exec -s udlab-1 -- 'echo hi'")
    assert res_exec_dash.returncode != 0
    assert "colab exec --: not an option" in res_exec_dash.stderr

    # Rejects `exec` without `-f`
    res_exec_nof = _run_fake("exec -s udlab-1 --timeout 10")
    assert res_exec_nof.returncode != 0
    assert "requires -s/--session and -f/--file" in res_exec_nof.stderr

    # Enforces `--timeout` semantics on `exec -f <pyfile>`
    slow_py = tmp_path / "slow_job.py"
    slow_py.write_text("import time\ntime.sleep(5)\n", encoding="utf-8", newline="\n")
    res_exec_timeout = _run_fake(f"exec -s udlab-1 -f '{_to_wsl_path(slow_py)}' --timeout 0.2")
    assert res_exec_timeout.returncode == 124
    assert "timed out" in res_exec_timeout.stderr


def test_launch_sh_dry_mode_prints_expected_command_sequence_without_secrets(tmp_path: Path) -> None:
    env_paths = _setup_fake_colab_env(tmp_path)
    out_dir = tmp_path / "dry_out"
    wsl_launch = _to_wsl_path(LAUNCH_SH)
    wsl_out_dir = _to_wsl_path(out_dir)
    wsl_vm_py = _to_wsl_path(VM_UDLAB_PY)

    # 1. Stub mode dry output
    cmd_stub = (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"bash '{wsl_launch}' --stub --dry --out-dir '{wsl_out_dir}' --run-id dry-stub-01"
    )
    proc_stub = subprocess.run(["bash", "-c", cmd_stub], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc_stub.returncode == 0, proc_stub.stderr
    assert not env_paths["sentinel_called"].exists()
    assert not env_paths["calls_log"].exists()

    lines_stub = [ln.strip() for ln in proc_stub.stdout.splitlines() if ln.strip()]
    assert lines_stub == [
        "colab sessions",
        "colab new -s udlab-dry-stub-01",
        f"colab exec -s udlab-dry-stub-01 --env UDLAB_STEP=gate -f {wsl_vm_py} --timeout=180",
        f"colab upload -s udlab-dry-stub-01 {wsl_out_dir}/stage/repo_bundle.tar.gz /content/repo_bundle.tar.gz",
        f"colab upload -s udlab-dry-stub-01 {wsl_out_dir}/stage/exact_ctx_prompts.tar.gz /content/exact_ctx_prompts.tar.gz",
        (
            f"colab exec -s udlab-dry-stub-01 --env UDLAB_STEP=run --env UDLAB_MODE=stub --env UDLAB_WRITER=mock "
            f"--env UDLAB_COUNT=3 --env UDLAB_SEED=5 --env UDLAB_GRANT_ID=cpu-stub-none-no-gpu "
            f"--env UDLAB_VRAM_HALT_MIB=0 --env UDLAB_WALL_SECONDS=6900 --env UDLAB_REPO_DIR=/content/udlab "
            f"--env UDLAB_REPO_TAR=/content/repo_bundle.tar.gz --env UDLAB_PROMPTS_DIR=/content/udlab_prompts "
            f"--env UDLAB_PROMPTS_TAR=/content/exact_ctx_prompts.tar.gz --env UDLAB_RUN_DIR=/content/udlab_run "
            f"--env UDLAB_RESULT_TAR=/content/result.tar.gz --env UDLAB_STATUS_FILE=/content/vm_status.json "
            f"--env UDLAB_RESUME=0 -f {wsl_vm_py} --timeout=6960"
        ),
        f"colab download -s udlab-dry-stub-01 /content/result.tar.gz {wsl_out_dir}/result.tar.gz",
        f"colab download -s udlab-dry-stub-01 /content/vm_status.json {wsl_out_dir}/vm_status.json",
        f"colab download -s udlab-dry-stub-01 /content/udlab_run/vm_receipt.json {wsl_out_dir}/vm_receipt.json",
        "colab stop -s udlab-dry-stub-01",
    ]

    # 2. GPU mode dry output
    cmd_gpu = (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"bash '{wsl_launch}' --dry --grant-id grant-a100-xyz --out-dir '{wsl_out_dir}' --run-id dry-gpu-01"
    )
    proc_gpu = subprocess.run(["bash", "-c", cmd_gpu], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc_gpu.returncode == 0, proc_gpu.stderr
    lines_gpu = [ln.strip() for ln in proc_gpu.stdout.splitlines() if ln.strip()]
    assert lines_gpu == [
        "colab sessions",
        "colab new --gpu A100 -s udlab-dry-gpu-01",
        f"colab exec -s udlab-dry-gpu-01 --env UDLAB_STEP=gate -f {wsl_vm_py} --timeout=180",
        f"colab upload -s udlab-dry-gpu-01 {wsl_out_dir}/stage/repo_bundle.tar.gz /content/repo_bundle.tar.gz",
        f"colab upload -s udlab-dry-gpu-01 {wsl_out_dir}/stage/exact_ctx_prompts.tar.gz /content/exact_ctx_prompts.tar.gz",
        (
            f"colab exec -s udlab-dry-gpu-01 --env UDLAB_STEP=run --env UDLAB_MODE=gpu --env UDLAB_WRITER=ollama "
            f"--env UDLAB_COUNT=3 --env UDLAB_SEED=5 --env UDLAB_GRANT_ID=grant-a100-xyz "
            f"--env UDLAB_VRAM_HALT_MIB=26000 --env UDLAB_WALL_SECONDS=5100 --env UDLAB_REPO_DIR=/content/udlab "
            f"--env UDLAB_REPO_TAR=/content/repo_bundle.tar.gz --env UDLAB_PROMPTS_DIR=/content/udlab_prompts "
            f"--env UDLAB_PROMPTS_TAR=/content/exact_ctx_prompts.tar.gz --env UDLAB_RUN_DIR=/content/udlab_run "
            f"--env UDLAB_RESULT_TAR=/content/result.tar.gz --env UDLAB_STATUS_FILE=/content/vm_status.json "
            f"--env UDLAB_RESUME=0 -f {wsl_vm_py} --timeout=5160"
        ),
        f"colab download -s udlab-dry-gpu-01 /content/result.tar.gz {wsl_out_dir}/result.tar.gz",
        f"colab download -s udlab-dry-gpu-01 /content/vm_status.json {wsl_out_dir}/vm_status.json",
        f"colab download -s udlab-dry-gpu-01 /content/udlab_run/vm_receipt.json {wsl_out_dir}/vm_receipt.json",
        "colab stop -s udlab-dry-gpu-01",
    ]

    # 3. Session-smoke dry output
    cmd_smoke = (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"bash '{wsl_launch}' --session-smoke --dry --out-dir '{wsl_out_dir}' --run-id smoke-01"
    )
    proc_smoke = subprocess.run(["bash", "-c", cmd_smoke], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc_smoke.returncode == 0, proc_smoke.stderr
    lines_smoke = [ln.strip() for ln in proc_smoke.stdout.splitlines() if ln.strip()]
    assert lines_smoke == [
        "colab sessions",
        "colab new -s udlab-smoke-01",
        "colab sessions",
        "colab stop -s udlab-smoke-01",
        "colab sessions",
    ]


def test_launch_sh_session_smoke_mode_with_fake_colab(tmp_path: Path) -> None:
    env_paths = _setup_fake_colab_env(tmp_path)
    out_dir = tmp_path / "run_session_smoke"
    lock_file = tmp_path / "colab-open-smoke.lock"
    locks_dir = env_paths["colab_locks_dir"]

    cmd = (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(locks_dir)}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_file)}'; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --session-smoke --out-dir '{_to_wsl_path(out_dir)}' --run-id smoke-01"
    )
    proc = subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert not env_paths["sentinel_called"].exists()

    calls = [
        json.loads(line)
        for line in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()
    ]
    # Exact sequence of subcommands: sessions, new, sessions, stop, sessions
    assert [c["subcmd"] for c in calls] == ["sessions", "new", "sessions", "stop", "sessions"]
    assert calls[1]["args"] == ["-s", "udlab-smoke-01"]
    assert calls[1]["lock_held"] is True

    # Lock file was written after `colab new` (visible during second `colab sessions` and `colab stop`)
    assert "udlab-smoke-01.lock" in calls[2]["reconcile_locks"]
    assert "udlab-smoke-01.lock" in calls[3]["reconcile_locks"]
    # And removed in the trap after `colab stop` (absent by the final `colab sessions` and after exit)
    assert "udlab-smoke-01.lock" not in calls[4]["reconcile_locks"]
    assert not (locks_dir / "udlab-smoke-01.lock").exists()

    # Neighbour session untouched
    remaining_sessions = json.loads(env_paths["sessions_file"].read_text(encoding="utf-8"))
    assert [s["session_id"] for s in remaining_sessions] == ["sess-neighbour-keep"]

    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["id"] == "smoke-01"
    assert outcome["hardware"] == "CPU"
    assert outcome["outcome"] == "ok"


def test_fix5_1_and_3_kernel_argv_entry_and_honest_fake_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import runpy

    # 1. Verify kernel-style sys.argv is ignored by parse_args() and runpy.run_path(__main__)
    kernel_argv = [
        "/usr/local/lib/python3.11/dist-packages/google/colab/_ipython_kernel_launcher.py",
        "-f",
        "/root/.local/share/jupyter/runtime/kernel-abc.json",
    ]
    monkeypatch.setattr(sys, "argv", kernel_argv)
    monkeypatch.setenv("UDLAB_STEP", "gate")
    status_gate = tmp_path / "gate_status.json"
    monkeypatch.setenv("UDLAB_STATUS_FILE", str(status_gate))

    assert vm_udlab.is_kernel_env() is True
    parsed_gate = vm_udlab.parse_args()
    assert parsed_gate.step == "gate"

    # Run vm_udlab.py via runpy.run_path(..., run_name='__main__') with kernel sys.argv
    runpy.run_path(str(VM_UDLAB_PY), run_name="__main__")
    gate_data = json.loads(status_gate.read_text(encoding="utf-8"))
    assert gate_data["step"] == "gate"
    assert gate_data["rc"] == 0

    # 2. Run full dryrun step via kernel-style subprocess (exact command from card Fix 3)
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)
    run_dir = tmp_path / "kernel_run"
    result_tar = tmp_path / "kernel_result.tar.gz"
    status_run = tmp_path / "kernel_status.json"

    kernel_env = os.environ.copy()
    kernel_env.update(
        {
            "UDLAB_STEP": "run",
            "UDLAB_MODE": "stub",
            "UDLAB_WRITER": "mock",
            "UDLAB_COUNT": "3",
            "UDLAB_SEED": "5",
            "UDLAB_GRANT_ID": "cpu-stub-none-no-gpu",
            "UDLAB_VRAM_HALT_MIB": "0",
            "UDLAB_WALL_SECONDS": "60",
            "UDLAB_REPO_DIR": str(REPO_ROOT),
            "UDLAB_INDEX_PATH": str(index_json),
            "UDLAB_PROMPTS_DIR": str(prompts_dir),
            "UDLAB_RUN_DIR": str(run_dir),
            "UDLAB_RESULT_TAR": str(result_tar),
            "UDLAB_STATUS_FILE": str(status_run),
            "VM_DRYRUN": "1",
        }
    )
    kernel_cmd = [
        sys.executable,
        "-c",
        (
            "import sys,runpy; "
            "sys.argv=['colab_kernel_launcher.py','-f','/tmp/kernel-fake.json']; "
            f"runpy.run_path({str(VM_UDLAB_PY)!r}, run_name='__main__')"
        ),
    ]
    proc = subprocess.run(kernel_cmd, env=kernel_env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert status_run.is_file()
    status_data = json.loads(status_run.read_text(encoding="utf-8"))
    assert status_data == {
        "step": "verify_logs_and_pack_tar",
        "rc": 0,
        "status": "ok",
        "reason": "ok",
    }
    assert result_tar.is_file()


def test_fix5_2_status_file_decides_outcome_when_exec_exits_zero_and_missing_status_fails(
    tmp_path: Path,
) -> None:
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)

    # Part A: VM script refuses (rc=2 in vm_status.json), fake `colab exec` exits 0 (kernel swallows SystemExit),
    # and launch.sh still fails with exit code 2 from vm_status.json.
    env_a = _setup_fake_colab_env(tmp_path / "case_a")
    out_a = tmp_path / "out_a"
    lock_a = tmp_path / "lock_a.lock"
    cmd_a = (
        f"export PATH='{_to_wsl_path(env_a['fake_bin'])}:{_to_wsl_path(env_a['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_a['state_dir'])}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_a)}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{_to_wsl_path(prompts_dir)}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{_to_wsl_path(index_json)}'; "
        f"export VM_DRYRUN=1; "
        f"export UDLAB_FAKE_API_PS_JSON='{{\"models\":[]}}'; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --grant-id grant-a100-status --out-dir '{_to_wsl_path(out_a)}' --run-id status-refusal-01"
    )
    proc_a = subprocess.run(["bash", "-c", cmd_a], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc_a.returncode == 2
    assert (out_a / "vm_status.json").is_file()
    status_a = json.loads((out_a / "vm_status.json").read_text(encoding="utf-8"))
    assert status_a["rc"] == 2
    assert status_a["status"] == "refused"
    assert status_a["failure_class"] == "MODEL_PARTIAL_OFFLOAD"
    outcome_a = json.loads((out_a / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome_a["outcome"] == "fail"
    assert "MODEL_PARTIAL_OFFLOAD" in outcome_a["why"]

    # Part B: VM script succeeds and builds result.tar.gz, `colab exec` exits 0, but vm_status.json is missing.
    # launch.sh must treat missing vm_status.json as failure (exit non-zero).
    env_b = _setup_fake_colab_env(tmp_path / "case_b")
    out_b = tmp_path / "out_b"
    lock_b = tmp_path / "lock_b.lock"
    cmd_b = (
        f"export PATH='{_to_wsl_path(env_b['fake_bin'])}:{_to_wsl_path(env_b['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_b['state_dir'])}'; "
        f"export FAKE_COLAB_DELETE_STATUS_FILE=1; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_b)}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{_to_wsl_path(prompts_dir)}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{_to_wsl_path(index_json)}'; "
        f"export VM_DRYRUN=1; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --stub --out-dir '{_to_wsl_path(out_b)}' --run-id status-missing-01"
    )
    proc_b = subprocess.run(["bash", "-c", cmd_b], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc_b.returncode != 0
    assert not (out_b / "vm_status.json").exists()
    assert (out_b / "result.tar.gz").is_file()
    outcome_b = json.loads((out_b / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome_b["outcome"] == "fail"
    assert "vm_status.json missing" in outcome_b["why"]


def test_fix5_4_timeout_hierarchy_margins_and_single_wall_derivation(tmp_path: Path) -> None:
    env_paths = _setup_fake_colab_env(tmp_path)
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)
    out_dir = tmp_path / "run_timeout_order"
    lock_file = tmp_path / "colab-open-timeout-order.lock"

    cmd = (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_file)}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{_to_wsl_path(prompts_dir)}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{_to_wsl_path(index_json)}'; "
        f"export VM_DRYRUN=1; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id timeout-order-01"
    )
    proc = subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    cli_log = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    # Stub WALL_SECONDS=7200 -> VM_WALL_SECONDS=6900 < EXEC_TIMEOUT_SECONDS=6960 < WATCHDOG_SLEEP_SECONDS=7020 < RIG_TIMEOUT=7205
    assert (
        "wall_s=7200 vm_wall_s=6900 exec_timeout_s=6960 watchdog_sleep_s=7020 rig_timeout_s=7205"
        in cli_log
    )
    assert 6900 < 6960 < 7020 < 7205
    assert (6960 - 6900) == 60
    assert (7020 - 6960) == 60
    assert (7200 - 7020) == 180
    assert (7205 - 7020) == 185

    # Verify custom single wall value derives all 4 values with the exact same margins
    out_custom = tmp_path / "run_timeout_custom"
    cmd_custom = (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"export UDLAB_WALL_SECONDS=3600; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --stub --dry --out-dir '{_to_wsl_path(out_custom)}' --run-id timeout-custom-01"
    )
    proc_custom = subprocess.run(["bash", "-c", cmd_custom], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc_custom.returncode == 0
    assert "--env UDLAB_WALL_SECONDS=3300" in proc_custom.stdout
    assert "--timeout=3360" in proc_custom.stdout


def test_fix5_5_session_name_reuse_refused_before_new_and_failed_new_does_not_stop_existing(
    tmp_path: Path,
) -> None:
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)

    # Case 1: Session name `udlab-reuse-01` already listed in `colab sessions` before `new` (normal mode & smoke mode)
    for mode_arg in ("--stub", "--session-smoke"):
        sub_tmp = tmp_path / f"reuse_{mode_arg.lstrip('-')}"
        env_paths = _setup_fake_colab_env(sub_tmp)
        existing_sessions = [
            {"session_id": "sess-neighbour-keep", "gpu": "T4"},
            {"session_id": "udlab-reuse-01", "gpu": "CPU"},
        ]
        env_paths["sessions_file"].write_text(
            json.dumps(existing_sessions),
            encoding="utf-8",
            newline="\n",
        )
        existing_lock = env_paths["colab_locks_dir"] / "udlab-reuse-01.lock"
        existing_lock.write_text("99999\n", encoding="utf-8", newline="\n")

        out_dir = sub_tmp / "out"
        lock_file = sub_tmp / "open.lock"
        cmd = (
            f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
            f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
            f"export COLAB_LOCK_DIR='{_to_wsl_path(env_paths['colab_locks_dir'])}'; "
            f"export COLAB_RECONCILE_SH='{_to_wsl_path(env_paths['state_dir'] / 'nonexistent_reconcile.sh')}'; "
            f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_file)}'; "
            f"export UDLAB_EXACT_CTX_PROMPTS='{_to_wsl_path(prompts_dir)}'; "
            f"export UDLAB_EXACT_CTX_INDEX='{_to_wsl_path(index_json)}'; "
            f"export VM_DRYRUN=1; "
            f"bash '{_to_wsl_path(LAUNCH_SH)}' {mode_arg} --out-dir '{_to_wsl_path(out_dir)}' --run-id reuse-01"
        )
        proc = subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
        assert proc.returncode != 0
        assert "SESSION_NAME_IN_USE" in proc.stderr

        calls = [
            json.loads(line)
            for line in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()
        ]
        # Only `sessions` was called; neither `new` nor `stop` was called
        assert [c["subcmd"] for c in calls] == ["sessions"]
        assert not env_paths["stopped_file"].exists()
        # Both existing sessions and the existing lock file remain untouched
        after_sessions = json.loads(env_paths["sessions_file"].read_text(encoding="utf-8"))
        assert [s["session_id"] for s in after_sessions] == ["sess-neighbour-keep", "udlab-reuse-01"]
        assert existing_lock.is_file()
        assert existing_lock.read_text(encoding="utf-8").strip() == "99999"

    # Case 2: `colab sessions` does not list the name, but `colab new` fails (`FAKE_COLAB_FAIL_NEW=1`).
    # The EXIT trap must NOT call `colab stop` and must NOT remove any lock file because `NEW_SUCCEEDED` is 0.
    sub_fail = tmp_path / "fail_new"
    env_fail = _setup_fake_colab_env(sub_fail)
    concurrent_lock = env_fail["colab_locks_dir"] / "udlab-failnew-01.lock"
    concurrent_lock.write_text("88888\n", encoding="utf-8", newline="\n")

    out_fail = sub_fail / "out"
    lock_fail = sub_fail / "open.lock"
    cmd_fail = (
        f"export PATH='{_to_wsl_path(env_fail['fake_bin'])}:{_to_wsl_path(env_fail['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_fail['state_dir'])}'; "
        f"export FAKE_COLAB_FAIL_NEW=1; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_fail['colab_locks_dir'])}'; "
        f"export COLAB_RECONCILE_SH='{_to_wsl_path(env_fail['state_dir'] / 'nonexistent_reconcile.sh')}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_fail)}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{_to_wsl_path(prompts_dir)}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{_to_wsl_path(index_json)}'; "
        f"export VM_DRYRUN=1; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --stub --out-dir '{_to_wsl_path(out_fail)}' --run-id failnew-01"
    )
    proc_fail = subprocess.run(["bash", "-c", cmd_fail], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert proc_fail.returncode != 0
    calls_fail = [
        json.loads(line)
        for line in env_fail["calls_log"].read_text(encoding="utf-8").splitlines()
    ]
    assert [c["subcmd"] for c in calls_fail] == ["sessions", "new"]
    assert not env_fail["stopped_file"].exists()
    assert concurrent_lock.is_file()
    assert concurrent_lock.read_text(encoding="utf-8").strip() == "88888"


def test_fix5_6_fake_colab_help_flag_prints_and_exits_zero_with_no_side_effects(
    tmp_path: Path,
) -> None:
    env_paths = _setup_fake_colab_env(tmp_path)
    wsl_fake_colab = _to_wsl_path(env_paths["fake_bin"] / "colab")
    wsl_state_dir = _to_wsl_path(env_paths["state_dir"])
    before_sessions = env_paths["sessions_file"].read_text(encoding="utf-8")

    def _run_help(args_str: str) -> subprocess.CompletedProcess[str]:
        cmd = f"export FAKE_COLAB_STATE_DIR='{wsl_state_dir}'; '{wsl_fake_colab}' {args_str}"
        return subprocess.run(["bash", "-c", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)

    for top_flag in ("-h", "--help"):
        res = _run_help(top_flag)
        assert res.returncode == 0
        assert "Usage: colab" in res.stdout

    contract_data = json.loads(CLI_CONTRACT_JSON.read_text(encoding="utf-8"))
    for subcmd, spec in contract_data["commands"].items():
        for hflag in ("-h", "--help"):
            res = _run_help(f"{subcmd} {hflag}")
            assert res.returncode == 0, f"{subcmd} {hflag} failed: {res.stderr}"
            assert f"Usage: colab {subcmd}" in res.stdout
            for opt in spec["options"]:
                assert opt in res.stdout

    # Zero side effects: no calls logged, sessions unchanged, no sessions stopped
    assert not env_paths["calls_log"].exists()
    assert env_paths["sessions_file"].read_text(encoding="utf-8") == before_sessions
    assert not env_paths["stopped_file"].exists()


def test_fix6_1_and_3_all_colab_calls_redirect_stdin_from_devnull_and_static_guard(
    tmp_path: Path,
) -> None:
    # 1. Static guard check on launch.sh: every colab / run_colab invocation must have </dev/null
    script_text = LAUNCH_SH.read_text(encoding="utf-8")
    checked, errors = check_colab_invocations_have_stdin_devnull(script_text)
    assert errors == [], f"Missing </dev/null on colab invocations in launch.sh: {errors}"
    assert len(checked) >= 16
    checked_targets = {nxt for _, _, nxt in checked}
    assert {"sessions", "new", "exec", "upload", "download", "stop", "$@"}.issubset(checked_targets)

    # 2. Negative fixture: any colab or run_colab invocation without </dev/null must fail the static guard
    negative_fixture = "\n".join(
        [
            'out="$(colab "$@" 2>"$err_file")" || rc=$?',
            'PRE_SESSIONS="$(run_colab sessions || true)"',
            'run_colab new -s "$SESSION_NAME" >/dev/null',
            'timeout "$GATE_TIMEOUT_SECONDS" colab exec -s "$S" --env UDLAB_STEP=gate -f "$SCRIPT_DIR/vm_udlab.py" --timeout="$GATE_TIMEOUT_SECONDS" 2>&1',
            'timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$REPO_TAR" "/content/repo_bundle.tar.gz" 2>&1',
            'colab download -s "$1" "/content/result.tar.gz" "$2/result.tar.gz" >/dev/null 2>&1 || true',
            'colab stop -s "$S" >/dev/null 2>&1 || true',
            'colab stop -s "$S" </dev/null >/dev/null 2>&1 || true',
        ]
    )
    neg_checked, neg_errors = check_colab_invocations_have_stdin_devnull(
        negative_fixture, script_name="negative_fixture.sh"
    )
    assert len(neg_checked) == 8
    assert len(neg_errors) == 7
    assert any("negative_fixture.sh:1: colab $@: missing </dev/null stdin redirect" in e for e in neg_errors)
    assert any("negative_fixture.sh:2: run_colab sessions: missing </dev/null stdin redirect" in e for e in neg_errors)
    assert any("negative_fixture.sh:3: run_colab new: missing </dev/null stdin redirect" in e for e in neg_errors)
    assert any("negative_fixture.sh:4: colab exec: missing </dev/null stdin redirect" in e for e in neg_errors)
    assert any("negative_fixture.sh:5: colab upload: missing </dev/null stdin redirect" in e for e in neg_errors)
    assert any("negative_fixture.sh:6: colab download: missing </dev/null stdin redirect" in e for e in neg_errors)
    assert any("negative_fixture.sh:7: colab stop: missing </dev/null stdin redirect" in e for e in neg_errors)

    # 3. Runtime verification: run launch.sh with a non-/dev/null stdin and FAKE_COLAB_READ_STDIN=1;
    # every colab CLI invocation must see /dev/null on fd 0 and never block on caller stdin.
    env_paths = _setup_fake_colab_env(tmp_path)
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)
    out_dir = tmp_path / "run_stdin_closed"
    lock_file = tmp_path / "colab-open-stdin.lock"
    caller_stdin_file = tmp_path / "caller_stdin.txt"
    caller_stdin_file.write_text("caller terminal input line\n", encoding="utf-8", newline="\n")

    cmd = (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
        f"export FAKE_COLAB_READ_STDIN=1; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_file)}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{_to_wsl_path(prompts_dir)}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{_to_wsl_path(index_json)}'; "
        f"export VM_DRYRUN=1; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id stdin-check-01 < '{_to_wsl_path(caller_stdin_file)}'"
    )
    proc = subprocess.run(
        ["bash", "-c", cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    calls = [
        json.loads(line)
        for line in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 10
    assert all(c.get("stdin_is_devnull") is True for c in calls), f"Expected all calls to have stdin=/dev/null: {calls}"


def test_fix6_2_dry_and_live_timeout_forms_and_argument_order_match(tmp_path: Path) -> None:
    env_paths = _setup_fake_colab_env(tmp_path)
    prompts_dir, index_json = _make_synthetic_prompts_fixture(tmp_path)
    out_dir = tmp_path / "run_match_forms"
    lock_file = tmp_path / "colab-open-match.lock"
    wsl_launch = _to_wsl_path(LAUNCH_SH)
    wsl_out_dir = _to_wsl_path(out_dir)

    base_exports = (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_file)}'; "
        f"export UDLAB_EXACT_CTX_PROMPTS='{_to_wsl_path(prompts_dir)}'; "
        f"export UDLAB_EXACT_CTX_INDEX='{_to_wsl_path(index_json)}'; "
        f"export VM_DRYRUN=1; "
    )

    proc_dry = subprocess.run(
        ["bash", "-c", f"{base_exports} bash '{wsl_launch}' --stub --dry --out-dir '{wsl_out_dir}' --run-id match-01"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_dry.returncode == 0, proc_dry.stderr
    dry_tokens_list = [ln.strip().split() for ln in proc_dry.stdout.splitlines() if ln.strip()]

    proc_live = subprocess.run(
        ["bash", "-c", f"{base_exports} bash '{wsl_launch}' --stub --out-dir '{wsl_out_dir}' --run-id match-01"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_live.returncode == 0, f"stdout={proc_live.stdout}\nstderr={proc_live.stderr}"
    calls = [
        json.loads(line)
        for line in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()
    ]
    live_tokens_list = [["colab", c["subcmd"], *c["args"]] for c in calls]

    assert len(dry_tokens_list) == len(live_tokens_list) == 10
    for dry_toks, live_toks in zip(dry_tokens_list, live_tokens_list, strict=True):
        if dry_toks[1] == "upload":
            # Stage directory in live mode is a mktemp dir, but option positions and archive basenames match
            assert dry_toks[:4] == live_toks[:4]
            assert Path(dry_toks[4]).name == Path(live_toks[4]).name
            assert dry_toks[5:] == live_toks[5:]
        else:
            assert dry_toks == live_toks, f"Dry command {dry_toks} != live command {live_toks}"

    # Specifically verify both gate exec and run exec use `--timeout=<seconds>` in both dry and live
    assert dry_tokens_list[2][-1] == "--timeout=180"
    assert live_tokens_list[2][-1] == "--timeout=180"
    assert dry_tokens_list[5][-1] == "--timeout=6960"
    assert live_tokens_list[5][-1] == "--timeout=6960"


def test_fix6_4_unverified_sessions_format_documented_and_smoke_receipt_records_shape_without_ids(
    tmp_path: Path,
) -> None:
    readme_path = SCRIPTS_COLAB_DIR / "README.md"
    launch_text = LAUNCH_SH.read_text(encoding="utf-8")
    readme_text = readme_path.read_text(encoding="utf-8")

    # 1. Verify comment in launch.sh and line in README.md document the verified `colab sessions` row shapes
    # while leaving the `[name]` match in place.
    assert 'grep -Fq "[$SESSION_NAME]"' in launch_text
    assert "colab sessions" in launch_text
    assert "colab sessions" in readme_text

    # 2. Run CPU `--session-smoke` and verify its receipt records the real output's shape with no tokens or ids.
    env_paths = _setup_fake_colab_env(tmp_path / "smoke_ok")
    out_dir = tmp_path / "smoke_ok_out"
    lock_file = tmp_path / "smoke_ok.lock"

    cmd = (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_paths['colab_locks_dir'])}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_file)}'; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --session-smoke --out-dir '{_to_wsl_path(out_dir)}' --run-id smoke-shape-01"
    )
    proc = subprocess.run(
        ["bash", "-c", cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    receipt_path = out_dir / "vm_receipt.json"
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))

    for doc in (receipt, outcome):
        assert "sessions_output_shape" in doc
        shape = doc["sessions_output_shape"]
        assert shape["format_verified_on_real_cli"] is False
        assert shape["pre_sessions"] == {
            "line_count": 1,
            "line_shapes": ["[<TOKEN>] <TOKEN> | <TOKEN>"],
            "contains_bracketed_session": False,
            "matches_bracket_prefix": False,
        }
        assert shape["mid_sessions"] == {
            "line_count": 2,
            "line_shapes": [
                "[<TOKEN>] <TOKEN> | <TOKEN>",
                "[<SESSION>] <TOKEN> | <TOKEN>",
            ],
            "contains_bracketed_session": True,
            "matches_bracket_prefix": True,
        }
        assert shape["post_sessions"] == {
            "line_count": 1,
            "line_shapes": ["[<TOKEN>] <TOKEN> | <TOKEN>"],
            "contains_bracketed_session": False,
            "matches_bracket_prefix": False,
        }
        serialized_shape = json.dumps(shape)
        assert "sess-neighbour-keep" not in serialized_shape
        assert "m-s-fake" not in serialized_shape
        assert "udlab-smoke-shape-01" not in serialized_shape
        assert "FAKE_SECRET_TOKEN" not in serialized_shape

    # 3. Even if `colab sessions` outputs an unbracketed format, `--session-smoke` still records the shape
    # (without tokens or ids) in the receipt and stops the session via the trap.
    env_alt = _setup_fake_colab_env(tmp_path / "smoke_alt")
    out_alt = tmp_path / "smoke_alt_out"
    lock_alt = tmp_path / "smoke_alt.lock"
    cmd_alt = (
        f"export PATH='{_to_wsl_path(env_alt['fake_bin'])}:{_to_wsl_path(env_alt['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_alt['state_dir'])}'; "
        f"export FAKE_COLAB_SESSIONS_TEMPLATE='{{sid}} :: m-s-secret-{{sid}} ({{gpu}})'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_alt['colab_locks_dir'])}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_alt)}'; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --session-smoke --out-dir '{_to_wsl_path(out_alt)}' --run-id smoke-alt-01"
    )
    proc_alt = subprocess.run(
        ["bash", "-c", cmd_alt],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_alt.returncode != 0
    receipt_alt = json.loads((out_alt / "vm_receipt.json").read_text(encoding="utf-8"))
    shape_alt = receipt_alt["sessions_output_shape"]
    assert shape_alt["mid_sessions"]["line_shapes"] == [
        "<TOKEN> :: <TOKEN> (<TOKEN>)",
        "<SESSION> :: <TOKEN> (<TOKEN>)",
    ]
    assert shape_alt["mid_sessions"]["contains_bracketed_session"] is False
    assert "m-s-secret" not in json.dumps(shape_alt)
    assert "sess-neighbour-keep" not in json.dumps(shape_alt)
    assert "udlab-smoke-alt-01" not in json.dumps(shape_alt)


def _load_launch_sh_smoke_helpers() -> dict[str, object]:
    """Extract and execute the helper functions from write_outcome_draft's embedded Python block in launch.sh."""
    script_text = LAUNCH_SH.read_text(encoding="utf-8")
    marker = '"$PYTHON_BIN" - "$OUTCOME_FILE" "$RUN_ID" "$CARD" "$CAP_UNITS" "$RATE_UNITS_PER_HOUR" "$runtime_s" "$RUN_OUTCOME" "$RUN_WHY" <<\'PY\'\n'
    start = script_text.index(marker) + len(marker)
    end = script_text.index("\nPY\n", start)
    py_block = script_text[start:end]
    defs_start = py_block.index("def _mask_line_shape")
    defs_end = py_block.index("\nreceipt_path =")
    defs_src = "from __future__ import annotations\nimport re\n\n" + py_block[defs_start:defs_end]
    ns: dict[str, object] = {}
    exec(defs_src, ns)
    return ns


def test_fix7_1_smoke_receipt_format_verified_flag_and_decision_function(
    tmp_path: Path,
) -> None:
    launch_text = LAUNCH_SH.read_text(encoding="utf-8")
    # launch.sh must never set UDLAB_COLAB_IS_FAKE=1 itself
    assert "UDLAB_COLAB_IS_FAKE=1" not in launch_text

    # 1. Direct unit test of the decision function with both inputs (since a fake-free CLI run cannot run in tests)
    helpers = _load_launch_sh_smoke_helpers()
    assert "_is_format_verified_on_real_cli" in helpers, (
        "Expected _is_format_verified_on_real_cli decision function in launch.sh embedded Python"
    )
    decide = helpers["_is_format_verified_on_real_cli"]
    assert callable(decide)
    assert decide(True, False) is True
    assert decide(True, True) is False
    assert decide(False, False) is False
    assert decide(False, True) is False

    assert "_build_smoke_shape" in helpers, (
        "Expected _build_smoke_shape helper in launch.sh embedded Python"
    )
    build_shape = helpers["_build_smoke_shape"]
    assert callable(build_shape)
    real_mid_line = "[udlab-smoke-01] m-s-123 | backend: CPU | status: READY | age: 0m"
    empty_line = "[colab] No active sessions found on server."
    shape_real = build_shape(empty_line, real_mid_line, empty_line, "udlab-smoke-01", False)
    assert shape_real["format_verified_on_real_cli"] is True
    assert shape_real["mid_sessions"]["matches_bracket_prefix"] is True

    shape_fake = build_shape(empty_line, real_mid_line, empty_line, "udlab-smoke-01", True)
    assert shape_fake["format_verified_on_real_cli"] is False
    assert shape_fake["mid_sessions"]["matches_bracket_prefix"] is True

    shape_missing_row = build_shape(empty_line, empty_line, empty_line, "udlab-smoke-01", False)
    assert shape_missing_row["format_verified_on_real_cli"] is False
    assert shape_missing_row["mid_sessions"]["matches_bracket_prefix"] is False

    # 2. Fake CLI run (UDLAB_COLAB_IS_FAKE=1 set by test harness) whose mid listing matches gives False
    env_fake = _setup_fake_colab_env(tmp_path / "fake_run")
    out_fake = tmp_path / "fake_run_out"
    lock_fake = tmp_path / "fake_run.lock"
    cmd_fake = (
        f"export PATH='{_to_wsl_path(env_fake['fake_bin'])}:{_to_wsl_path(env_fake['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_fake['state_dir'])}'; "
        f"export UDLAB_COLAB_IS_FAKE=1; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_fake['colab_locks_dir'])}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_fake)}'; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --session-smoke --out-dir '{_to_wsl_path(out_fake)}' --run-id fix7-fake-01"
    )
    proc_fake = subprocess.run(
        ["bash", "-c", cmd_fake],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_fake.returncode == 0, f"stdout={proc_fake.stdout}\nstderr={proc_fake.stderr}"
    receipt_fake = json.loads((out_fake / "vm_receipt.json").read_text(encoding="utf-8"))
    assert receipt_fake["sessions_output_shape"]["mid_sessions"]["matches_bracket_prefix"] is True
    assert receipt_fake["sessions_output_shape"]["format_verified_on_real_cli"] is False

    # 3. A run whose mid listing lacks the session row gives False
    env_lack = _setup_fake_colab_env(tmp_path / "lack_run")
    out_lack = tmp_path / "lack_run_out"
    lock_lack = tmp_path / "lack_run.lock"
    cmd_lack = (
        f"export PATH='{_to_wsl_path(env_lack['fake_bin'])}:{_to_wsl_path(env_lack['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_lack['state_dir'])}'; "
        f"export UDLAB_COLAB_IS_FAKE=1; "
        f"export FAKE_COLAB_SESSIONS_OVERRIDE='[colab] No active sessions found on server.'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_lack['colab_locks_dir'])}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_lack)}'; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --session-smoke --out-dir '{_to_wsl_path(out_lack)}' --run-id fix7-lack-01"
    )
    proc_lack = subprocess.run(
        ["bash", "-c", cmd_lack],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_lack.returncode != 0
    receipt_lack = json.loads((out_lack / "vm_receipt.json").read_text(encoding="utf-8"))
    assert receipt_lack["sessions_output_shape"]["mid_sessions"]["matches_bracket_prefix"] is False
    assert receipt_lack["sessions_output_shape"]["format_verified_on_real_cli"] is False


def test_fix7_2_colab_message_line_never_treated_as_session(
    tmp_path: Path,
) -> None:
    empty_msg_line = "[colab] No active sessions found on server."
    live_row_shape = "[<SESSION>] <TOKEN> | <TOKEN>: <TOKEN> | <TOKEN>: <TOKEN> | <TOKEN>: <TOKEN>"

    # Verify the launcher's pre-check and mid-check expressions remain in launch.sh
    launch_text = LAUNCH_SH.read_text(encoding="utf-8")
    assert 'if printf \'%s\\n\' "$PRE_SESSIONS" | grep -Fq "[$SESSION_NAME]"; then' in launch_text
    assert 'if ! printf \'%s\\n\' "$MID_SESSIONS" | grep -q "^\\[$S\\]"; then' in launch_text

    # (a) `[colab] No active sessions found on server.` alone:
    #     pre-check passes (no reuse) and mid-check fails (session not listed).
    env_a = _setup_fake_colab_env(tmp_path / "case_a")
    out_a = tmp_path / "case_a_out"
    lock_a = tmp_path / "case_a.lock"
    cmd_a = (
        f"export PATH='{_to_wsl_path(env_a['fake_bin'])}:{_to_wsl_path(env_a['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_a['state_dir'])}'; "
        f"export UDLAB_COLAB_IS_FAKE=1; "
        f"export FAKE_COLAB_SESSIONS_OVERRIDE='{empty_msg_line}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_a['colab_locks_dir'])}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_a)}'; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --session-smoke --out-dir '{_to_wsl_path(out_a)}' --run-id smoke-msg-a"
    )
    proc_a = subprocess.run(
        ["bash", "-c", cmd_a],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_a.returncode != 0
    assert "SESSION_NAME_IN_USE" not in proc_a.stderr
    calls_a = [
        json.loads(ln)
        for ln in env_a["calls_log"].read_text(encoding="utf-8").splitlines()
    ]
    # Pre-check passed (so `new` was called), then mid-check failed (so `stop` was called by trap)
    assert [c["subcmd"] for c in calls_a] == ["sessions", "new", "sessions", "stop", "sessions"]
    outcome_a = json.loads((out_a / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome_a["outcome"] == "fail"
    assert outcome_a["why"] == "colab sessions did not list [udlab-smoke-msg-a] after colab new"
    shape_a = outcome_a["sessions_output_shape"]
    assert shape_a["pre_sessions"]["contains_bracketed_session"] is False
    assert shape_a["mid_sessions"]["matches_bracket_prefix"] is False

    # (b) `[colab] No active sessions found on server.` followed by a live-row line for the session
    #     in the same listing: mid-check passes.
    env_b = _setup_fake_colab_env(tmp_path / "case_b")
    # Start with empty sessions so pre-sessions is `[colab] No active sessions found on server.`
    env_b["sessions_file"].write_text("[]\n", encoding="utf-8", newline="\n")
    out_b = tmp_path / "case_b_out"
    lock_b = tmp_path / "case_b.lock"
    cmd_b = (
        f"export PATH='{_to_wsl_path(env_b['fake_bin'])}:{_to_wsl_path(env_b['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_b['state_dir'])}'; "
        f"export UDLAB_COLAB_IS_FAKE=1; "
        f"export FAKE_COLAB_SESSIONS_PREFIX_LINE='{empty_msg_line}'; "
        f"export FAKE_COLAB_SESSIONS_TEMPLATE='[{{sid}}] m-s-tok1 | hw: {{gpu}} | status: READY | age: 1m'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_b['colab_locks_dir'])}'; "
        f"export UDLAB_OPEN_LOCK='{_to_wsl_path(lock_b)}'; "
        f"bash '{_to_wsl_path(LAUNCH_SH)}' --session-smoke --out-dir '{_to_wsl_path(out_b)}' --run-id smoke-msg-b"
    )
    proc_b = subprocess.run(
        ["bash", "-c", cmd_b],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_b.returncode == 0, f"stdout={proc_b.stdout}\nstderr={proc_b.stderr}"
    outcome_b = json.loads((out_b / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome_b["outcome"] == "ok"
    shape_b = outcome_b["sessions_output_shape"]
    assert shape_b["pre_sessions"]["line_shapes"] == [
        "[<TOKEN>] <TOKEN> <TOKEN> <TOKEN> <TOKEN> <TOKEN> <TOKEN>"
    ]
    assert shape_b["mid_sessions"]["line_shapes"] == [
        "[<TOKEN>] <TOKEN> <TOKEN> <TOKEN> <TOKEN> <TOKEN> <TOKEN>",
        live_row_shape,
    ]
    assert shape_b["mid_sessions"]["matches_bracket_prefix"] is True

    # (c) Launcher name rules (`launch.sh:143-147`) force `udlab-` prefix on SESSION_NAME, so a session
    #     literally named `colab` is not allowed; even `--run-id colab` resolves to `udlab-colab`.
    assert 'if [[ "$RUN_ID" == udlab-* ]]; then\n  SESSION_NAME="$RUN_ID"\nelse\n  SESSION_NAME="udlab-$RUN_ID"\nfi' in launch_text
    proc_c = subprocess.run(
        ["bash", "-c", f"bash '{_to_wsl_path(LAUNCH_SH)}' --session-smoke --dry --run-id colab"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_c.returncode == 0
    assert "colab new -s udlab-colab" in proc_c.stdout
    assert "colab new -s colab\n" not in proc_c.stdout


def test_fix7_3_readme_and_launch_sh_record_verified_real_session_shapes() -> None:
    readme_text = (SCRIPTS_COLAB_DIR / "README.md").read_text(encoding="utf-8")
    launch_text = LAUNCH_SH.read_text(encoding="utf-8")

    live_row_shape = "[<SESSION>] <TOKEN> | <TOKEN>: <TOKEN> | <TOKEN>: <TOKEN> | <TOKEN>: <TOKEN>"
    empty_list_shape = "[colab] No active sessions found on server."

    for text_name, text in (("README.md", readme_text), ("launch.sh", launch_text)):
        assert "row format has not been checked yet" not in text, f"{text_name} still has unverified note"
        assert "row format not checked yet" not in text, f"{text_name} still has unverified note"
        assert live_row_shape in text, f"{text_name} missing live session row shape"
        assert empty_list_shape in text, f"{text_name} missing empty list shape"
        assert "2026-11-02" in text, f"{text_name} missing verification date 2026-11-02"
        assert "smoke-01" in text, f"{text_name} missing smoke-01 reference"








