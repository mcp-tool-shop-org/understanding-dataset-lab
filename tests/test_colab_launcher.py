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
        capture_output=True,
        text=True,
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
    """Create a front-of-PATH fake `colab` CLI and a later-on-PATH real-colab sentinel."""
    fake_bin = tmp_path / "fake_bin"
    sentinel_bin = tmp_path / "sentinel_real_bin"
    state_dir = tmp_path / "fake_colab_state"
    fake_bin.mkdir(parents=True, exist_ok=True)
    sentinel_bin.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

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

    fake_driver_py = state_dir / "fake_colab_driver.py"
    fake_driver_py.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

state_dir = Path(os.environ["FAKE_COLAB_STATE_DIR"])
vm_fs = state_dir / "vm_fs"
vm_fs.mkdir(parents=True, exist_ok=True)
calls_log = state_dir / "calls.jsonl"
sessions_file = state_dir / "sessions.json"
lock_file = Path(os.environ["UDLAB_OPEN_LOCK"])

args = sys.argv[1:]
subcmd = args[0] if args else ""

# Check whether the open lock is currently held by the caller
lock_held = False
if lock_file.exists():
    try:
        res = subprocess.run(["flock", "-n", str(lock_file), "true"], check=False)
        lock_held = (res.returncode != 0)
    except Exception:
        lock_held = False

record = {"subcmd": subcmd, "args": args, "lock_held": lock_held, "ts": time.time()}
with calls_log.open("a", encoding="utf-8", newline="\\n") as fh:
    fh.write(json.dumps(record) + "\\n")

# Always emit a proxy token in stderr to verify launch.sh scrubs CLI output
sys.stderr.write("debug proxy https://colab.research.google.com/?colab-runtime-proxy-token=FAKE_SECRET_TOKEN_777\\n")

sessions = json.loads(sessions_file.read_text(encoding="utf-8"))

if subcmd == "ls":
    print(json.dumps(sessions))
    raise SystemExit(0)

if subcmd == "new":
    sid = "sess-udlab-test-001"
    gpu = "CPU"
    if "--gpu" in args:
        idx = args.index("--gpu")
        if idx + 1 < len(args):
            gpu = args[idx + 1]
    sessions.append({"session_id": sid, "gpu": gpu})
    sessions_file.write_text(json.dumps(sessions), encoding="utf-8", newline="\\n")
    print(json.dumps({"session_id": sid, "gpu": gpu}))
    raise SystemExit(0)

if subcmd == "cp":
    # colab cp -s <S> <src> <dst>
    s_idx = args.index("-s")
    sid = args[s_idx + 1]
    src = args[s_idx + 2]
    dst = args[s_idx + 3]
    if src.startswith("/content/"):
        if not any(s.get("session_id") == sid for s in sessions):
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
        elif fetch_sleep_s > 0:
            time.sleep(fetch_sleep_s)
        if not vm_src.is_file():
            raise SystemExit(1)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(vm_src, dst)
    elif dst.startswith("/content/"):
        vm_dst = vm_fs / dst.removeprefix("/content/")
        vm_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, vm_dst)
    raise SystemExit(0)

if subcmd == "exec":
    s_idx = args.index("-s")
    sid = args[s_idx + 1]
    sep_idx = args.index("--")
    remote_cmd = " ".join(args[sep_idx + 1:])
    mapped_cmd = remote_cmd.replace("/content", str(vm_fs))
    sleep_s = float(os.environ.get("FAKE_COLAB_EXEC_SLEEP", "0"))
    if "vm_udlab.py" in mapped_cmd and sleep_s > 0:
        time.sleep(sleep_s)
    proc = subprocess.run(["bash", "-c", mapped_cmd], check=False)
    raise SystemExit(proc.returncode)

if subcmd == "stop":
    s_idx = args.index("-s")
    sid = args[s_idx + 1]
    stopped_file = state_dir / "stopped_sessions.jsonl"
    with stopped_file.open("a", encoding="utf-8", newline="\\n") as fh:
        fh.write(json.dumps({"session_id": sid, "ts": time.time()}) + "\\n")
    sessions = [s for s in sessions if s.get("session_id") != sid]
    sessions_file.write_text(json.dumps(sessions), encoding="utf-8", newline="\\n")
    print(json.dumps({"stopped": sid}))
    raise SystemExit(0)

raise SystemExit(2)
""",
        encoding="utf-8",
        newline="\n",
    )

    fake_colab_sh = fake_bin / "colab"
    fake_colab_sh.write_text(
        f"#!/usr/bin/env bash\nexec python3 '{_to_wsl_path(fake_driver_py)}' \"$@\"\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_colab_sh.chmod(fake_colab_sh.stat().st_mode | stat.S_IEXEC)

    return {
        "fake_bin": fake_bin,
        "sentinel_bin": sentinel_bin,
        "state_dir": state_dir,
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
        f"bash '{wsl_launch}' --stub --out-dir '{wsl_out_dir}' --run-id test-stub-window"
    )
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
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

    # Result fetch (`cp` from /content/result.tar.gz) happened BEFORE `stop`
    stop_idx = next(i for i, c in enumerate(calls) if c["subcmd"] == "stop")
    fetch_idx = next(
        i
        for i, c in enumerate(calls)
        if c["subcmd"] == "cp" and "/content/result.tar.gz" in c["args"]
    )
    assert fetch_idx < stop_idx

    # Only own session was stopped; neighbour session remains active
    stopped = [
        json.loads(line)["session_id"]
        for line in env_paths["stopped_file"].read_text(encoding="utf-8").splitlines()
    ]
    assert stopped == ["sess-udlab-test-001"]
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
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert not env_paths["sentinel_called"].exists()

    # Session was still stopped in trap, and neighbour session was untouched
    stopped = [
        json.loads(line)["session_id"]
        for line in env_paths["stopped_file"].read_text(encoding="utf-8").splitlines()
    ]
    assert stopped == ["sess-udlab-test-001"]
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
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert not env_paths["sentinel_called"].exists()

    stopped = [
        json.loads(line)["session_id"]
        for line in env_paths["stopped_file"].read_text(encoding="utf-8").splitlines()
    ]
    assert "sess-udlab-test-001" in stopped
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
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
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
    monkeypatch.setenv("PATH", str(tmp_path))
    gpu_info = vm_udlab.probe_vm_gpu("stub")
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
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
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
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
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
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
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
        "import shutil, sys\n"
        "if sys.argv and str(sys.argv[0]).endswith('vm_udlab.py'):\n"
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
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
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
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert not env_paths["sentinel_called"].exists()

    calls = [
        json.loads(line)
        for line in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()
    ]
    fetch_idx = next(
        i
        for i, c in enumerate(calls)
        if c["subcmd"] == "cp" and "/content/result.tar.gz" in c["args"]
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
    proc_hang = subprocess.run(["bash", "-c", cmd_hang], capture_output=True, text=True, check=False)
    assert proc_hang.returncode != 0
    assert not env_paths_hang["sentinel_called"].exists()

    calls_hang = [
        json.loads(line)
        for line in env_paths_hang["calls_log"].read_text(encoding="utf-8").splitlines()
    ]
    fetch_hang_idx = next(
        i
        for i, c in enumerate(calls_hang)
        if c["subcmd"] == "cp" and "/content/result.tar.gz" in c["args"]
    )
    stop_hang_idx = next(i for i, c in enumerate(calls_hang) if c["subcmd"] == "stop")
    assert fetch_hang_idx < stop_hang_idx
    stopped_hang = [
        json.loads(line)["session_id"]
        for line in env_paths_hang["stopped_file"].read_text(encoding="utf-8").splitlines()
    ]
    assert "sess-udlab-test-001" in stopped_hang


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
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
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
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
    assert not env_paths["sentinel_called"].exists()

    calls = [
        json.loads(line)
        for line in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()
    ]
    result_tar_cp_calls = [
        c for c in calls if c["subcmd"] == "cp" and "/content/result.tar.gz" in c["args"]
    ]
    # Exactly one fetch of result.tar.gz runs
    assert len(result_tar_cp_calls) == 1

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




