#!/usr/bin/env python3
"""VM-side runner for the understanding-dataset-lab Colab scenario writer window.

Installs pinned Ollama 0.40.13, pulls and verifies qwen3.6:35b manifest and
layer blob digests, builds llama-tokenize at pinned llama.cpp commit, proves
the exact-count gate on the 4 recorded fixture prompts (and deletes the prompt
files before any writer call), runs udlab-gen under a hard wall timeout,
verifies log hygiene, and packs only allow-listed sealed/count artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

OLLAMA_VERSION = "0.40.13"
OLLAMA_RELEASE_URL = (
    "https://github.com/ollama/ollama/releases/download/"
    f"v{OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
)
OLLAMA_ARCHIVE_SHA256 = (
    "c9490e0f520be21271d969e6703f012268b1d39f8b70dd58854e5bfef7a75bb6"
)

WRITER_MODEL = "qwen3.6:35b"
EXPECTED_MANIFEST_DIGEST = (
    "a7eb95c53bcf96b4bdd008d0fab4a5dac88047d9c1a7a9ab88ed453423fbd87c"
)
EXPECTED_MODEL_LAYER_DIGEST = (
    "d372de8e934898a59e6ccfabc3368474711384d8f1fd4d22d87a3f0a45400cdc"
)
EXPECTED_MODEL_LAYER_BLOB = f"sha256-{EXPECTED_MODEL_LAYER_DIGEST}"

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_COMMIT = "50569eb87df530daff11afda229ceb9ab8e6cae8"

EXPECTED_EXACT_COUNTS: tuple[int, ...] = (3124, 3423, 3393, 3394)
MAX_LOG_LINE_CHARS = 2000

PROXY_TOKEN_RE = re.compile(r"colab-runtime-proxy-token=[^\s\"'&;]+")
UNREDACTED_PROXY_TOKEN_RE = re.compile(
    r"colab-runtime-proxy-token=(?!\[REDACTED\])[^\s\"'&;]+"
)

TAR_ALLOWED_EXACT_FILES: frozenset[str] = frozenset(
    {
        "generations.jsonl",
        "build_summary.json",
        "checkpoint.json",
        "vm_receipt.json",
        "sealed_store/keys.sealed.jsonl",
        "sealed_store/sealing.sealed.json",
        "sealed_store/checkpoint.sealed.json",
        "sealed_store/responses.sealed.jsonl",
        "part_public/items.jsonl",
        "part_public/part.json",
    }
)
TAR_ALLOWED_DIRS: frozenset[str] = frozenset(
    {
        "sealed_store",
        "part_public",
        "logs",
    }
)


class RefusalError(RuntimeError):
    """Hard-stop failure with a named failure class."""

    def __init__(self, failure_class: str, message: str) -> None:
        super().__init__(f"{failure_class}: {message}")
        self.failure_class = failure_class
        self.detail = message


def ensure_not_drive_path(path: Path) -> None:
    """Refuse any path located on Google Drive."""
    norm = str(path).replace("\\", "/").lower()
    if "/content/drive" in norm or "/gdrive" in norm:
        raise RefusalError(
            "FORBIDDEN_STORAGE_PATH",
            f"Google Drive paths are forbidden: {path}",
        )


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scrub_log_text(text: str) -> str:
    """Scrub colab-runtime-proxy-token= values from CLI or VM log text."""
    return PROXY_TOKEN_RE.sub("colab-runtime-proxy-token=[REDACTED]", text)


def scrub_log_file_in_place(path: Path) -> None:
    """Scrub proxy tokens in an existing UTF-8 log file."""
    if not path.is_file():
        return
    raw = path.read_text(encoding="utf-8", errors="replace")
    scrubbed = scrub_log_text(raw)
    if scrubbed != raw:
        path.write_text(scrubbed, encoding="utf-8", newline="\n")


def verify_logs_clean(
    log_paths: Iterable[Path],
    max_line_chars: int = MAX_LOG_LINE_CHARS,
) -> None:
    """Fail the window if any log has a line > max_line_chars, response_text, or token."""
    for path in log_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if len(line) > max_line_chars:
                raise RefusalError(
                    "LOG_POLICY_VIOLATION",
                    f"{path.name}:{lineno} length {len(line)} exceeds {max_line_chars}",
                )
            if "response_text" in line:
                raise RefusalError(
                    "LOG_POLICY_VIOLATION",
                    f"{path.name}:{lineno} contains forbidden key 'response_text'",
                )
            if UNREDACTED_PROXY_TOKEN_RE.search(line):
                raise RefusalError(
                    "LOG_POLICY_VIOLATION",
                    f"{path.name}:{lineno} contains unredacted colab-runtime-proxy-token=",
                )


def is_allowed_tar_member(rel_posix_path: str) -> bool:
    """Return True only if rel_posix_path is on the result tar allow-list."""
    cleaned = rel_posix_path.strip()
    if not cleaned or cleaned.startswith("/") or "\\" in cleaned:
        return False
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.rstrip("/")
    if not cleaned:
        return False
    parts = cleaned.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return False
    if cleaned.endswith(".prompt.txt"):
        return False
    if cleaned in TAR_ALLOWED_DIRS:
        return True
    if cleaned in TAR_ALLOWED_EXACT_FILES:
        return True
    if len(parts) == 2 and parts[0] == "logs" and parts[1].endswith(".log"):
        return True
    return False


def validate_tar_members(member_names: Iterable[str]) -> None:
    """Raise RefusalError(TAR_ALLOWLIST_VIOLATION) if any member is outside allow-list."""
    for name in member_names:
        if not is_allowed_tar_member(name):
            raise RefusalError(
                "TAR_ALLOWLIST_VIOLATION",
                f"Path outside result tar allow-list: {name!r}",
            )


def validate_tar_file(tar_path: Path) -> list[str]:
    """Inspect a tar archive and refuse any disallowed member or link/device."""
    ensure_not_drive_path(tar_path)
    names: list[str] = []
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            if member.issym() or member.islnk() or member.ischr() or member.isfifo() or member.isblk():
                raise RefusalError(
                    "TAR_ALLOWLIST_VIOLATION",
                    f"Non-regular tar member forbidden: {member.name!r}",
                )
            norm_name = member.name
            while norm_name.startswith("./"):
                norm_name = norm_name[2:]
            norm_name = norm_name.rstrip("/")
            if not norm_name:
                continue
            if not is_allowed_tar_member(norm_name):
                raise RefusalError(
                    "TAR_ALLOWLIST_VIOLATION",
                    f"Tar member outside allow-list: {member.name!r}",
                )
            names.append(norm_name)
    return names


def build_allowlisted_result_tar(run_dir: Path, tar_path: Path) -> list[str]:
    """Pack run_dir into tar_path, refusing if any file in run_dir violates allow-list."""
    ensure_not_drive_path(run_dir)
    ensure_not_drive_path(tar_path)
    rel_files: list[str] = []
    for item in sorted(run_dir.rglob("*")):
        rel = item.relative_to(run_dir).as_posix()
        if item.is_symlink():
            raise RefusalError(
                "TAR_ALLOWLIST_VIOLATION",
                f"Symlink forbidden in run_dir: {rel!r}",
            )
        if item.is_dir():
            if not is_allowed_tar_member(rel):
                raise RefusalError(
                    "TAR_ALLOWLIST_VIOLATION",
                    f"Directory outside allow-list in run_dir: {rel!r}",
                )
            continue
        if not is_allowed_tar_member(rel):
            raise RefusalError(
                "TAR_ALLOWLIST_VIOLATION",
                f"File outside allow-list in run_dir: {rel!r}",
            )
        rel_files.append(rel)

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tf:
        for rel in rel_files:
            tf.add(run_dir / rel, arcname=rel, recursive=False)
    validate_tar_file(tar_path)
    return rel_files


def verify_ollama_archive_and_version(
    archive_path: Path,
    version_output: str,
    expected_sha256: str = OLLAMA_ARCHIVE_SHA256,
    expected_version: str = OLLAMA_VERSION,
) -> tuple[str, str]:
    """Verify the downloaded Ollama archive SHA-256 and reported version string."""
    if not archive_path.is_file():
        raise RefusalError(
            "OLLAMA_ARCHIVE_SHA256_MISMATCH",
            f"Ollama archive missing at {archive_path}",
        )
    actual_sha256 = sha256_file(archive_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RefusalError(
            "OLLAMA_ARCHIVE_SHA256_MISMATCH",
            f"expected {expected_sha256}, got {actual_sha256}",
        )
    cleaned_version = version_output.strip()
    if expected_version not in cleaned_version:
        raise RefusalError(
            "OLLAMA_VERSION_MISMATCH",
            f"expected version {expected_version} in {cleaned_version!r}",
        )
    return actual_sha256, cleaned_version


def verify_writer_model_digests(
    models_root: Path,
    model: str = WRITER_MODEL,
    expected_manifest_digest: str = EXPECTED_MANIFEST_DIGEST,
    expected_layer_blob: str = EXPECTED_MODEL_LAYER_BLOB,
    verify_blob_sha256: bool = True,
) -> tuple[str, str, Path]:
    """Verify Ollama manifest digest and model layer blob for qwen3.6:35b."""
    ensure_not_drive_path(models_root)
    name, _, tag = model.partition(":")
    if not tag:
        tag = "latest"
    manifest_path = (
        models_root / "manifests" / "registry.ollama.ai" / "library" / name / tag
    )
    if not manifest_path.is_file():
        raise RefusalError(
            "MANIFEST_DIGEST_MISMATCH",
            f"Manifest missing at {manifest_path}",
        )

    actual_manifest_digest = sha256_file(manifest_path)
    if actual_manifest_digest.lower() != expected_manifest_digest.lower():
        raise RefusalError(
            "MANIFEST_DIGEST_MISMATCH",
            f"expected {expected_manifest_digest}, got {actual_manifest_digest}",
        )

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RefusalError(
            "MANIFEST_DIGEST_MISMATCH",
            f"Invalid manifest JSON at {manifest_path}: {exc}",
        ) from exc

    expected_colon_digest = expected_layer_blob.replace("sha256-", "sha256:", 1)
    layers = manifest_data.get("layers") or []
    model_layers = [
        layer
        for layer in layers
        if isinstance(layer, dict)
        and layer.get("mediaType") == "application/vnd.ollama.image.model"
    ]
    if not model_layers:
        raise RefusalError(
            "MODEL_LAYER_DIGEST_MISMATCH",
            "No application/vnd.ollama.image.model layer in manifest",
        )
    recorded_layer_digest = str(model_layers[0].get("digest", ""))
    if recorded_layer_digest.lower() != expected_colon_digest.lower():
        raise RefusalError(
            "MODEL_LAYER_DIGEST_MISMATCH",
            f"manifest layer digest {recorded_layer_digest!r} != {expected_colon_digest!r}",
        )

    blob_path = models_root / "blobs" / expected_layer_blob
    if not blob_path.is_file():
        raise RefusalError(
            "MODEL_LAYER_DIGEST_MISMATCH",
            f"Model layer blob missing at {blob_path}",
        )

    if verify_blob_sha256:
        expected_hex = expected_layer_blob.removeprefix("sha256-")
        actual_blob_hex = sha256_file(blob_path)
        if actual_blob_hex.lower() != expected_hex.lower():
            raise RefusalError(
                "MODEL_LAYER_DIGEST_MISMATCH",
                f"blob sha256 {actual_blob_hex} != {expected_hex}",
            )

    return actual_manifest_digest, expected_layer_blob, blob_path


def compute_prompt_messages_sha256(raw_prompt_text: str) -> str:
    """Compute prompt_sha256 the exact way udlab_generator and test 91 record it."""
    messages = [{"role": "user", "content": raw_prompt_text}]
    canonical = json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def run_exact_count_gate(
    index_path: Path,
    prompts_dir: Path,
    llama_tokenize_bin: Path | None = None,
    gguf_path: Path | None = None,
    count_fn: Callable[[str], int] | None = None,
) -> list[dict[str, object]]:
    """Verify the 4 recorded prompts hash and tokenize to 3124, 3423, 3393, 3394, then delete them.

    Never logs or returns prompt text. Always removes all .prompt.txt files in
    prompts_dir before returning or raising.
    """
    ensure_not_drive_path(index_path)
    ensure_not_drive_path(prompts_dir)

    try:
        if not index_path.is_file():
            raise RefusalError(
                "EXACT_COUNT_INDEX_MISMATCH",
                f"Index file missing: {index_path}",
            )
        entries = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(entries, list) or len(entries) != len(EXPECTED_EXACT_COUNTS):
            raise RefusalError(
                "EXACT_COUNT_INDEX_MISMATCH",
                f"Expected {len(EXPECTED_EXACT_COUNTS)} index entries, got {len(entries) if isinstance(entries, list) else type(entries).__name__}",
            )
        recorded_counts = tuple(int(e.get("prompt_eval_count", -1)) for e in entries)
        if recorded_counts != EXPECTED_EXACT_COUNTS:
            raise RefusalError(
                "EXACT_COUNT_INDEX_MISMATCH",
                f"Index counts {recorded_counts} != expected {EXPECTED_EXACT_COUNTS}",
            )

        if count_fn is None:
            if llama_tokenize_bin is None or gguf_path is None:
                raise RefusalError(
                    "EXACT_COUNT_TOKEN_MISMATCH",
                    "llama_tokenize_bin and gguf_path are required when count_fn is None",
                )
            from udlab_generator.render import (
                DEFAULT_WRITER_MODEL_PIN,
                LlamaTokenizeCounter,
                count_prompt_tokens_exact,
            )

            counter = LlamaTokenizeCounter(Path(llama_tokenize_bin), Path(gguf_path))
            counter.verify_pinned_digest(DEFAULT_WRITER_MODEL_PIN.model_layer_digest)

            def _default_count_fn(raw_prompt_text: str) -> int:
                messages = [{"role": "user", "content": raw_prompt_text}]
                return count_prompt_tokens_exact(
                    messages,
                    think=False,
                    counter=counter,
                )

            count_fn = _default_count_fn

        verified: list[dict[str, object]] = []
        for idx, entry in enumerate(entries):
            prompt_file = str(entry.get("file") or entry.get("prompt_file") or "")
            if not prompt_file:
                raise RefusalError(
                    "EXACT_COUNT_INDEX_MISMATCH",
                    f"Missing 'file' key in index entry {idx}",
                )
            prompt_path = prompts_dir / prompt_file
            if not prompt_path.is_file():
                raise RefusalError(
                    "EXACT_COUNT_PROMPT_MISSING",
                    f"Missing prompt file for entry {idx}: {prompt_file}",
                )
            raw_prompt = prompt_path.read_bytes().decode("utf-8")
            actual_sha = compute_prompt_messages_sha256(raw_prompt)
            expected_sha = str(entry["prompt_sha256"])
            if actual_sha.lower() != expected_sha.lower():
                raise RefusalError(
                    "EXACT_COUNT_SHA256_MISMATCH",
                    f"Prompt {prompt_file} sha256 {actual_sha} != {expected_sha}",
                )
            actual_count = int(count_fn(raw_prompt))
            expected_count = int(entry["prompt_eval_count"])
            if actual_count != expected_count:
                raise RefusalError(
                    "EXACT_COUNT_TOKEN_MISMATCH",
                    f"Prompt {prompt_file} token count {actual_count} != expected {expected_count}",
                )
            verified.append(
                {
                    "run": entry.get("run"),
                    "call_id": entry.get("call_id"),
                    "file": prompt_file,
                    "prompt_sha256": actual_sha,
                    "expected_count": expected_count,
                    "actual_count": actual_count,
                }
            )
    finally:
        if prompts_dir.exists():
            for p in list(prompts_dir.glob("*.prompt.txt")):
                try:
                    p.unlink()
                except OSError:
                    pass
            try:
                if not any(prompts_dir.iterdir()):
                    prompts_dir.rmdir()
            except OSError:
                pass

    remaining = list(prompts_dir.glob("*.prompt.txt")) if prompts_dir.exists() else []
    if remaining:
        raise RefusalError(
            "EXACT_COUNT_CLEANUP_FAILED",
            f"Failed to delete {len(remaining)} prompt file(s) from {prompts_dir}",
        )
    return verified


def install_pinned_ollama(work_dir: Path, log_fh) -> tuple[str, str]:
    """Install system packages and pinned Ollama 0.40.13 archive, checking SHA-256."""
    ensure_not_drive_path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "apt-get",
            "update",
            "-y",
        ],
        check=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "zstd",
            "curl",
            "ca-certificates",
            "cmake",
            "build-essential",
            "git",
            "pciutils",
            "lshw",
        ],
        check=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    archive_path = work_dir / f"ollama-linux-amd64-v{OLLAMA_VERSION}.tar.zst"
    if not archive_path.is_file():
        urllib.request.urlretrieve(OLLAMA_RELEASE_URL, archive_path)

    actual_sha = sha256_file(archive_path)
    if actual_sha.lower() != OLLAMA_ARCHIVE_SHA256.lower():
        raise RefusalError(
            "OLLAMA_ARCHIVE_SHA256_MISMATCH",
            f"expected {OLLAMA_ARCHIVE_SHA256}, got {actual_sha}",
        )

    subprocess.run(
        [
            "tar",
            "--use-compress-program=unzstd",
            "-xf",
            str(archive_path),
            "-C",
            "/usr/local",
        ],
        check=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    proc = subprocess.run(
        ["/usr/local/bin/ollama", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    version_out = (proc.stdout or "") + (proc.stderr or "")
    _, cleaned_ver = verify_ollama_archive_and_version(
        archive_path=archive_path,
        version_output=version_out,
    )
    return actual_sha, cleaned_ver


def pull_and_verify_writer_model(
    models_root: Path,
    ollama_log_path: Path,
    vm_log_fh,
) -> tuple[str, str, Path]:
    """Start Ollama temporarily to pull qwen3.6:35b, stop it, and verify digests."""
    ensure_not_drive_path(models_root)
    models_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = str(models_root)
    env["OLLAMA_HOST"] = "127.0.0.1:11434"

    with ollama_log_path.open("a", encoding="utf-8") as ol_fh:
        srv = subprocess.Popen(
            ["/usr/local/bin/ollama", "serve"],
            env=env,
            stdout=ol_fh,
            stderr=subprocess.STDOUT,
        )
        try:
            time.sleep(3)
            subprocess.run(
                ["/usr/local/bin/ollama", "pull", WRITER_MODEL],
                env=env,
                check=True,
                stdout=vm_log_fh,
                stderr=subprocess.STDOUT,
            )
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=10)
            except subprocess.TimeoutExpired:
                srv.kill()
                srv.wait(timeout=5)

    return verify_writer_model_digests(
        models_root=models_root,
        model=WRITER_MODEL,
        expected_manifest_digest=EXPECTED_MANIFEST_DIGEST,
        expected_layer_blob=EXPECTED_MODEL_LAYER_BLOB,
        verify_blob_sha256=True,
    )


def build_llama_tokenize(build_root: Path, log_fh) -> Path:
    """Clone llama.cpp at pinned commit 50569eb8... and build CPU llama-tokenize."""
    ensure_not_drive_path(build_root)
    src_dir = build_root / "llama.cpp"
    bin_path = src_dir / "build" / "bin" / "llama-tokenize"
    if bin_path.is_file():
        head = (
            subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        if head.lower() == LLAMA_CPP_COMMIT.lower():
            return bin_path
        shutil.rmtree(src_dir)

    build_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", LLAMA_CPP_REPO, str(src_dir)],
        check=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        ["git", "-C", str(src_dir), "checkout", "--force", LLAMA_CPP_COMMIT],
        check=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    actual_head = (
        subprocess.check_output(["git", "-C", str(src_dir), "rev-parse", "HEAD"])
        .decode("utf-8")
        .strip()
    )
    if actual_head.lower() != LLAMA_CPP_COMMIT.lower():
        raise RefusalError(
            "LLAMA_CPP_COMMIT_MISMATCH",
            f"expected {LLAMA_CPP_COMMIT}, got {actual_head}",
        )

    subprocess.run(
        [
            "cmake",
            "-S",
            str(src_dir),
            "-B",
            str(src_dir / "build"),
            "-DGGML_CUDA=OFF",
            "-DBUILD_SHARED_LIBS=OFF",
        ],
        check=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(src_dir / "build"),
            "--target",
            "llama-tokenize",
            "-j",
            str(max(1, (os.cpu_count() or 2))),
        ],
        check=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    if not bin_path.is_file():
        raise RefusalError(
            "LLAMA_CPP_COMMIT_MISMATCH",
            f"llama-tokenize binary missing after build at {bin_path}",
        )
    return bin_path


def run_dryrun_pipeline(args: argparse.Namespace, logs_dir: Path, vm_log_path: Path) -> dict[str, object]:
    """Execute deterministic dry-run pipeline for rig fake-CLI testing (no GPU/network)."""
    inject_failure = os.environ.get("UDLAB_DRYRUN_INJECT_FAILURE", "").strip()
    if inject_failure:
        raise RefusalError(inject_failure, f"Injected dry-run failure: {inject_failure}")

    index_path = args.index_path or (
        args.repo_dir / "tests" / "fixtures" / "exact_ctx" / "index.json"
    )
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    count_map = {
        str(e["prompt_sha256"]).lower(): int(e["prompt_eval_count"]) for e in entries
    }

    def _dryrun_counter(raw_prompt: str) -> int:
        sha = compute_prompt_messages_sha256(raw_prompt).lower()
        return count_map.get(sha, -1)

    verified_gate = run_exact_count_gate(
        index_path=index_path,
        prompts_dir=args.prompts_dir,
        count_fn=_dryrun_counter,
    )

    sealed_dir = args.run_dir / "sealed_store"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    (sealed_dir / "keys.sealed.jsonl").write_text("", encoding="utf-8", newline="\n")
    (sealed_dir / "sealing.sealed.json").write_text("{\"sealed\":true}\n", encoding="utf-8", newline="\n")
    (sealed_dir / "checkpoint.sealed.json").write_text("{\"completed\":3}\n", encoding="utf-8", newline="\n")
    (sealed_dir / "responses.sealed.jsonl").write_text("", encoding="utf-8", newline="\n")

    part_dir = args.run_dir / "part_public"
    part_dir.mkdir(parents=True, exist_ok=True)
    (part_dir / "items.jsonl").write_text("", encoding="utf-8", newline="\n")
    (part_dir / "part.json").write_text("{\"items\":3}\n", encoding="utf-8", newline="\n")

    (args.run_dir / "generations.jsonl").write_text("", encoding="utf-8", newline="\n")
    (args.run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "count": args.count,
                "base_seed": args.seed,
                "writer": args.writer,
                "resume": bool(args.resume),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.run_dir / "build_summary.json").write_text(
        json.dumps(
            {
                "count": args.count,
                "seed": args.seed,
                "writer": args.writer,
                "grant_id": args.grant_id,
                "vram_halt_mib": args.vram_halt_mib,
                "dryrun": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (logs_dir / "udlab-gen.log").write_text(
        f"udlab-gen dryrun count={args.count} seed={args.seed} writer={args.writer}\n",
        encoding="utf-8",
        newline="\n",
    )
    (logs_dir / "ollama-serve.log").write_text(
        "ollama serve dryrun\n",
        encoding="utf-8",
        newline="\n",
    )
    with vm_log_path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(
            f"vm_udlab dryrun ok mode={args.mode} writer={args.writer} count={args.count} seed={args.seed}\n"
        )

    return {
        "status": "ok",
        "failure_class": None,
        "mode": args.mode,
        "writer": args.writer,
        "count": args.count,
        "seed": args.seed,
        "grant_id": args.grant_id,
        "vram_halt_mib": args.vram_halt_mib,
        "resume": bool(args.resume),
        "ollama_version": f"ollama version is {OLLAMA_VERSION}",
        "ollama_archive_sha256": OLLAMA_ARCHIVE_SHA256,
        "writer_model": WRITER_MODEL,
        "manifest_digest": EXPECTED_MANIFEST_DIGEST,
        "model_layer_blob": EXPECTED_MODEL_LAYER_BLOB,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "exact_count_gate": verified_gate,
        "dryrun": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VM-side udlab Colab runner")
    parser.add_argument("--mode", choices=("gpu", "stub"), required=True)
    parser.add_argument("--writer", choices=("ollama", "mock"), required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--grant-id", required=True)
    parser.add_argument("--vram-halt-mib", type=int, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--prompts-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/udlab_work"))
    parser.add_argument("--models-root", type=Path, default=Path("/root/.ollama/models"))
    parser.add_argument("--result-tar", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dryrun", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for p in (args.repo_dir, args.prompts_dir, args.run_dir, args.work_dir, args.result_tar):
        ensure_not_drive_path(p)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    vm_log_path = logs_dir / "vm.log"
    ollama_log_path = logs_dir / "ollama-serve.log"
    gen_log_path = logs_dir / "udlab-gen.log"
    receipt_path = args.run_dir / "vm_receipt.json"

    dryrun = args.dryrun or os.environ.get("VM_DRYRUN") == "1"

    try:
        if dryrun:
            receipt = run_dryrun_pipeline(args, logs_dir, vm_log_path)
        else:
            with vm_log_path.open("a", encoding="utf-8", newline="\n") as vm_log_fh:
                archive_sha, ollama_ver = install_pinned_ollama(args.work_dir, vm_log_fh)
                manifest_digest, layer_blob, blob_path = pull_and_verify_writer_model(
                    models_root=args.models_root,
                    ollama_log_path=ollama_log_path,
                    vm_log_fh=vm_log_fh,
                )
                llama_tokenize_bin = build_llama_tokenize(args.work_dir, vm_log_fh)

            index_path = args.index_path or (
                args.repo_dir / "tests" / "fixtures" / "exact_ctx" / "index.json"
            )
            verified_gate = run_exact_count_gate(
                index_path=index_path,
                prompts_dir=args.prompts_dir,
                llama_tokenize_bin=llama_tokenize_bin,
                gguf_path=blob_path,
            )

            gen_cmd = [
                "uv",
                "run",
                "udlab-gen",
                "--count",
                str(args.count),
                "--seed",
                str(args.seed),
                "--out-dir",
                str(args.run_dir),
                "--write",
                "--writer",
                args.writer,
                "--grant-id",
                args.grant_id,
                "--vram-halt-mib",
                str(args.vram_halt_mib),
                "--llama-tokenize",
                str(llama_tokenize_bin),
                "--writer-gguf",
                str(blob_path),
            ]
            if args.resume or (args.run_dir / "checkpoint.json").is_file():
                gen_cmd.append("--resume")

            gen_env = os.environ.copy()
            gen_env["OLLAMA_MODELS"] = str(args.models_root)
            if args.mode == "stub":
                gen_env["CUDA_VISIBLE_DEVICES"] = "-1"

            with gen_log_path.open("a", encoding="utf-8", newline="\n") as gen_fh:
                try:
                    proc = subprocess.run(
                        gen_cmd,
                        cwd=str(args.repo_dir),
                        env=gen_env,
                        stdout=gen_fh,
                        stderr=subprocess.STDOUT,
                        timeout=max(1, args.wall_seconds),
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RefusalError(
                        "GENERATOR_WALL_TIMEOUT",
                        f"udlab-gen exceeded VM wall timeout of {args.wall_seconds}s",
                    ) from exc

            if proc.returncode != 0:
                raise RefusalError(
                    "GENERATOR_FAILED",
                    f"udlab-gen exited with code {proc.returncode}",
                )

            receipt = {
                "status": "ok",
                "failure_class": None,
                "mode": args.mode,
                "writer": args.writer,
                "count": args.count,
                "seed": args.seed,
                "grant_id": args.grant_id,
                "vram_halt_mib": args.vram_halt_mib,
                "resume": bool(args.resume),
                "ollama_version": ollama_ver,
                "ollama_archive_sha256": archive_sha,
                "writer_model": WRITER_MODEL,
                "manifest_digest": manifest_digest,
                "model_layer_blob": layer_blob,
                "llama_cpp_commit": LLAMA_CPP_COMMIT,
                "exact_count_gate": verified_gate,
                "dryrun": False,
            }

        receipt_path.write_text(
            json.dumps(receipt, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        for log_file in logs_dir.glob("*.log"):
            scrub_log_file_in_place(log_file)
        verify_logs_clean(logs_dir.glob("*.log"))
        build_allowlisted_result_tar(args.run_dir, args.result_tar)
        return 0

    except RefusalError as err:
        fail_receipt = {
            "status": "refused",
            "failure_class": err.failure_class,
            "detail": scrub_log_text(err.detail),
            "mode": args.mode,
            "writer": args.writer,
            "count": args.count,
            "seed": args.seed,
        }
        receipt_path.write_text(
            json.dumps(fail_receipt, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with vm_log_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"REFUSAL {err.failure_class}: {scrub_log_text(err.detail)}\n")
        for log_file in logs_dir.glob("*.log"):
            scrub_log_file_in_place(log_file)
        sys.stderr.write(f"{err.failure_class}: {scrub_log_text(err.detail)}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
