#!/usr/bin/env python3
"""VM-side runner for the understanding-dataset-lab Colab scenario writer window.

Installs pinned Ollama 0.40.13 and pinned uv 0.13.4 (checking archive SHA-256),
installs Python 3.15 and syncs the repo from uv.lock, records GPU info from
nvidia-smi (or CPU in stub mode), pulls and verifies qwen3.6:35b manifest and
layer blob digests, builds llama-tokenize at pinned llama.cpp commit, proves
the exact-count gate on the 4 recorded fixture prompts inside the uv Python
environment (and deletes the prompt files before any writer call), verifies
100% GPU model offload via a short-lived Ollama warm-up server in GPU mode,
runs udlab-gen under a hard wall timeout, verifies log hygiene, and packs only
allow-listed sealed/count artifacts.
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
import urllib.error
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

UV_VERSION = "0.13.4"
UV_RELEASE_URL = (
    "https://github.com/astral-sh/uv/releases/download/"
    f"{UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
)
UV_ARCHIVE_SHA256 = (
    "2b428396f12572d4429cf210ca234dd599a33b3af0958a4d5ed569d63e3f344f"
)
REQUIRED_PYTHON_VERSION = "3.15"

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

WARMUP_OLLAMA_HOST = "127.0.0.1:11435"
NVIDIA_SMI_QUERY_CMD: tuple[str, ...] = (
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader",
)

# A change to tests/fixtures/exact_ctx/index.json must update this tuple in the same PR.
EXPECTED_EXACT_COUNTS: tuple[int, ...] = (3124, 3423, 3393, 3394)
MAX_LOG_LINE_CHARS = 2000
MIN_PROMPT_LINE_HASH_CHARS = 40
LOG_SEGMENT_SPLIT_RE = re.compile(r"(?:[.!?;]|\t)+")
MIN_FREE_DISK_BYTES = 40 * 1024 * 1024 * 1024  # 40 GiB: 22 GB model + hashing + build

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


def normalize_whitespace_line(text: str) -> str:
    """Collapse consecutive whitespace to a single space and strip leading/trailing whitespace."""
    return " ".join(text.split())


def collect_prompt_line_sha256s(
    raw_prompt_text: str,
    min_chars: int = MIN_PROMPT_LINE_HASH_CHARS,
) -> list[str]:
    """Return sorted unique sha256 hex digests of normalized prompt lines >= min_chars."""
    seen: set[str] = set()
    for raw_line in raw_prompt_text.splitlines():
        norm = normalize_whitespace_line(raw_line)
        if len(norm) >= min_chars:
            seen.add(hashlib.sha256(norm.encode("utf-8")).hexdigest())
    return sorted(seen)


def verify_logs_clean(
    log_paths: Iterable[Path],
    max_line_chars: int = MAX_LOG_LINE_CHARS,
    prompt_line_hashes: Iterable[str] | None = None,
    min_prompt_line_chars: int = MIN_PROMPT_LINE_HASH_CHARS,
) -> None:
    """Fail the window if any log has a line > max_line_chars, response_text, token, or prompt line hash."""
    forbidden_hashes: frozenset[str] = (
        frozenset(h.lower() for h in prompt_line_hashes)
        if prompt_line_hashes is not None
        else frozenset()
    )
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
            if forbidden_hashes:
                norm_line = normalize_whitespace_line(line)
                if len(norm_line) >= min_prompt_line_chars:
                    line_sha = hashlib.sha256(norm_line.encode("utf-8")).hexdigest()
                    if line_sha in forbidden_hashes:
                        raise RefusalError(
                            "LOG_POLICY_VIOLATION",
                            f"{path.name}:{lineno} matches forbidden prompt line sha256 {line_sha[:12]}",
                        )
                for seg in LOG_SEGMENT_SPLIT_RE.split(line):
                    norm_seg = normalize_whitespace_line(seg)
                    if len(norm_seg) >= min_prompt_line_chars:
                        seg_sha = hashlib.sha256(norm_seg.encode("utf-8")).hexdigest()
                        if seg_sha in forbidden_hashes:
                            raise RefusalError(
                                "LOG_POLICY_VIOLATION",
                                f"{path.name}:{lineno} segment matches forbidden prompt line sha256 {seg_sha[:12]}",
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


def verify_uv_archive_and_version(
    archive_path: Path,
    version_output: str,
    expected_sha256: str = UV_ARCHIVE_SHA256,
    expected_version: str = UV_VERSION,
) -> tuple[str, str]:
    """Verify the downloaded uv archive SHA-256 and reported version string."""
    if not archive_path.is_file():
        raise RefusalError(
            "UV_ARCHIVE_SHA256_MISMATCH",
            f"uv archive missing at {archive_path}",
        )
    actual_sha256 = sha256_file(archive_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RefusalError(
            "UV_ARCHIVE_SHA256_MISMATCH",
            f"expected {expected_sha256}, got {actual_sha256}",
        )
    cleaned_version = version_output.strip()
    if expected_version not in cleaned_version:
        raise RefusalError(
            "UV_VERSION_MISMATCH",
            f"expected version {expected_version} in {cleaned_version!r}",
        )
    return actual_sha256, cleaned_version


def probe_vm_gpu(
    mode: str,
    allow_missing_in_dryrun: bool = False,
) -> dict[str, object]:
    """Query GPU name, memory.total, and driver_version via nvidia-smi, or record CPU in stub mode."""
    if mode == "stub":
        return {
            "hardware": "CPU",
            "gpu_present": False,
            "gpu_name": "CPU",
            "gpu_memory_total": None,
            "gpu_driver_version": None,
            "nvidia_smi": None,
        }

    if allow_missing_in_dryrun:
        resolved = shutil.which("nvidia-smi")
        norm_resolved = (resolved or "").replace("\\", "/")
        if (
            not resolved
            or norm_resolved.startswith(("/usr/", "/bin/", "/sbin/", "/lib/"))
            or "windows/system32" in norm_resolved.lower()
        ):
            return {
                "hardware": "A100",
                "gpu_present": True,
                "gpu_name": "A100",
                "gpu_memory_total": None,
                "gpu_driver_version": None,
                "nvidia_smi": None,
            }

    try:
        proc = subprocess.run(
            list(NVIDIA_SMI_QUERY_CMD),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        if allow_missing_in_dryrun:
            return {
                "hardware": "A100",
                "gpu_present": True,
                "gpu_name": "A100",
                "gpu_memory_total": None,
                "gpu_driver_version": None,
                "nvidia_smi": None,
            }
        raise RefusalError(
            "GPU_NOT_FOUND",
            f"nvidia-smi failed to execute in GPU mode: {exc}",
        ) from exc

    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if proc.returncode != 0 or not lines:
        if allow_missing_in_dryrun:
            return {
                "hardware": "A100",
                "gpu_present": True,
                "gpu_name": "A100",
                "gpu_memory_total": None,
                "gpu_driver_version": None,
                "nvidia_smi": None,
            }
        raise RefusalError(
            "GPU_NOT_FOUND",
            f"nvidia-smi exited with code {proc.returncode} and output {proc.stdout!r}",
        )

    first_line = lines[0]
    parts = [p.strip() for p in first_line.split(",")]
    gpu_name = parts[0] if parts else first_line
    gpu_memory_total = parts[1] if len(parts) > 1 else None
    gpu_driver_version = parts[2] if len(parts) > 2 else None
    return {
        "hardware": first_line,
        "gpu_present": True,
        "gpu_name": gpu_name,
        "gpu_memory_total": gpu_memory_total,
        "gpu_driver_version": gpu_driver_version,
        "nvidia_smi": first_line,
    }


def verify_model_full_gpu_offload(
    ps_payload: dict[str, object] | str | bytes,
    model: str = WRITER_MODEL,
) -> dict[str, object]:
    """Verify Ollama /api/ps reports the loaded writer model 100% in VRAM (size_vram == size)."""
    if isinstance(ps_payload, (str, bytes)):
        try:
            data = json.loads(ps_payload)
        except Exception as exc:
            raise RefusalError(
                "MODEL_PARTIAL_OFFLOAD",
                f"Invalid JSON from Ollama /api/ps: {exc}",
            ) from exc
    elif isinstance(ps_payload, dict):
        data = ps_payload
    else:
        raise RefusalError(
            "MODEL_PARTIAL_OFFLOAD",
            f"Unexpected /api/ps payload type: {type(ps_payload).__name__}",
        )

    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise RefusalError(
            "MODEL_PARTIAL_OFFLOAD",
            f"No loaded models reported by Ollama /api/ps when checking {model}",
        )

    matched = None
    for entry in models:
        if isinstance(entry, dict) and (
            entry.get("model") == model or entry.get("name") == model
        ):
            matched = entry
            break
    if matched is None and isinstance(models[0], dict):
        matched = models[0]
    if not isinstance(matched, dict):
        raise RefusalError(
            "MODEL_PARTIAL_OFFLOAD",
            f"Malformed model entry in Ollama /api/ps for {model}",
        )

    try:
        size = int(matched.get("size", 0))
        size_vram = int(matched.get("size_vram", 0))
    except (TypeError, ValueError) as exc:
        raise RefusalError(
            "MODEL_PARTIAL_OFFLOAD",
            f"Non-integer size/size_vram in Ollama /api/ps: {matched!r}",
        ) from exc

    if size <= 0:
        raise RefusalError(
            "MODEL_PARTIAL_OFFLOAD",
            f"Model {model} reported non-positive size={size} in /api/ps",
        )
    if size_vram != size:
        raise RefusalError(
            "MODEL_PARTIAL_OFFLOAD",
            f"Model {model} not 100% on GPU: size_vram={size_vram} != size={size}",
        )

    return {
        "checked": True,
        "model": str(matched.get("model") or matched.get("name") or model),
        "size": size,
        "size_vram": size_vram,
        "gpu_offload_percent": 100.0,
    }


def check_ollama_server_full_gpu_offload(
    base_url: str = f"http://{WARMUP_OLLAMA_HOST}",
    model: str = WRITER_MODEL,
    timeout_s: float = 120.0,
) -> dict[str, object]:
    """Warm-load model on an Ollama server, check /api/ps for 100% GPU offload, and unload it."""
    base = base_url.rstrip("/")
    load_body = json.dumps(
        {"model": model, "prompt": "", "stream": False}
    ).encode("utf-8")
    unload_body = json.dumps(
        {"model": model, "prompt": "", "keep_alive": 0, "stream": False}
    ).encode("utf-8")

    try:
        load_req = urllib.request.Request(
            f"{base}/api/generate",
            data=load_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(load_req, timeout=timeout_s) as resp:
                resp.read()
        except Exception as exc:
            raise RefusalError(
                "MODEL_PARTIAL_OFFLOAD",
                f"Warm-up load request to {base}/api/generate failed: {exc}",
            ) from exc

        ps_req = urllib.request.Request(f"{base}/api/ps", method="GET")
        try:
            with urllib.request.urlopen(ps_req, timeout=timeout_s) as resp:
                ps_raw = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            raise RefusalError(
                "MODEL_PARTIAL_OFFLOAD",
                f"Failed to query {base}/api/ps: {exc}",
            ) from exc

        return verify_model_full_gpu_offload(ps_raw, model=model)
    finally:
        unload_req = urllib.request.Request(
            f"{base}/api/generate",
            data=unload_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(unload_req, timeout=30.0) as resp:
                resp.read()
        except Exception:
            pass


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


def cleanup_prompts_dir(
    prompts_dir: Path,
    prompts_tar: Path | None = None,
) -> None:
    """Remove all .prompt.txt files, prompts_dir if empty, and optional prompts_tar."""
    if prompts_tar is not None and prompts_tar.exists():
        try:
            prompts_tar.unlink()
        except OSError:
            pass
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


def extract_prompts_tar(prompts_tar: Path, prompts_dir: Path) -> None:
    """Extract prompts_tar into prompts_dir just before the exact-count gate and delete the tar."""
    ensure_not_drive_path(prompts_tar)
    ensure_not_drive_path(prompts_dir)
    if not prompts_tar.is_file():
        raise RefusalError(
            "EXACT_COUNT_PROMPT_MISSING",
            f"Prompts archive missing: {prompts_tar}",
        )
    prompts_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(prompts_tar, "r:*") as tf:
            for member in tf.getmembers():
                if (
                    member.issym()
                    or member.islnk()
                    or ".." in member.name
                    or member.name.startswith("/")
                    or "\\" in member.name
                ):
                    raise RefusalError(
                        "EXACT_COUNT_INDEX_MISMATCH",
                        f"Forbidden prompts tar member: {member.name!r}",
                    )
            tf.extractall(prompts_dir)
    finally:
        try:
            prompts_tar.unlink()
        except OSError:
            pass


def run_exact_count_gate(
    index_path: Path,
    prompts_dir: Path,
    llama_tokenize_bin: Path | None = None,
    gguf_path: Path | None = None,
    count_fn: Callable[[str], int] | None = None,
    repo_dir: Path | None = None,
    uv_bin: Path | str = "uv",
    prompt_line_hashes_out: set[str] | None = None,
    prompts_tar: Path | None = None,
) -> list[dict[str, object]]:
    """Verify the 4 recorded prompts hash and tokenize to 3124, 3423, 3393, 3394, then delete them.

    Never logs or returns prompt text. Always removes all .prompt.txt files in
    prompts_dir before returning or raising.
    """
    ensure_not_drive_path(index_path)
    ensure_not_drive_path(prompts_dir)
    if prompts_tar is not None:
        ensure_not_drive_path(prompts_tar)

    verified: list[dict[str, object]] = []
    prompt_paths_checked: list[Path] = []
    try:
        if prompts_tar is not None and prompts_tar.is_file():
            extract_prompts_tar(prompts_tar, prompts_dir)
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
            effective_repo_dir = repo_dir or Path(__file__).resolve().parents[2]
            ensure_not_drive_path(effective_repo_dir)
            uv_exe = str(uv_bin)
            tok_bin_str = str(Path(llama_tokenize_bin))
            gguf_str = str(Path(gguf_path))
            inline_code = (
                "import sys; from pathlib import Path; "
                "from udlab_generator.render import DEFAULT_WRITER_MODEL_PIN, LlamaTokenizeCounter, count_prompt_tokens_exact; "
                "counter = LlamaTokenizeCounter(Path(sys.argv[1]), Path(sys.argv[2])); "
                "counter.verify_pinned_digest(DEFAULT_WRITER_MODEL_PIN.model_layer_digest); "
                "raw = sys.stdin.buffer.read().decode('utf-8'); "
                "print(count_prompt_tokens_exact([{'role': 'user', 'content': raw}], think=False, counter=counter))"
            )

            def _default_count_fn(raw_prompt_text: str) -> int:
                proc = subprocess.run(
                    [
                        uv_exe,
                        "run",
                        "--frozen",
                        "python",
                        "-c",
                        inline_code,
                        tok_bin_str,
                        gguf_str,
                    ],
                    cwd=str(effective_repo_dir),
                    input=raw_prompt_text.encode("utf-8"),
                    capture_output=True,
                    check=False,
                )
                if proc.returncode != 0:
                    err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
                    raise RefusalError(
                        "EXACT_COUNT_TOKEN_MISMATCH",
                        f"uv run exact-count helper failed (exit {proc.returncode}): {err_msg[:300]}",
                    )
                out_str = proc.stdout.decode("utf-8", errors="replace").strip()
                try:
                    return int(out_str.splitlines()[-1].strip())
                except (IndexError, ValueError) as exc:
                    raise RefusalError(
                        "EXACT_COUNT_TOKEN_MISMATCH",
                        f"Non-integer token count from uv run exact-count helper: {out_str!r}",
                    ) from exc

            count_fn = _default_count_fn

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
            prompt_paths_checked.append(prompt_path)
            raw_bytes = prompt_path.read_bytes()
            raw_file_sha = hashlib.sha256(raw_bytes).hexdigest()
            raw_prompt = raw_bytes.decode("utf-8")
            if prompt_line_hashes_out is not None:
                prompt_line_hashes_out.update(collect_prompt_line_sha256s(raw_prompt))
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
                    "raw_file_sha256": raw_file_sha,
                    "prompt_sha256": actual_sha,
                    "expected_count": expected_count,
                    "actual_count": actual_count,
                    "deleted": False,
                    "absent_verified": False,
                }
            )
    finally:
        cleanup_prompts_dir(prompts_dir, prompts_tar=prompts_tar)

    for rec, p_path in zip(verified, prompt_paths_checked):
        try:
            p_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        rec["deleted"] = True
        still_exists = p_path.exists() or p_path.is_file()
        rec["absent_verified"] = not still_exists
        if still_exists:
            raise RefusalError(
                "EXACT_COUNT_CLEANUP_FAILED",
                f"Prompt file still exists after delete: {p_path.name}",
            )

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


def install_pinned_uv_and_sync_repo(
    work_dir: Path,
    repo_dir: Path,
    log_fh,
    install_dir: Path = Path("/usr/local/bin"),
    expected_sha256: str = UV_ARCHIVE_SHA256,
    expected_version: str = UV_VERSION,
    required_python: str = REQUIRED_PYTHON_VERSION,
) -> tuple[str, str, str, Path]:
    """Install pinned uv 0.13.4 archive (verifying SHA-256), install Python 3.15, and uv sync --frozen."""
    ensure_not_drive_path(work_dir)
    ensure_not_drive_path(repo_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)

    archive_path = work_dir / f"uv-x86_64-unknown-linux-gnu-v{expected_version}.tar.gz"
    if not archive_path.is_file():
        urllib.request.urlretrieve(UV_RELEASE_URL, archive_path)

    actual_sha = sha256_file(archive_path)
    if actual_sha.lower() != expected_sha256.lower():
        raise RefusalError(
            "UV_ARCHIVE_SHA256_MISMATCH",
            f"expected {expected_sha256}, got {actual_sha}",
        )

    subprocess.run(
        [
            "tar",
            "-xzf",
            str(archive_path),
            "--strip-components=1",
            "-C",
            str(install_dir),
        ],
        check=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    uv_bin = install_dir / "uv"
    proc_ver = subprocess.run(
        [str(uv_bin), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    uv_ver_out = (proc_ver.stdout or "") + (proc_ver.stderr or "")
    _, cleaned_uv_ver = verify_uv_archive_and_version(
        archive_path=archive_path,
        version_output=uv_ver_out,
        expected_sha256=expected_sha256,
        expected_version=expected_version,
    )

    subprocess.run(
        [str(uv_bin), "python", "install", required_python],
        cwd=str(repo_dir),
        check=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        [str(uv_bin), "sync", "--frozen"],
        cwd=str(repo_dir),
        check=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    proc_py = subprocess.run(
        [str(uv_bin), "run", "--frozen", "python", "--version"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
        text=True,
    )
    cleaned_py_ver = ((proc_py.stdout or "") + (proc_py.stderr or "")).strip()
    if required_python not in cleaned_py_ver:
        raise RefusalError(
            "PYTHON_VERSION_MISMATCH",
            f"expected Python {required_python} in {cleaned_py_ver!r}",
        )
    return actual_sha, cleaned_uv_ver, cleaned_py_ver, uv_bin


def check_model_disk_space(
    models_root: Path,
    min_free_bytes: int = MIN_FREE_DISK_BYTES,
) -> int:
    """Measure free disk on models_root's filesystem and refuse with DISK_TOO_SMALL below min_free_bytes."""
    ensure_not_drive_path(models_root)
    probe_dir = models_root
    while not probe_dir.exists() and probe_dir.parent != probe_dir:
        probe_dir = probe_dir.parent
    usage = shutil.disk_usage(probe_dir)
    free_bytes = int(usage.free)
    if free_bytes < min_free_bytes:
        raise RefusalError(
            "DISK_TOO_SMALL",
            f"free_bytes={free_bytes} < min_free_bytes={min_free_bytes}",
        )
    return free_bytes


def pull_and_verify_writer_model(
    models_root: Path,
    ollama_log_path: Path,
    vm_log_fh,
) -> tuple[str, str, Path]:
    """Start Ollama temporarily to pull qwen3.6:35b, stop it, and verify digests."""
    ensure_not_drive_path(models_root)
    models_root.mkdir(parents=True, exist_ok=True)
    check_model_disk_space(models_root)
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


def warmup_and_verify_full_gpu_offload(
    models_root: Path,
    ollama_log_path: Path,
    vm_log_fh,
    warmup_host: str = WARMUP_OLLAMA_HOST,
) -> dict[str, object]:
    """Start a short-lived ollama serve on another port, warm-load the model, verify 100% GPU offload, unload, and stop."""
    ensure_not_drive_path(models_root)
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = str(models_root)
    env["OLLAMA_HOST"] = warmup_host
    base_url = f"http://{warmup_host}"

    with ollama_log_path.open("a", encoding="utf-8") as ol_fh:
        srv = subprocess.Popen(
            ["/usr/local/bin/ollama", "serve"],
            env=env,
            stdout=ol_fh,
            stderr=subprocess.STDOUT,
        )
        try:
            ready = False
            for _ in range(30):
                try:
                    with urllib.request.urlopen(f"{base_url}/api/ps", timeout=2.0) as resp:
                        if resp.status == 200:
                            ready = True
                            break
                except Exception:
                    time.sleep(0.5)
            if not ready:
                raise RefusalError(
                    "MODEL_PARTIAL_OFFLOAD",
                    f"Warm-up Ollama server on {warmup_host} did not become ready",
                )
            result = check_ollama_server_full_gpu_offload(
                base_url=base_url,
                model=WRITER_MODEL,
            )
            vm_log_fh.write(
                f"gpu_offload_check ok model={result['model']} size={result['size']} size_vram={result['size_vram']}\n"
            )
            vm_log_fh.flush()
            return result
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=10)
            except subprocess.TimeoutExpired:
                srv.kill()
                srv.wait(timeout=5)


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


def run_dryrun_pipeline(
    args: argparse.Namespace,
    logs_dir: Path,
    vm_log_path: Path,
    gpu_info: dict[str, object] | None = None,
    prompt_line_hashes_out: set[str] | None = None,
) -> dict[str, object]:
    """Execute deterministic dry-run pipeline for rig fake-CLI testing (no GPU/network)."""
    if gpu_info is None:
        gpu_info = probe_vm_gpu(args.mode, allow_missing_in_dryrun=True)

    free_disk_bytes = check_model_disk_space(args.models_root)

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

    local_line_hashes: set[str] = (
        prompt_line_hashes_out if prompt_line_hashes_out is not None else set()
    )
    verified_gate = run_exact_count_gate(
        index_path=index_path,
        prompts_dir=args.prompts_dir,
        count_fn=_dryrun_counter,
        prompt_line_hashes_out=local_line_hashes,
        prompts_tar=getattr(args, "prompts_tar", None),
    )

    if args.mode == "gpu":
        fake_ps_json = os.environ.get("UDLAB_FAKE_API_PS_JSON", "").strip()
        if fake_ps_json:
            gpu_offload_check = verify_model_full_gpu_offload(
                fake_ps_json,
                model=WRITER_MODEL,
            )
        else:
            gpu_offload_check = verify_model_full_gpu_offload(
                {
                    "models": [
                        {
                            "name": WRITER_MODEL,
                            "model": WRITER_MODEL,
                            "size": 23456789012,
                            "size_vram": 23456789012,
                        }
                    ]
                },
                model=WRITER_MODEL,
            )
    else:
        gpu_offload_check = {"checked": False, "reason": "stub_mode"}

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
        **gpu_info,
        "free_disk_bytes": free_disk_bytes,
        "uv_version": f"uv {UV_VERSION}",
        "uv_archive_sha256": UV_ARCHIVE_SHA256,
        "python_version": f"Python {REQUIRED_PYTHON_VERSION}.0",
        "ollama_version": f"ollama version is {OLLAMA_VERSION}",
        "ollama_archive_sha256": OLLAMA_ARCHIVE_SHA256,
        "writer_model": WRITER_MODEL,
        "manifest_digest": EXPECTED_MANIFEST_DIGEST,
        "model_layer_blob": EXPECTED_MODEL_LAYER_BLOB,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "exact_count_gate": verified_gate,
        "prompt_line_sha256s": sorted(local_line_hashes),
        "gpu_offload_check": gpu_offload_check,
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
    parser.add_argument("--prompts-tar", type=Path, default=None)
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
    if args.prompts_tar is not None:
        ensure_not_drive_path(args.prompts_tar)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    vm_log_path = logs_dir / "vm.log"
    ollama_log_path = logs_dir / "ollama-serve.log"
    gen_log_path = logs_dir / "udlab-gen.log"
    receipt_path = args.run_dir / "vm_receipt.json"

    dryrun = args.dryrun or os.environ.get("VM_DRYRUN") == "1"
    gpu_info: dict[str, object] = {}
    free_disk_bytes: int | None = None
    prompt_line_hashes: set[str] = set()
    current_step = "probe_vm_gpu"

    try:
        gpu_info = probe_vm_gpu(args.mode, allow_missing_in_dryrun=dryrun)
        if dryrun:
            current_step = "dryrun_pipeline"
            receipt = run_dryrun_pipeline(
                args,
                logs_dir,
                vm_log_path,
                gpu_info=gpu_info,
                prompt_line_hashes_out=prompt_line_hashes,
            )
            free_disk_bytes = int(receipt["free_disk_bytes"])
        else:
            with vm_log_path.open("a", encoding="utf-8", newline="\n") as vm_log_fh:
                current_step = "install_pinned_ollama"
                archive_sha, ollama_ver = install_pinned_ollama(args.work_dir, vm_log_fh)
                current_step = "install_pinned_uv_and_sync_repo"
                uv_archive_sha, uv_ver, py_ver, uv_bin = install_pinned_uv_and_sync_repo(
                    work_dir=args.work_dir,
                    repo_dir=args.repo_dir,
                    log_fh=vm_log_fh,
                )
                current_step = "check_model_disk_space"
                free_disk_bytes = check_model_disk_space(args.models_root)
                current_step = "pull_and_verify_writer_model"
                manifest_digest, layer_blob, blob_path = pull_and_verify_writer_model(
                    models_root=args.models_root,
                    ollama_log_path=ollama_log_path,
                    vm_log_fh=vm_log_fh,
                )
                current_step = "build_llama_tokenize"
                llama_tokenize_bin = build_llama_tokenize(args.work_dir, vm_log_fh)

                current_step = "run_exact_count_gate"
                index_path = args.index_path or (
                    args.repo_dir / "tests" / "fixtures" / "exact_ctx" / "index.json"
                )
                verified_gate = run_exact_count_gate(
                    index_path=index_path,
                    prompts_dir=args.prompts_dir,
                    llama_tokenize_bin=llama_tokenize_bin,
                    gguf_path=blob_path,
                    repo_dir=args.repo_dir,
                    uv_bin=uv_bin,
                    prompt_line_hashes_out=prompt_line_hashes,
                    prompts_tar=args.prompts_tar,
                )

                if args.mode == "gpu":
                    current_step = "warmup_and_verify_full_gpu_offload"
                    gpu_offload_check = warmup_and_verify_full_gpu_offload(
                        models_root=args.models_root,
                        ollama_log_path=ollama_log_path,
                        vm_log_fh=vm_log_fh,
                    )
                else:
                    gpu_offload_check = {"checked": False, "reason": "stub_mode"}

            current_step = "run_udlab_gen"
            gen_cmd = [
                str(uv_bin),
                "run",
                "--frozen",
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
                **gpu_info,
                "free_disk_bytes": free_disk_bytes,
                "uv_version": uv_ver,
                "uv_archive_sha256": uv_archive_sha,
                "python_version": py_ver,
                "ollama_version": ollama_ver,
                "ollama_archive_sha256": archive_sha,
                "writer_model": WRITER_MODEL,
                "manifest_digest": manifest_digest,
                "model_layer_blob": layer_blob,
                "llama_cpp_commit": LLAMA_CPP_COMMIT,
                "exact_count_gate": verified_gate,
                "prompt_line_sha256s": sorted(prompt_line_hashes),
                "gpu_offload_check": gpu_offload_check,
                "dryrun": False,
            }

        current_step = "verify_logs_and_pack_tar"
        receipt_path.write_text(
            json.dumps(receipt, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        for log_file in logs_dir.glob("*.log"):
            scrub_log_file_in_place(log_file)
        verify_logs_clean(
            logs_dir.glob("*.log"),
            prompt_line_hashes=prompt_line_hashes,
        )
        build_allowlisted_result_tar(args.run_dir, args.result_tar)
        return 0

    except RefusalError as err:
        fail_receipt: dict[str, object] = {
            "status": "refused",
            "failure_class": err.failure_class,
            "detail": scrub_log_text(err.detail),
            "mode": args.mode,
            "writer": args.writer,
            "count": args.count,
            "seed": args.seed,
            **gpu_info,
        }
        if free_disk_bytes is not None:
            fail_receipt["free_disk_bytes"] = free_disk_bytes
        if prompt_line_hashes:
            fail_receipt["prompt_line_sha256s"] = sorted(prompt_line_hashes)
        receipt_path.write_text(
            json.dumps(fail_receipt, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with vm_log_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"REFUSAL {err.failure_class}: {scrub_log_text(err.detail)}\n")
        for log_file in logs_dir.glob("*.log"):
            scrub_log_file_in_place(log_file)
        try:
            build_allowlisted_result_tar(args.run_dir, args.result_tar)
        except Exception:
            pass
        sys.stderr.write(f"{err.failure_class}: {scrub_log_text(err.detail)}\n")
        return 2

    except Exception as exc:
        exc_type = type(exc).__name__
        detail = f"step={current_step} exception_type={exc_type}"
        unexpected_receipt: dict[str, object] = {
            "status": "refused",
            "failure_class": "UNEXPECTED_ERROR",
            "step": current_step,
            "exception_type": exc_type,
            "detail": detail,
            "mode": args.mode,
            "writer": args.writer,
            "count": args.count,
            "seed": args.seed,
            **gpu_info,
        }
        if free_disk_bytes is not None:
            unexpected_receipt["free_disk_bytes"] = free_disk_bytes
        if prompt_line_hashes:
            unexpected_receipt["prompt_line_sha256s"] = sorted(prompt_line_hashes)
        receipt_path.write_text(
            json.dumps(unexpected_receipt, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with vm_log_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"REFUSAL UNEXPECTED_ERROR: {detail}\n")
        for log_file in logs_dir.glob("*.log"):
            scrub_log_file_in_place(log_file)
        try:
            build_allowlisted_result_tar(args.run_dir, args.result_tar)
        except Exception:
            pass
        sys.stderr.write(f"UNEXPECTED_ERROR: {detail}\n")
        return 2

    finally:
        cleanup_prompts_dir(args.prompts_dir, prompts_tar=args.prompts_tar)


if __name__ == "__main__":
    raise SystemExit(main())
