from __future__ import annotations

import argparse
import base64
import csv
import ctypes
import http.client
import io
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ASSETS_DEMO_JSONL = REPO_ROOT / "assets" / "demo.jsonl"
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmark_results"
DEFAULT_SERVER_BASE_URL = "http://127.0.0.1:18083"
DEFAULT_CONCURRENCY_SWEEP = (1, 2, 4, 8, 16, 32)
DEFAULT_LANGUAGE = "en"
LANGUAGE_NAME_PREFIX = {
    "en": "🇺🇸",
    "zh": "🇨🇳",
}
PRESET_PROMPTS = {
    "english_news": (
        "🇺🇸 English News",
    ),
    "english_mix": (
        "🇺🇸 Welcome to OpenMOSS",
        "🇺🇸 English News",
        "🇺🇸 The Quiet Motion of the World",
    ),
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MOSS-TTS-Nano local server endpoints.")
    parser.add_argument("--server-base-url", default=DEFAULT_SERVER_BASE_URL, help="Base URL of app.py or app_onnx.py.")
    parser.add_argument(
        "--mode",
        choices=("streaming", "nonstreaming", "both"),
        default="both",
        help="Which server path to benchmark.",
    )
    parser.add_argument("--text", default=None, help="Override demo text.")
    parser.add_argument("--text-file", default=None, help="UTF-8 text file with custom prompt.")
    parser.add_argument("--demo-id", default=None, help="Demo row id from /assets/demo.jsonl; defaults to the first row.")
    parser.add_argument(
        "--language",
        choices=tuple(LANGUAGE_NAME_PREFIX.keys()),
        default=DEFAULT_LANGUAGE,
        help="Language subset to benchmark. Default: en.",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESET_PROMPTS.keys()),
        default=None,
        help="Benchmark preset. English-only presets are english_news and english_mix.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for benchmark outputs.")
    parser.add_argument(
        "--concurrency",
        default="1,2,4,8,16,32",
        help="Comma-separated concurrency sweep values.",
    )
    parser.add_argument("--requests-per-level", type=int, default=4, help="Requests per concurrency level.")
    parser.add_argument("--warmup-requests", type=int, default=1, help="Warmup requests per mode.")
    parser.add_argument("--timeout-seconds", type=float, default=300.0, help="Per-request timeout.")
    parser.add_argument("--first-chunk-budget-ms", type=float, default=200.0, help="CCU pass/fail target for streaming.")
    parser.add_argument("--error-rate-budget-pct", type=float, default=1.0, help="CCU pass/fail target for errors.")
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0, help="Warmup poll interval.")
    return parser.parse_args(argv)


def _load_demo_entries(language: str = DEFAULT_LANGUAGE) -> list[dict[str, str]]:
    if not DEFAULT_ASSETS_DEMO_JSONL.is_file():
        return []
    name_prefix = LANGUAGE_NAME_PREFIX.get(language, "")
    role_prefix = f"assets/audio/{language}_"
    entries: list[dict[str, str]] = []
    for raw_line in DEFAULT_ASSETS_DEMO_JSONL.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        role = str(payload.get("role", "")).strip()
        text = str(payload.get("text", "")).strip()
        name = str(payload.get("name", "")).strip()
        if role and text and role.startswith(role_prefix):
            if name_prefix and not name.startswith(name_prefix):
                continue
            entries.append({"role": role, "text": text, "name": name})
    return entries


def _build_demo_index(language: str = DEFAULT_LANGUAGE) -> dict[str, dict[str, str]]:
    entries = _load_demo_entries(language=language)
    index: dict[str, dict[str, str]] = {}
    for entry in entries:
        name = str(entry.get("name", "")).strip()
        role = str(entry.get("role", "")).strip()
        if name:
            index[name] = {"name": name, "role": role, "text": str(entry.get("text", ""))}
        if role:
            index[role] = {"name": name, "role": role, "text": str(entry.get("text", ""))}
    return index


def _get_demo_payload(
    text_override: str | None,
    text_file: str | None,
    demo_id: str | None,
    *,
    language: str,
) -> dict[str, str]:
    if text_override:
        text = text_override
    elif text_file:
        text = Path(text_file).read_text(encoding="utf-8")
    else:
        entries = _load_demo_entries(language=language)
        if not entries:
            raise FileNotFoundError(f"Demo corpus not found for language={language}: {DEFAULT_ASSETS_DEMO_JSONL}")
        if demo_id:
            selected = None
            try:
                selected_index = int(demo_id)
            except Exception:
                selected_index = -1
            if selected_index >= 1 and selected_index <= len(entries):
                selected = entries[selected_index - 1]
            if selected is None:
                for entry in entries:
                    if entry["name"] == demo_id:
                        selected = entry
                        break
            if selected is None:
                raise ValueError(f"Demo not found: {demo_id}")
            return {"demo_id": demo_id, "text": selected["text"], "name": selected.get("name", ""), "role": selected["role"]}
        selected = entries[0]
        return {"demo_id": "demo-1", "text": selected["text"], "name": selected.get("name", ""), "role": selected["role"]}
    entries = _load_demo_entries(language=language)
    if demo_id:
        for index, entry in enumerate(entries, start=1):
            if entry["name"] == demo_id or str(index) == str(demo_id):
                return {"demo_id": str(demo_id), "text": text, "name": entry.get("name", ""), "role": entry["role"]}
    selected = entries[0] if entries else {"role": "", "name": ""}
    return {"demo_id": str(demo_id or "demo-1"), "text": text, "name": selected.get("name", ""), "role": selected.get("role", "")}


def _resolve_benchmark_payloads(
    *,
    language: str,
    preset: str | None,
    text_override: str | None,
    text_file: str | None,
    demo_id: str | None,
) -> list[dict[str, str]]:
    if preset is not None:
        if text_override or text_file or demo_id:
            raise ValueError("--preset cannot be combined with --text, --text-file, or --demo-id.")
        demo_index = _build_demo_index(language=language)
        payloads: list[dict[str, str]] = []
        for preset_name in PRESET_PROMPTS[preset]:
            selected = demo_index.get(preset_name)
            if selected is None:
                raise FileNotFoundError(f"Preset entry not found: {preset_name}")
            payloads.append(
                {
                    "demo_id": selected["role"],
                    "text": selected["text"],
                    "name": selected["name"],
                    "role": selected["role"],
                }
            )
        return payloads
    return [
        _get_demo_payload(
            text_override,
            text_file,
            demo_id,
            language=language,
        )
    ]


def _machine_profile() -> dict[str, Any]:
    profile: dict[str, Any] = {}
    profile["os"] = {
        "platform": platform.platform(),
        "version": platform.version(),
        "release": platform.release(),
        "system": platform.system(),
    }
    profile["python"] = {
        "version": sys.version,
        "executable": sys.executable,
        "venv": os.environ.get("VIRTUAL_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "pip": _safe_run([sys.executable, "-m", "pip", "--version"]),
        "conda": _safe_run(["conda", "--version"]),
    }
    profile["cpu"] = _detect_cpu_profile()
    profile["memory"] = _detect_memory_profile()
    profile["disk"] = _detect_disk_profile()
    profile["gpu"] = _detect_gpu_profile()
    return profile


def _safe_run(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except Exception:
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    return output or None


def _detect_cpu_profile() -> dict[str, Any]:
    cpu: dict[str, Any] = {
        "model": platform.processor() or None,
        "cores": None,
        "threads": None,
    }
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            try:
                cpu["model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            except OSError:
                pass
            try:
                cpu["mhz"] = int(winreg.QueryValueEx(key, "~MHz")[0])
            except OSError:
                pass
            try:
                identifier = winreg.QueryValueEx(key, "Identifier")[0]
                cpu["identifier"] = identifier
            except OSError:
                pass
        except Exception:
            pass
    try:
        cpu["threads"] = os.cpu_count()
    except Exception:
        cpu["threads"] = None
    try:
        cpu["cores"] = max(1, (os.cpu_count() or 1) // 2)
    except Exception:
        cpu["cores"] = None
    return cpu


def _detect_memory_profile() -> dict[str, Any]:
    memory: dict[str, Any] = {"total_gb": None, "available_gb": None}
    if sys.platform == "win32":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                memory["total_gb"] = round(status.ullTotalPhys / 1024**3, 2)
                memory["available_gb"] = round(status.ullAvailPhys / 1024**3, 2)
        except Exception:
            pass
    return memory


def _detect_disk_profile() -> list[dict[str, Any]]:
    drives: list[dict[str, Any]] = []
    for root in ("C:\\", "D:\\", "E:\\"):
        try:
            usage = shutil.disk_usage(root)
        except Exception:
            continue
        drives.append(
            {
                "root": root,
                "total_gb": round(usage.total / 1024**3, 2),
                "free_gb": round(usage.free / 1024**3, 2),
            }
        )
    return drives


def _detect_gpu_profile() -> dict[str, Any]:
    profile: dict[str, Any] = {"present": False, "raw": None, "name": None, "driver_version": None, "cuda_version": None, "vram_gb": None}
    output = _safe_run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return profile
    profile["present"] = True
    profile["raw"] = output
    first_line = output.splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) >= 1:
        profile["name"] = parts[0]
    if len(parts) >= 2:
        profile["driver_version"] = parts[1]
    if len(parts) >= 3:
        try:
            profile["vram_gb"] = round(float(parts[2]) / 1024.0, 2)
        except Exception:
            profile["vram_gb"] = parts[2]
    smi = _safe_run(["nvidia-smi"])
    if smi:
        match = re.search(r"CUDA Version:\s*([0-9.]+)", smi)
        if match:
            profile["cuda_version"] = match.group(1)
        if not profile.get("driver_version"):
            match = re.search(r"Driver Version:\s*([0-9.]+)", smi)
            if match:
                profile["driver_version"] = match.group(1)
    return profile


def _normalize_concurrency(values: str) -> list[int]:
    concurrency: list[int] = []
    for raw_value in values.split(","):
        item = raw_value.strip()
        if not item:
            continue
        concurrency.append(max(1, int(item)))
    if not concurrency:
        raise ValueError("At least one concurrency value is required.")
    return concurrency


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def _open_json(url: str, data: dict[str, Any] | None = None, timeout: float = 300.0) -> dict[str, Any]:
    body = None
    headers = {}
    if data is not None:
        body = urllib.parse.urlencode(data).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _ensure_server_ready(base_url: str, timeout_seconds: float, poll_interval_seconds: float) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while time.time() < deadline:
        try:
            warmup = _open_json(f"{base_url}/api/warmup-status", timeout=min(10.0, timeout_seconds))
            if warmup.get("ready") is True:
                return warmup
            last_error = str(warmup.get("status_text") or warmup.get("message") or "warmup pending")
        except Exception as exc:
            last_error = str(exc)
        time.sleep(max(0.1, float(poll_interval_seconds)))
    raise TimeoutError(f"Server did not become ready within {timeout_seconds}s. Last error: {last_error}")


def _download_audio_duration_from_wav_bytes(payload: bytes) -> tuple[float, int, int]:
    with wave.open(io.BytesIO(payload), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        frames = wav_file.getnframes()
    duration = frames / float(sample_rate) if sample_rate > 0 else 0.0
    return duration, sample_rate, channels


def _pcm_duration_seconds(byte_count: int, sample_rate: int, channels: int) -> float:
    if sample_rate <= 0 or channels <= 0:
        return 0.0
    return byte_count / float(2 * sample_rate * channels)


@dataclass
class SampleRecord:
    mode: str
    concurrency: int
    request_index: int
    ok: bool
    latency_seconds: float | None
    first_chunk_seconds: float | None
    audio_duration_seconds: float | None
    rtf: float | None
    error: str | None
    extra: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        payload = {
            "mode": self.mode,
            "concurrency": self.concurrency,
            "request_index": self.request_index,
            "ok": self.ok,
            "latency_seconds": self.latency_seconds,
            "first_chunk_seconds": self.first_chunk_seconds,
            "audio_duration_seconds": self.audio_duration_seconds,
            "rtf": self.rtf,
            "error": self.error,
        }
        payload.update(self.extra)
        return payload


def _build_request_fields(demo_payload: dict[str, str]) -> dict[str, str]:
    return {
        "text": demo_payload["text"],
        "demo_id": demo_payload["demo_id"],
        "enable_text_normalization": "0",
        "enable_normalize_tts_text": "1",
        "do_sample": "1",
        "seed": "0",
    }


def _benchmark_nonstreaming_once(base_url: str, demo_payload: dict[str, str], timeout_seconds: float) -> SampleRecord:
    started = time.perf_counter()
    payload = _open_json(f"{base_url}/api/generate", data=_build_request_fields(demo_payload), timeout=timeout_seconds)
    finished = time.perf_counter()
    audio_base64 = str(payload.get("audio_base64") or "")
    if not audio_base64:
        raise RuntimeError("generate response did not contain audio_base64")
    audio_bytes = base64.b64decode(audio_base64)
    duration_seconds, sample_rate, channels = _download_audio_duration_from_wav_bytes(audio_bytes)
    latency_seconds = finished - started
    rtf = latency_seconds / duration_seconds if duration_seconds > 0 else None
    return SampleRecord(
        mode="nonstreaming",
        concurrency=1,
        request_index=0,
        ok=True,
        latency_seconds=latency_seconds,
        first_chunk_seconds=None,
        audio_duration_seconds=duration_seconds,
        rtf=rtf,
        error=None,
        extra={
            "sample_rate": sample_rate,
            "channels": channels,
            "run_status": payload.get("run_status"),
            "demo_name": demo_payload.get("name"),
            "demo_role": demo_payload.get("role"),
        },
    )


def _read_http_json_response(response: Any) -> dict[str, Any]:
    return json.loads(response.read().decode("utf-8"))


def _benchmark_streaming_once(base_url: str, demo_payload: dict[str, str], timeout_seconds: float) -> SampleRecord:
    started = time.perf_counter()
    start_payload = _open_json(f"{base_url}/api/generate-stream/start", data=_build_request_fields(demo_payload), timeout=timeout_seconds)
    stream_id = str(start_payload.get("stream_id") or "")
    audio_url = str(start_payload.get("audio_url") or "")
    if not stream_id or not audio_url:
        raise RuntimeError("generate-stream/start response missing stream_id or audio_url")

    parsed = urllib.parse.urlsplit(audio_url)
    connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_class(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), timeout=timeout_seconds)
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"

    first_chunk_at: float | None = None
    audio_bytes = bytearray()
    sample_rate = 0
    channels = 0
    try:
        connection.request("GET", path, headers={"Accept": "application/octet-stream"})
        response = connection.getresponse()
        try:
            sample_rate = int(response.getheader("X-Audio-Sample-Rate") or "0")
        except Exception:
            sample_rate = 0
        try:
            channels = int(response.getheader("X-Audio-Channels") or "0")
        except Exception:
            channels = 0
        first_byte = response.read(1)
        if first_byte:
            first_chunk_at = time.perf_counter()
            audio_bytes.extend(first_byte)
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            audio_bytes.extend(chunk)
    finally:
        try:
            connection.close()
        except Exception:
            pass

    finished = time.perf_counter()
    first_chunk_seconds = (first_chunk_at - started) if first_chunk_at is not None else None
    audio_duration_seconds = _pcm_duration_seconds(len(audio_bytes), sample_rate, channels)
    latency_seconds = finished - started
    rtf = latency_seconds / audio_duration_seconds if audio_duration_seconds > 0 else None
    return SampleRecord(
        mode="streaming",
        concurrency=1,
        request_index=0,
        ok=True,
        latency_seconds=latency_seconds,
        first_chunk_seconds=first_chunk_seconds,
        audio_duration_seconds=audio_duration_seconds,
        rtf=rtf,
        error=None,
        extra={
            "stream_id": stream_id,
            "sample_rate": sample_rate,
            "channels": channels,
            "run_status": start_payload.get("run_status"),
            "demo_name": demo_payload.get("name"),
            "demo_role": demo_payload.get("role"),
        },
    )


def _run_concurrent_bench(
    *,
    mode: str,
    concurrency: int,
    requests_per_level: int,
    base_url: str,
    demo_payload: dict[str, str],
    timeout_seconds: float,
) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    lock = threading.Lock()
    total_requests = max(1, int(concurrency) * max(1, int(requests_per_level)))

    def _task(request_index: int) -> SampleRecord:
        try:
            if mode == "streaming":
                record = _benchmark_streaming_once(base_url, demo_payload, timeout_seconds)
            else:
                record = _benchmark_nonstreaming_once(base_url, demo_payload, timeout_seconds)
            return SampleRecord(
                mode=record.mode,
                concurrency=concurrency,
                request_index=request_index,
                ok=True,
                latency_seconds=record.latency_seconds,
                first_chunk_seconds=record.first_chunk_seconds,
                audio_duration_seconds=record.audio_duration_seconds,
                rtf=record.rtf,
                error=None,
                extra=record.extra,
            )
        except Exception as exc:
            return SampleRecord(
                mode=mode,
                concurrency=concurrency,
                request_index=request_index,
                ok=False,
                latency_seconds=None,
                first_chunk_seconds=None,
                audio_duration_seconds=None,
                rtf=None,
                error=str(exc),
                extra={},
            )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_task, request_index) for request_index in range(total_requests)]
        for future in as_completed(futures):
            record = future.result()
            with lock:
                records.append(record)
    records.sort(key=lambda item: item.request_index)
    return records


def _summarize_records(records: list[SampleRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[SampleRecord]] = {}
    for record in records:
        grouped.setdefault((record.mode, record.concurrency), []).append(record)
    for (mode, concurrency), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        total = len(group)
        success = [item for item in group if item.ok]
        errors = total - len(success)
        latency_values = [item.latency_seconds for item in success if item.latency_seconds is not None]
        first_chunk_values = [item.first_chunk_seconds for item in success if item.first_chunk_seconds is not None]
        rtf_values = [item.rtf for item in success if item.rtf is not None]
        error_rate = (errors / total * 100.0) if total else 0.0
        row = {
            "mode": mode,
            "concurrency": concurrency,
            "requests": total,
            "successes": len(success),
            "errors": errors,
            "error_rate_pct": round(error_rate, 3),
            "latency_p50_s": _percentile(latency_values, 50.0),
            "latency_p95_s": _percentile(latency_values, 95.0),
            "latency_p99_s": _percentile(latency_values, 99.0),
            "first_chunk_p50_s": _percentile(first_chunk_values, 50.0),
            "first_chunk_p95_s": _percentile(first_chunk_values, 95.0),
            "first_chunk_p99_s": _percentile(first_chunk_values, 99.0),
            "rtf_mean": _mean(rtf_values),
            "rtf_p50": _percentile(rtf_values, 50.0),
            "rtf_p95": _percentile(rtf_values, 95.0),
            "rtf_p99": _percentile(rtf_values, 99.0),
        }
        rows.append(row)
    return rows


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _find_ccu_limit(rows: list[dict[str, Any]], first_chunk_budget_ms: float, error_rate_budget_pct: float) -> dict[str, Any] | None:
    streaming_rows = [row for row in rows if row["mode"] == "streaming"]
    passing: list[dict[str, Any]] = []
    for row in streaming_rows:
        p95 = _safe_float(row.get("first_chunk_p95_s"))
        error_rate = _safe_float(row.get("error_rate_pct"))
        if p95 is None or error_rate is None:
            continue
        if p95 * 1000.0 < first_chunk_budget_ms and error_rate < error_rate_budget_pct:
            passing.append(row)
    if not passing:
        return None
    return max(passing, key=lambda row: int(row["concurrency"]))


def _write_jsonl(path: Path, records: Iterable[SampleRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), ensure_ascii=False))
            handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "concurrency",
        "requests",
        "successes",
        "errors",
        "error_rate_pct",
        "latency_p50_s",
        "latency_p95_s",
        "latency_p99_s",
        "first_chunk_p50_s",
        "first_chunk_p95_s",
        "first_chunk_p99_s",
        "rtf_mean",
        "rtf_p50",
        "rtf_p95",
        "rtf_p99",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _format_cell(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(path: Path, profile: dict[str, Any], rows: list[dict[str, Any]], ccu_limit: dict[str, Any] | None) -> None:
    lines: list[str] = []
    lines.append("# MOSS-TTS-Nano Benchmark")
    lines.append("")
    lines.append("## Machine Profile")
    lines.append("")
    lines.append(f"- OS: `{profile['os']['platform']}`")
    lines.append(f"- Python: `{profile['python']['version'].splitlines()[0]}`")
    lines.append(f"- Python executable: `{profile['python']['executable']}`")
    lines.append(
        f"- Venv/Conda: venv=`{profile['python']['venv'] or ''}` conda=`{profile['python']['conda_default_env'] or ''}`"
    )
    cpu = profile.get("cpu", {})
    lines.append(f"- CPU: `{cpu.get('model') or 'unknown'}`")
    lines.append(f"- CPU threads: `{cpu.get('threads') or 'unknown'}`")
    lines.append(f"- RAM total: `{profile['memory'].get('total_gb') or 'unknown'} GB`")
    lines.append(f"- RAM available: `{profile['memory'].get('available_gb') or 'unknown'} GB`")
    gpu = profile.get("gpu", {})
    if gpu.get("present"):
        lines.append(
            f"- GPU: `{gpu.get('name') or 'unknown'}` VRAM `{gpu.get('vram_gb') or 'unknown'} GB` driver `{gpu.get('driver_version') or 'unknown'}` CUDA `{gpu.get('cuda_version') or 'unknown'}`"
        )
    else:
        lines.append("- GPU: not detected")
    lines.append("- Disk:")
    for item in profile["disk"]:
        lines.append(f"  - `{item['root']}` total `{item['total_gb']} GB` free `{item['free_gb']} GB`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| mode | concurrency | requests | errors | error_rate % | latency p50 s | latency p95 s | latency p99 s | first_chunk p50 s | first_chunk p95 s | first_chunk p99 s | RTF mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {mode} | {concurrency} | {requests} | {errors} | {error_rate_pct} | {latency_p50_s} | {latency_p95_s} | {latency_p99_s} | {first_chunk_p50_s} | {first_chunk_p95_s} | {first_chunk_p99_s} | {rtf_mean} |".format(
                mode=row["mode"],
                concurrency=row["concurrency"],
                requests=row["requests"],
                errors=row["errors"],
                error_rate_pct=_format_cell(row["error_rate_pct"]),
                latency_p50_s=_format_cell(row["latency_p50_s"]),
                latency_p95_s=_format_cell(row["latency_p95_s"]),
                latency_p99_s=_format_cell(row["latency_p99_s"]),
                first_chunk_p50_s=_format_cell(row["first_chunk_p50_s"]),
                first_chunk_p95_s=_format_cell(row["first_chunk_p95_s"]),
                first_chunk_p99_s=_format_cell(row["first_chunk_p99_s"]),
                rtf_mean=_format_cell(row["rtf_mean"]),
            )
        )
    lines.append("")
    lines.append("## CCU")
    lines.append("")
    if ccu_limit is None:
        lines.append("No concurrency level met `p95 first chunk < 200 ms` and `error rate < 1%`.")
    else:
        lines.append(
            f"Max passing CCU: `{ccu_limit['concurrency']}` at mode `{ccu_limit['mode']}` with `first_chunk_p95_s={_format_cell(ccu_limit['first_chunk_p95_s'])}` and `error_rate_pct={_format_cell(ccu_limit['error_rate_pct'])}`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    demo_payloads = _resolve_benchmark_payloads(
        language=args.language,
        preset=args.preset,
        text_override=args.text,
        text_file=args.text_file,
        demo_id=args.demo_id,
    )
    concurrency_values = _normalize_concurrency(args.concurrency)
    output_dir = Path(args.output_dir or (DEFAULT_RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = _machine_profile()

    base_url = args.server_base_url.rstrip("/")
    if args.preset:
        print(f"Using language={args.language} preset={args.preset} with {len(demo_payloads)} prompts.")
        for index, payload in enumerate(demo_payloads, start=1):
            print(f"  [{index}] {payload['name'] or payload['demo_id']} -> {payload['text'][:80]!r}")
    else:
        demo_payload = demo_payloads[0]
        print(
            f"Using language={args.language} demo: {demo_payload['name'] or demo_payload['demo_id']} -> {demo_payload['text'][:80]!r}"
        )
    print(f"Waiting for server readiness at {base_url}...")
    warmup = _ensure_server_ready(base_url, timeout_seconds=args.timeout_seconds, poll_interval_seconds=args.poll_interval_seconds)
    print(f"Server ready: {warmup.get('status_text') or warmup.get('message') or 'ready'}")

    all_records: list[SampleRecord] = []
    if args.mode in {"nonstreaming", "both"}:
        for _ in range(max(0, int(args.warmup_requests))):
            try:
                _benchmark_nonstreaming_once(base_url, demo_payloads[0], args.timeout_seconds)
            except Exception:
                pass
        for demo_payload in demo_payloads:
            for concurrency in concurrency_values:
                print(
                    f"Benchmarking nonstreaming at concurrency={concurrency} prompt={demo_payload['name'] or demo_payload['demo_id']}..."
                )
                all_records.extend(
                    _run_concurrent_bench(
                        mode="nonstreaming",
                        concurrency=concurrency,
                        requests_per_level=args.requests_per_level,
                        base_url=base_url,
                        demo_payload=demo_payload,
                        timeout_seconds=args.timeout_seconds,
                    )
                )

    if args.mode in {"streaming", "both"}:
        for _ in range(max(0, int(args.warmup_requests))):
            try:
                _benchmark_streaming_once(base_url, demo_payloads[0], args.timeout_seconds)
            except Exception:
                pass
        for demo_payload in demo_payloads:
            for concurrency in concurrency_values:
                print(
                    f"Benchmarking streaming at concurrency={concurrency} prompt={demo_payload['name'] or demo_payload['demo_id']}..."
                )
                all_records.extend(
                    _run_concurrent_bench(
                        mode="streaming",
                        concurrency=concurrency,
                        requests_per_level=args.requests_per_level,
                        base_url=base_url,
                        demo_payload=demo_payload,
                        timeout_seconds=args.timeout_seconds,
                    )
                )

    summary_rows = _summarize_records(all_records)
    ccu_limit = _find_ccu_limit(summary_rows, args.first_chunk_budget_ms, args.error_rate_budget_pct)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"benchmark_{timestamp}.jsonl"
    csv_path = output_dir / f"benchmark_{timestamp}.csv"
    md_path = output_dir / f"benchmark_{timestamp}.md"
    profile_path = output_dir / f"machine_profile_{timestamp}.json"

    _write_jsonl(jsonl_path, all_records)
    _write_csv(csv_path, summary_rows)
    _write_markdown(md_path, profile, summary_rows, ccu_limit)
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {profile_path}")
    if ccu_limit is not None:
        print(f"Max passing CCU: {ccu_limit['concurrency']} at mode={ccu_limit['mode']}")
    else:
        print("No CCU level met the target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
