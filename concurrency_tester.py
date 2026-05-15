import asyncio
import time
import json
import uuid
import os
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any

from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import httpx
import psutil

# GPU detection (optional — only works with NVIDIA GPUs)
_pynvml_available = False
try:
    import pynvml
    pynvml.nvmlInit()
    _pynvml_available = True
except Exception:
    pass

# ── App Setup ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
CONFIG_FILE = BASE_DIR / "providers_config.json"

app = FastAPI(title="LLM Concurrency Tester")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

active_providers: Dict[str, dict] = {}
active_tests: Dict[str, dict] = {}

# ── Pydantic Models ──────────────────────────────────────────────────────────

class ProviderConfig(BaseModel):
    name: str
    api_base: str
    api_key: str = ""
    model: str
    max_tokens: int = Field(default=512, ge=1, le=16384)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    extra_headers: Dict[str, str] = Field(default_factory=dict)
    extra_body: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("api_base")
    @classmethod
    def check_api_base(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("api_base must start with http:// or https://")
        return v.rstrip("/")


class TestRequest(BaseModel):
    provider_id: str
    prompt: str = Field(..., min_length=1)
    concurrency: int = Field(default=10, ge=1, le=200)


class RequestResult(BaseModel):
    request_id: int
    status: str  # "success" or "error"
    total_latency_ms: float
    ttft_ms: Optional[float] = None
    token_count: int = 0
    error_message: Optional[str] = None
    model: str = ""
    start_time_iso: str = ""


class AggregateResults(BaseModel):
    test_id: str
    test_duration_ms: float
    prompt: str
    concurrency: int
    total_requests: int
    success_count: int
    error_count: int
    success_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    avg_ttft_ms: Optional[float] = None
    min_ttft_ms: Optional[float] = None
    max_ttft_ms: Optional[float] = None
    model: str = ""
    provider_name: str = ""


# ── Provider Persistence ─────────────────────────────────────────────────────

def load_providers():
    global active_providers
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            active_providers = data.get("providers", {})
        except (json.JSONDecodeError, KeyError):
            active_providers = {}
    else:
        active_providers = {}


def save_providers():
    CONFIG_FILE.write_text(
        json.dumps({"providers": active_providers}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Hardware Detection ────────────────────────────────────────────────────────

def get_hardware_info() -> dict:
    """Auto-detect CPU, RAM, and GPU hardware specs."""
    info = {}

    # CPU
    cpu_info = {}
    cpu_info["name"] = platform.processor() or "Unknown"
    cpu_info["physical_cores"] = psutil.cpu_count(logical=False)
    cpu_info["logical_cores"] = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        cpu_info["base_clock_mhz"] = round(cpu_freq.max, 0) if cpu_freq.max else None
        cpu_info["current_clock_mhz"] = round(cpu_freq.current, 0)
    else:
        cpu_info["base_clock_mhz"] = None
        cpu_info["current_clock_mhz"] = None
    cpu_info["architecture"] = platform.machine()
    info["cpu"] = cpu_info

    # RAM
    mem = psutil.virtual_memory()
    ram_info = {
        "total_gb": round(mem.total / (1024**3), 1),
        "available_gb": round(mem.available / (1024**3), 1),
        "used_percent": mem.percent,
    }
    # Get RAM speed and stick count
    ram_sticks = []
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,Speed | ConvertTo-Json"],
                capture_output=True, text=True, timeout=10
            )
            try:
                data = json.loads(result.stdout.strip())
                if isinstance(data, dict):
                    data = [data]
                for stick in data:
                    cap = stick.get("Capacity", 0)
                    speed = stick.get("Speed", 0) or 0
                    ram_sticks.append({
                        "capacity_gb": round(cap / (1024**3), 1),
                        "speed_mhz": speed,
                    })
            except (json.JSONDecodeError, KeyError):
                pass
        elif sys.platform == "linux":
            result = subprocess.run(
                ["dmidecode", "-t", "memory"],
                capture_output=True, text=True, timeout=10
            )
            current = {}
            for line in result.stdout.split("\n"):
                line = line.strip()
                if "Size:" in line and "MB" not in line and "No Module" not in line:
                    if current:
                        ram_sticks.append(current)
                    current = {}
                elif "Size:" in line and "MB" in line:
                    try:
                        current["capacity_gb"] = round(int(line.split(":")[1].strip().replace(" MB", "")) / 1024, 1)
                    except ValueError:
                        pass
                elif "Speed:" in line and "MHz" in line:
                    try:
                        current["speed_mhz"] = int(line.split(":")[1].strip().replace(" MHz", "").split()[0])
                    except ValueError:
                        pass
    except Exception:
        pass
    ram_info["sticks"] = ram_sticks
    info["ram"] = ram_info

    # GPU (NVIDIA via pynvml)
    gpu_info = {"available": False, "cards": []}
    if _pynvml_available:
        try:
            gpu_count = pynvml.nvmlDeviceGetCount()
            gpu_info["available"] = True
            for i in range(gpu_count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h) if hasattr(pynvml, "nvmlDeviceGetName") else ""
                # memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(h)
                vram_total_gb = round(mem_info.total / (1024**3), 1)
                vram_free_gb = round(mem_info.free / (1024**3), 1)
                # clocks
                try:
                    mem_clock = pynvml.nvmlDeviceGetMaxClockInfo(h, 2)  # 2 = memory
                except Exception:
                    mem_clock = 0
                try:
                    sm_clock = pynvml.nvmlDeviceGetMaxClockInfo(h, 0)  # 0 = SM
                except Exception:
                    sm_clock = 0
                # bus width
                try:
                    bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(h)
                except Exception:
                    bus_width = 0
                # CUDA cores
                try:
                    cuda_cores = pynvml.nvmlDeviceGetNumGpuCores(h)
                except Exception:
                    cuda_cores = 0
                # compute capability
                try:
                    cc = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                    cc_str = f"{cc[0]}.{cc[1]}"
                except Exception:
                    cc_str = "N/A"
                # PCIe
                try:
                    pcie_gen = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(h)
                    pcie_width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(h)
                except Exception:
                    pcie_gen = 0
                    pcie_width = 0
                # memory bandwidth estimate: bus_width_bytes * mem_clock_effective
                # GDDR6 effective clock = 2× command clock. pynvml reports the command clock in MHz.
                # For GDDR6, the data rate is 2× the command clock (DDR), so effective = command_clock × 2
                # Common: mem_clock of 7000 → effective 14000 MHz → 14 Gbps
                # Bandwidth (GB/s) = bus_width(bits) / 8 * effective_clock(GHz)
                # Bandwidth (GB/s) = bus_width / 8 * (mem_clock * 2 / 1000)
                if bus_width > 0 and mem_clock > 0:
                    bandwidth_gb_s = round(bus_width / 8 * (mem_clock * 2 / 1000), 1)
                elif "2080 Ti" in name:
                    bandwidth_gb_s = 616.0  # known spec: 352-bit × 14 Gbps
                else:
                    bandwidth_gb_s = None

                gpu_info["cards"].append({
                    "index": i,
                    "name": name or "NVIDIA GPU",
                    "vram_total_gb": vram_total_gb,
                    "vram_free_gb": vram_free_gb,
                    "memory_bus_width": bus_width,
                    "memory_clock_mhz": mem_clock,
                    "memory_bandwidth_gb_s": bandwidth_gb_s,
                    "max_graphics_clock_mhz": sm_clock,
                    "cuda_cores": cuda_cores,
                    "compute_capability": cc_str,
                    "pcie_generation": pcie_gen,
                    "pcie_link_width": pcie_width,
                })
        except Exception:
            gpu_info["available"] = False

    info["gpu"] = gpu_info
    return info


# ── Model Source Detection ────────────────────────────────────────────────────

# Quantization bits per parameter (approximate)
QUANT_BYTES = {
    "F32": 4.0, "FP32": 4.0,
    "F16": 2.0, "FP16": 2.0,
    "Q8_0": 1.0, "Q8_1": 1.0,
    "Q6_K": 0.75, "Q6_K_M": 0.75,
    "Q5_0": 0.625, "Q5_1": 0.625, "Q5_K_S": 0.625, "Q5_K_M": 0.625,
    "Q4_0": 0.5, "Q4_1": 0.5, "Q4_K_S": 0.5, "Q4_K_M": 0.55,
    "Q3_K_S": 0.4, "Q3_K_M": 0.425, "Q3_K_L": 0.45,
    "Q2_K": 0.325, "Q2_K_S": 0.325,
    "IQ4_NL": 0.5, "IQ4_XS": 0.5,
    "IQ3_XXS": 0.375, "IQ3_S": 0.4, "IQ3_M": 0.425,
    "IQ2_XXS": 0.3, "IQ2_XS": 0.325, "IQ2_S": 0.35, "IQ2_M": 0.375,
    "IQ1_S": 0.225, "IQ1_M": 0.25,
}


def parse_param_count(param_str: str) -> Optional[float]:
    """Parse parameter count string like '8.0B', '7B', '70B' to billions."""
    if not param_str:
        return None
    s = param_str.strip().upper().replace(" ", "")
    try:
        if s.endswith("B"):
            return float(s[:-1])
        elif s.endswith("M"):
            return float(s[:-1]) / 1000
    except ValueError:
        pass
    return None


def guess_quant_from_name(name: str) -> Optional[str]:
    """Try to extract quantization level from model name."""
    name_upper = name.upper()
    # Common patterns in GGUF filenames
    quant_patterns = [
        "Q8_0", "Q8_1", "Q6_K_M", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1",
        "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1", "Q3_K_M", "Q3_K_S", "Q3_K_L",
        "Q2_K", "Q2_K_S", "IQ4_NL", "IQ4_XS", "IQ3_XXS", "IQ3_S", "IQ3_M",
        "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ1_S", "IQ1_M",
        "F16", "F32", "FP16", "FP32",
    ]
    for q in quant_patterns:
        if q in name_upper:
            return q
    return None


async def detect_ollama_models() -> list:
    """Detect models loaded in Ollama."""
    models = []
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
            if resp.status_code != 200:
                return models
            data = resp.json()
            for m in data.get("models", []):
                details = m.get("details", {})
                param_str = details.get("parameter_size", "")
                quant = details.get("quantization_level", "") or guess_quant_from_name(m.get("name", ""))
                param_b = parse_param_count(param_str)
                bytes_per_param = QUANT_BYTES.get(quant, 2.0)
                model_size_gb = round(param_b * bytes_per_param, 2) if param_b else None
                models.append({
                    "name": m.get("name", ""),
                    "source": "ollama",
                    "parameter_size": param_str,
                    "quantization": quant or "unknown",
                    "estimated_vram_gb": model_size_gb,
                    "size_bytes": m.get("size", 0),
                    "modified": m.get("modified_at", ""),
                })
    except Exception:
        pass
    return models


async def detect_lmstudio_models() -> list:
    """Detect models loaded in LM Studio."""
    models = []
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://127.0.0.1:1234/v1/models", timeout=5.0)
            if resp.status_code != 200:
                return models
            data = resp.json()
            for m in data.get("data", []):
                model_id = m.get("id", "")
                quant = guess_quant_from_name(model_id)
                # Try to guess parameter size from model name patterns like "7b", "13b", etc.
                param_b = None
                import re
                param_match = re.search(r'(\d+\.?\d*)\s*[bB]', model_id)
                if param_match:
                    param_b = float(param_match.group(1))
                bytes_per_param = QUANT_BYTES.get(quant, 2.0) if quant else 2.0
                model_size_gb = round(param_b * bytes_per_param, 2) if param_b else None
                models.append({
                    "name": model_id,
                    "source": "lmstudio",
                    "parameter_size": f"{param_b}B" if param_b else "unknown",
                    "quantization": quant or "unknown",
                    "estimated_vram_gb": model_size_gb,
                })
    except Exception:
        pass
    return models


async def detect_model_sources() -> dict:
    """Detect running model serving platforms and their loaded models."""
    ollama_available = False
    lmstudio_available = False

    # Quick connectivity check
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://127.0.0.1:11434/api/tags", timeout=3.0)
            ollama_available = r.status_code == 200
    except Exception:
        pass

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://127.0.0.1:1234/v1/models", timeout=3.0)
            lmstudio_available = r.status_code == 200
    except Exception:
        pass

    ollama_models = await detect_ollama_models() if ollama_available else []
    lmstudio_models = await detect_lmstudio_models() if lmstudio_available else []

    return {
        "ollama": {
            "available": ollama_available,
            "endpoint": "http://127.0.0.1:11434",
            "models": ollama_models,
        },
        "lmstudio": {
            "available": lmstudio_available,
            "endpoint": "http://127.0.0.1:1234",
            "models": lmstudio_models,
        },
    }


# ── Auto-Optimize QPS ─────────────────────────────────────────────────────────

class AutoOptimizeRequest(BaseModel):
    provider_id: str
    prompt: str = Field(..., min_length=1)
    max_ttft_ms: float = Field(default=1500, ge=100, le=30000)
    max_tpot_ms: float = Field(default=50, ge=10, le=5000)
    stress_mode: bool = False  # True = use full max_tokens for realistic simulation


async def run_auto_optimize(
    test_id: str,
    provider: dict,
    prompt: str,
    max_ttft_ms: float,
    max_tpot_ms: float,
    stress_mode: bool,
    result_queue: asyncio.Queue,
):
    """Iteratively increase concurrency to find optimal QPS under TTFT/TPOT constraints."""
    concurrency = 1
    best_concurrency = 0
    best_qps = 0.0
    best_results = None
    max_concurrency = 200
    opt_provider = dict(provider)
    if stress_mode:
        # Use full provider max_tokens for realistic stress testing
        pass  # keep provider's original max_tokens
    else:
        # Quick scan: cap at 32 tokens for fast iteration
        opt_provider["max_tokens"] = min(opt_provider.get("max_tokens", 512), 32)

    await result_queue.put({
        "type": "auto_start",
        "max_ttft_ms": max_ttft_ms,
        "max_tpot_ms": max_tpot_ms,
        "stress_mode": stress_mode,
        "prompt": prompt,
        "provider_name": provider.get("name", ""),
        "model": provider.get("model", ""),
    })

    while concurrency <= max_concurrency:
        await result_queue.put({
            "type": "iteration_start",
            "concurrency": concurrency,
            "iteration": concurrency,
        })

        # Run test at current concurrency
        semaphore = asyncio.Semaphore(concurrency)
        test_queue = asyncio.Queue()
        test_start = time.time()

        async with httpx.AsyncClient() as client:
            tasks = [
                execute_single_request(client, i, opt_provider, prompt, semaphore, test_queue)
                for i in range(concurrency)
            ]
            await asyncio.gather(*tasks)
        await test_queue.put(None)

        all_results = []
        while True:
            r = await test_queue.get()
            if r is None:
                break
            all_results.append(r)

        agg = compute_aggregate(test_id, test_start, opt_provider, prompt, concurrency, all_results)

        # Calculate TPOT for each successful request
        success = [r for r in all_results if r.status == "success" and r.ttft_ms is not None and r.token_count > 1]
        tpot_values = []
        for r in success:
            decode_time_ms = r.total_latency_ms - (r.ttft_ms or 0)
            decode_tokens = max(r.token_count - 1, 1)
            tpot_values.append(decode_time_ms / decode_tokens)
        avg_tpot = sum(tpot_values) / len(tpot_values) if tpot_values else None

        # QPS = concurrency / average_total_latency (in seconds)
        success_latencies = [r.total_latency_ms for r in all_results if r.status == "success"]
        avg_latency_s = (sum(success_latencies) / len(success_latencies) / 1000) if success_latencies else 999
        qps = concurrency / avg_latency_s if avg_latency_s > 0 else 0

        await result_queue.put({
            "type": "iteration_complete",
            "concurrency": concurrency,
            "avg_ttft_ms": agg["avg_ttft_ms"],
            "avg_tpot_ms": round(avg_tpot, 1) if avg_tpot else None,
            "qps": round(qps, 2),
            "success_rate": agg["success_rate"],
            "total_requests": agg["total_requests"],
            "success_count": agg["success_count"],
            "error_count": agg["error_count"],
            "avg_latency_ms": agg["avg_latency_ms"],
        })

        # Check if we exceeded thresholds
        ttft_exceeded = agg["avg_ttft_ms"] is not None and agg["avg_ttft_ms"] > max_ttft_ms
        tpot_exceeded = avg_tpot is not None and avg_tpot > max_tpot_ms

        if ttft_exceeded or tpot_exceeded:
            reason = []
            if ttft_exceeded:
                reason.append(f"TTFT {agg['avg_ttft_ms']:.0f}ms > {max_ttft_ms:.0f}ms")
            if tpot_exceeded:
                reason.append(f"TPOT {avg_tpot:.0f}ms > {max_tpot_ms:.0f}ms")
            await result_queue.put({
                "type": "threshold_exceeded",
                "reason": "; ".join(reason),
                "concurrency": concurrency,
            })
            break

        # Record this as the best so far
        best_concurrency = concurrency
        best_qps = qps
        best_results = {"aggregate": agg, "all_results": [r.dict() for r in all_results]}

        # Stop if error rate too high
        if agg["success_rate"] < 50:
            await result_queue.put({
                "type": "high_error_rate",
                "success_rate": agg["success_rate"],
                "concurrency": concurrency,
            })
            break

        # Increase concurrency: double each step, but cap at max
        if concurrency >= max_concurrency:
            break
        concurrency = min(concurrency * 2, max_concurrency)

    await result_queue.put({
        "type": "auto_complete",
        "best_concurrency": best_concurrency,
        "best_qps": round(best_qps, 2),
        "best_results": best_results,
    })
    await result_queue.put(None)


# ── SSE Stream Parser ────────────────────────────────────────────────────────

async def parse_openai_stream(response: httpx.Response, start_time: float):
    first_token_time: Optional[float] = None
    token_count = 0
    error_msg: Optional[str] = None

    try:
        first_line_flag = True
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                data_str = line[5:].lstrip()
            else:
                # Non-streaming fallback: try to parse the whole response JSON
                if first_line_flag and line.strip():
                    try:
                        chunk = json.loads(line)
                        # Check for error response first
                        if "error" in chunk and "choices" not in chunk:
                            error_msg = str(chunk.get("error", ""))[:300]
                            print(f"  DEBUG server error: {error_msg}")
                            break
                        choices = chunk.get("choices", [])
                        if choices:
                            msg = choices[0].get("message", {}) or choices[0].get("delta", {})
                            content = msg.get("content", "") or msg.get("reasoning", "")
                            if content:
                                first_token_time = time.time()
                                token_count = 1
                        end_time = time.time()
                        return first_token_time, token_count, end_time, error_msg
                    except json.JSONDecodeError:
                        pass
                continue

            first_line_flag = False

            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
                # Check for error in streaming chunk
                if "error" in chunk and "choices" not in chunk:
                    error_msg = str(chunk.get("error", ""))[:300]
                    break
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    text = delta.get("content", "") or delta.get("reasoning", "")
                    if text and first_token_time is None:
                        first_token_time = time.time()
                    if text:
                        token_count += 1
            except json.JSONDecodeError:
                continue
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"

    end_time = time.time()
    return first_token_time, token_count, end_time, error_msg


# ── Single Request Executor ──────────────────────────────────────────────────

async def execute_single_request(
    client: httpx.AsyncClient,
    request_id: int,
    provider: dict,
    prompt: str,
    semaphore: asyncio.Semaphore,
    result_queue: asyncio.Queue,
):
    async with semaphore:
        start_time = time.time()
        start_time_iso = time.strftime("%H:%M:%S", time.localtime(start_time))

        url = f"{provider['api_base']}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if provider.get("api_key"):
            headers["Authorization"] = f"Bearer {provider['api_key']}"
        headers.update(provider.get("extra_headers", {}))
        body = {
            "model": provider["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": provider.get("max_tokens", 512),
            "temperature": provider.get("temperature", 0.0),
            "stream": True,
            **provider.get("extra_body", {}),
        }

        try:
            async with client.stream(
                "POST", url, json=body, headers=headers, timeout=httpx.Timeout(120.0)
            ) as response:
                if response.status_code != 200:
                    error_text = ""
                    try:
                        error_text = await response.aread()
                        error_text = error_text.decode("utf-8", errors="replace")[:200]
                    except Exception:
                        pass
                    result = RequestResult(
                        request_id=request_id,
                        status="error",
                        total_latency_ms=(time.time() - start_time) * 1000,
                        error_message=f"HTTP {response.status_code}: {error_text}",
                        model=provider["model"],
                        start_time_iso=start_time_iso,
                    )
                else:
                    first_token_time, token_count, end_time, stream_error = await parse_openai_stream(
                        response, start_time
                    )
                    if stream_error:
                        result = RequestResult(
                            request_id=request_id,
                            status="error",
                            total_latency_ms=(end_time - start_time) * 1000,
                            error_message=stream_error,
                            token_count=token_count,
                            model=provider["model"],
                            start_time_iso=start_time_iso,
                        )
                    else:
                        ttft_ms = (
                            (first_token_time - start_time) * 1000
                            if first_token_time is not None
                            else None
                        )
                        result = RequestResult(
                            request_id=request_id,
                            status="success",
                            total_latency_ms=(end_time - start_time) * 1000,
                            ttft_ms=ttft_ms,
                            token_count=token_count,
                            model=provider["model"],
                            start_time_iso=start_time_iso,
                        )
        except httpx.TimeoutException:
            result = RequestResult(
                request_id=request_id,
                status="error",
                total_latency_ms=(time.time() - start_time) * 1000,
                error_message="Request timeout (120s)",
                model=provider["model"],
                start_time_iso=start_time_iso,
            )
        except Exception as e:
            result = RequestResult(
                request_id=request_id,
                status="error",
                total_latency_ms=(time.time() - start_time) * 1000,
                error_message=str(e)[:300],
                model=provider["model"],
                start_time_iso=start_time_iso,
            )

        await result_queue.put(result)


# ── Test Orchestrator ─────────────────────────────────────────────────────────

def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = (len(sorted_values) - 1) * p / 100.0
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return sorted_values[lower]
    frac = idx - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def compute_aggregate(
    test_id: str,
    test_start: float,
    provider: dict,
    prompt: str,
    concurrency: int,
    all_results: List[RequestResult],
) -> dict:
    test_end = time.time()
    test_duration_ms = (test_end - test_start) * 1000

    success_results = [r for r in all_results if r.status == "success"]
    error_count = len(all_results) - len(success_results)
    total = len(all_results)

    success_rate = (len(success_results) / total * 100) if total > 0 else 0.0

    latencies = sorted([r.total_latency_ms for r in success_results])
    ttfts = sorted([r.ttft_ms for r in success_results if r.ttft_ms is not None])

    return {
        "test_id": test_id,
        "test_duration_ms": round(test_duration_ms, 1),
        "prompt": prompt,
        "concurrency": concurrency,
        "total_requests": total,
        "success_count": len(success_results),
        "error_count": error_count,
        "success_rate": round(success_rate, 1),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "p50_latency_ms": round(percentile(latencies, 50), 1),
        "p90_latency_ms": round(percentile(latencies, 90), 1),
        "p95_latency_ms": round(percentile(latencies, 95), 1),
        "p99_latency_ms": round(percentile(latencies, 99), 1),
        "min_latency_ms": round(min(latencies), 1) if latencies else 0,
        "max_latency_ms": round(max(latencies), 1) if latencies else 0,
        "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
        "min_ttft_ms": round(min(ttfts), 1) if ttfts else None,
        "max_ttft_ms": round(max(ttfts), 1) if ttfts else None,
        "model": provider.get("model", ""),
        "provider_name": provider.get("name", ""),
    }


async def run_concurrency_test(
    test_id: str,
    provider: dict,
    prompt: str,
    concurrency: int,
    result_queue: asyncio.Queue,
):
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [
            execute_single_request(client, i, provider, prompt, semaphore, result_queue)
            for i in range(concurrency)
        ]
        await asyncio.gather(*tasks)
    await result_queue.put(None)  # sentinel


# ── SSE Event Generator ──────────────────────────────────────────────────────

async def event_stream(test_id: str):
    test_state = active_tests.get(test_id)
    if not test_state:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Test not found'})}\n\n"
        return

    result_queue: asyncio.Queue = asyncio.Queue()
    test_start = time.time()

    asyncio.create_task(
        run_concurrency_test(
            test_id,
            test_state["provider"],
            test_state["prompt"],
            test_state["concurrency"],
            result_queue,
        )
    )

    yield f"data: {json.dumps({'type': 'test_start', 'test_id': test_id, 'total': test_state['concurrency']})}\n\n"

    all_results: List[RequestResult] = []
    while True:
        try:
            result = await asyncio.wait_for(result_queue.get(), timeout=130.0)
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Test timed out'})}\n\n"
            return

        if result is None:
            break
        all_results.append(result)
        yield f"data: {json.dumps({'type': 'request_complete', 'completed': len(all_results), 'total': test_state['concurrency'], 'result': result.dict()})}\n\n"

    agg = compute_aggregate(
        test_id, test_start,
        test_state["provider"], test_state["prompt"],
        test_state["concurrency"], all_results,
    )
    test_state["results"] = agg
    test_state["all_results"] = [r.dict() for r in all_results]
    yield f"data: {json.dumps({'type': 'test_complete', 'results': agg})}\n\n"


# ── API Routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    masked = []
    for pid, p in active_providers.items():
        pd = dict(p)
        key = pd.get("api_key", "")
        if len(key) > 8:
            pd["api_key"] = key[:4] + "*" * (len(key) - 8) + key[-4:]
        pd["id"] = pid
        masked.append(pd)
    return templates.TemplateResponse(
        "index.html", {"request": request, "providers_json": json.dumps(masked)}
    )


@app.get("/api/hardware")
async def get_hardware():
    """Return auto-detected hardware information for the local machine."""
    return get_hardware_info()


@app.get("/api/model-sources")
async def get_model_sources():
    """Detect running model serving platforms (Ollama, LM Studio) and their loaded models."""
    return await detect_model_sources()


@app.get("/api/providers")
async def list_providers():
    masked = []
    for pid, p in active_providers.items():
        pd = dict(p)
        key = pd.get("api_key", "")
        if len(key) > 8:
            pd["api_key"] = key[:4] + "*" * (len(key) - 8) + key[-4:]
        pd["id"] = pid
        masked.append(pd)
    return masked


@app.post("/api/providers")
async def create_provider(config: ProviderConfig):
    pid = str(uuid.uuid4())
    active_providers[pid] = config.dict()
    save_providers()
    return {"id": pid, **config.dict()}


@app.put("/api/providers/{provider_id}")
async def update_provider(provider_id: str, config: ProviderConfig):
    if provider_id not in active_providers:
        raise HTTPException(status_code=404, detail="Provider not found")
    updated = config.dict()
    # Preserve existing API key if the user didn't provide one
    if not updated.get("api_key") or "***" in updated.get("api_key", ""):
        updated["api_key"] = active_providers[provider_id].get("api_key", "")
    active_providers[provider_id] = updated
    save_providers()
    return {"id": provider_id, **updated}


async def _do_test_connection(api_base: str, api_key: str, model: str,
                            extra_headers: dict, extra_body: dict,
                            max_tokens: int = 10, temperature: float = 0.0):
    """Core connection test logic shared by both endpoints."""
    url = f"{api_base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(extra_headers)

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        **extra_body,
    }

    start = time.time()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=body, headers=headers, timeout=httpx.Timeout(30.0))
            elapsed = (time.time() - start) * 1000
            if resp.status_code == 200:
                data = resp.json()
                if "error" in data and "choices" not in data:
                    return {
                        "success": False,
                        "latency_ms": round(elapsed, 1),
                        "error": str(data.get("error", "Unknown"))[:300],
                    }
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {
                    "success": True,
                    "latency_ms": round(elapsed, 1),
                    "model": data.get("model", model),
                    "preview": content[:100],
                }
            else:
                error_detail = ""
                try:
                    error_detail = resp.text[:300]
                except Exception:
                    pass
                return {
                    "success": False,
                    "latency_ms": round(elapsed, 1),
                    "error": f"HTTP {resp.status_code}",
                    "detail": error_detail,
                }
    except httpx.TimeoutException:
        return {"success": False, "error": "Connection timed out (30s)"}
    except Exception as e:
        return {"success": False, "error": str(e)[:300]}


@app.post("/api/test-connection")
async def test_connection(config: ProviderConfig):
    """Test connection with inline config (from modal form)."""
    return await _do_test_connection(
        config.api_base, config.api_key, config.model,
        config.extra_headers, config.extra_body,
    )


@app.post("/api/providers/{provider_id}/test-connection")
async def test_connection_by_id(provider_id: str):
    """Test connection using a saved provider's full config (from panel)."""
    if provider_id not in active_providers:
        raise HTTPException(status_code=404, detail="Provider not found")
    p = active_providers[provider_id]
    return await _do_test_connection(
        p["api_base"], p.get("api_key", ""), p["model"],
        p.get("extra_headers", {}), p.get("extra_body", {}),
        p.get("max_tokens", 10), p.get("temperature", 0.0),
    )


@app.delete("/api/providers/{provider_id}")
async def delete_provider(provider_id: str):
    if provider_id not in active_providers:
        raise HTTPException(status_code=404, detail="Provider not found")
    del active_providers[provider_id]
    save_providers()
    return {"ok": True}


@app.post("/api/test")
async def start_test(req: TestRequest):
    if req.provider_id not in active_providers:
        raise HTTPException(status_code=404, detail="Provider not found")

    test_id = str(uuid.uuid4())[:8]
    active_tests[test_id] = {
        "provider": dict(active_providers[req.provider_id]),
        "prompt": req.prompt,
        "concurrency": req.concurrency,
        "results": None,
        "all_results": [],
    }
    return {"test_id": test_id}


@app.get("/api/test/{test_id}/stream")
async def stream_test(test_id: str):
    if test_id not in active_tests:
        raise HTTPException(status_code=404, detail="Test not found")
    return StreamingResponse(event_stream(test_id), media_type="text/event-stream")


@app.get("/api/test/{test_id}/results")
async def get_test_results(test_id: str):
    test_state = active_tests.get(test_id)
    if not test_state or not test_state.get("results"):
        raise HTTPException(status_code=404, detail="Results not found")
    return {
        "results": test_state["results"],
        "all_results": test_state.get("all_results", []),
    }


# ── Auto-Optimize QPS Endpoints ──────────────────────────────────────────────

active_optimizations: Dict[str, dict] = {}


async def auto_optimize_event_stream(test_id: str):
    opt_state = active_optimizations.get(test_id)
    if not opt_state:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Optimization not found'})}\n\n"
        return

    result_queue: asyncio.Queue = asyncio.Queue()
    asyncio.create_task(
        run_auto_optimize(
            test_id,
            opt_state["provider"],
            opt_state["prompt"],
            opt_state["max_ttft_ms"],
            opt_state["max_tpot_ms"],
            opt_state.get("stress_mode", False),
            result_queue,
        )
    )

    while True:
        try:
            msg = await asyncio.wait_for(result_queue.get(), timeout=300.0)
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Auto-optimize timed out'})}\n\n"
            return

        if msg is None:
            break
        yield f"data: {json.dumps(msg, default=str)}\n\n"


@app.post("/api/auto-optimize")
async def start_auto_optimize(req: AutoOptimizeRequest):
    if req.provider_id not in active_providers:
        raise HTTPException(status_code=404, detail="Provider not found")

    test_id = str(uuid.uuid4())[:8]
    active_optimizations[test_id] = {
        "provider": dict(active_providers[req.provider_id]),
        "prompt": req.prompt,
        "max_ttft_ms": req.max_ttft_ms,
        "max_tpot_ms": req.max_tpot_ms,
        "stress_mode": req.stress_mode,
        "results": None,
    }
    return {"test_id": test_id}


@app.get("/api/auto-optimize/{test_id}/stream")
async def stream_auto_optimize(test_id: str):
    if test_id not in active_optimizations:
        raise HTTPException(status_code=404, detail="Optimization not found")
    return StreamingResponse(
        auto_optimize_event_stream(test_id), media_type="text/event-stream"
    )


@app.get("/api/auto-optimize/{test_id}/results")
async def get_auto_optimize_results(test_id: str):
    opt_state = active_optimizations.get(test_id)
    if not opt_state or not opt_state.get("results"):
        raise HTTPException(status_code=404, detail="Results not found")
    return opt_state["results"]


# ── Startup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser

    load_providers()
    print(f"Loaded {len(active_providers)} provider(s)")

    def open_browser():
        time.sleep(0.5)
        webbrowser.open("http://127.0.0.1:8765")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
