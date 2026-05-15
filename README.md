# Concurrent LLM Evaluation Framework

A graphical desktop tool for benchmarking LLM concurrency performance. Compatible with any OpenAI-format API, supporting multi-provider management, real-time streaming metrics, hardware auto-detection, and automatic QPS optimization.

## Features

### Concurrency Testing
- Configurable concurrency (1–200) with custom test prompts
- Real-time SSE progress streaming during tests
- Per-request metrics: total latency, TTFT (Time To First Token), token count
- Aggregate statistics: avg/P50/P90/P95/P99 latency, success rate, min/max TTFT
- Summary cards with test duration, model/provider info

### Charts & Visualization
- Latency distribution histogram (20 bins)
- Percentile comparison bar chart (avg/P50/P90/P95/P99)
- Request timeline scatter plot (success/error color-coded)
- TTFT distribution histogram (15 bins)
- All charts powered by Chart.js with dark theme

### Provider Management
- CRUD interface for multiple model providers (OpenAI-compatible API)
- Optional API key support (blank = no Authorization header)
- Connection test button to verify endpoints
- Custom headers and extra request body fields (JSON)
- Persistent storage in `providers_config.json`

### Hardware Auto-Detection
- CPU: model name, physical/logical cores, clock speed, architecture
- RAM: total/available capacity, per-stick capacity and speed
- GPU (NVIDIA): model name, VRAM, CUDA cores, memory bus width, memory bandwidth, compute capability, PCIe generation and link width
- Refresh button for re-detection

### Model Source Detection
- Auto-discovers running Ollama (`127.0.0.1:11434`) and LM Studio (`127.0.0.1:1234`)
- Displays loaded models with parameter count, quantization level (Q4_K_M/F16/etc.)
- Estimates VRAM usage based on parameter count × quantization bit-width

### Auto-Optimize QPS
- Two modes: **Quick Scan** (max_tokens=32 for fast iteration) and **Stress Test** (full tokens for realistic simulation)
- Iteratively doubles concurrency (1→2→4→8→...) until TTFT or TPOT exceeds user-defined SLA thresholds
- Real-time iteration log with pass/fail indicators
- 4 auto-updating charts: QPS curve, TTFT growth, TPOT change, success rate trend
- Reports optimal concurrency and QPS at the end

### Test History
- Stores last 20 test results in browser localStorage
- Full chart and metric card replay for any history entry
- Sortable detail table (by request ID, latency, TTFT, token count)
- Delete individual entries

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.8+, FastAPI, httpx (async), uvicorn |
| Frontend | Single-file HTML5 + CSS3 + Vanilla JS, Chart.js 4 |
| Hardware | psutil, pynvml (NVIDIA), PowerShell/WMIC |
| Persistence | JSON file (providers), localStorage (history) |
| Desktop | Auto-opens browser on startup |

## Quick Start

```bash
pip install fastapi uvicorn httpx pydantic jinja2 psutil nvidia-ml-py
python concurrency_tester.py
```

Browser opens automatically at `http://127.0.0.1:8765`.

## Project Structure

```
├── concurrency_tester.py       # FastAPI backend (all routes, test logic, hardware detection)
├── templates/
│   └── index.html              # Single-file SPA (inline CSS + JS + Chart.js)
├── providers_config.json       # Auto-generated provider configuration
└── README.md
```

## Supported Platforms

| Component | Support |
|-----------|---------|
| API Format | OpenAI-compatible chat completions (streaming + non-streaming fallback) |
| OS | Windows 10/11 (primary), Linux (partial — GPU detection limited) |
| GPU Detection | NVIDIA only (via pynvml) |
| Model Sources | Ollama, LM Studio |
| Browsers | Chrome, Edge, Firefox (modern versions) |

## Limitations

- **Single provider per test**: Each test targets one provider; cross-provider comparisons require manual runs
- **NVIDIA GPU only**: GPU detection uses pynvml; AMD/Intel GPUs are not detected
- **Windows-first**: Some hardware detection paths (RAM stick info via PowerShell) are Windows-specific; Linux uses dmidecode where available
- **No persistent test database**: Test history is stored in browser localStorage (max 20 entries, per-browser, lost on clear)
- **Sequential auto-optimize**: Each concurrency level runs a full synchronous batch; total optimization time grows with the number of iterations (~2× concurrency steps)
- **No distributed testing**: Single-machine only; no multi-node or cluster support
- **No authentication**: The web UI has no login/auth — intended for local use only
- **Streaming parsing**: Assumes standard SSE format (`data: {...}` lines); non-standard streaming formats may not capture TTFT correctly
- **Quantization detection**: LM Studio model quantization is guessed from filename patterns; Ollama provides accurate quantization info via API
- **Theoretical estimates are approximate**: Single-stream tok/s and max batch estimates use simplified formulas (bandwidth ÷ model_size × 0.75) and assume uniform KV cache overhead
- **Reasoning models**: Models that use `delta.reasoning` instead of `delta.content` are partially supported (TTFT detection works, but token counting may differ)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve SPA HTML |
| GET | `/api/providers` | List all providers (API keys masked) |
| POST | `/api/providers` | Create a provider |
| PUT | `/api/providers/{id}` | Update a provider |
| DELETE | `/api/providers/{id}` | Delete a provider |
| POST | `/api/test-connection` | Test connection with inline config |
| POST | `/api/providers/{id}/test-connection` | Test connection using stored credentials |
| POST | `/api/test` | Start a concurrency test |
| GET | `/api/test/{id}/stream` | SSE stream for test progress |
| GET | `/api/test/{id}/results` | Get final test results |
| GET | `/api/hardware` | Auto-detect hardware specs |
| GET | `/api/model-sources` | Detect Ollama / LM Studio models |
| POST | `/api/auto-optimize` | Start auto-optimize QPS |
| GET | `/api/auto-optimize/{id}/stream` | SSE stream for auto-optimize progress |
| GET | `/api/auto-optimize/{id}/results` | Get auto-optimize results |

## License

MIT
