# VoiceBotLocal

A Pipecat AI voice agent built with a cascade pipeline (STT → LLM → TTS), optimized for Vietnamese language support.

## Configuration

- **Bot Type**: Web
- **Transport(s)**: SmallWebRTC, Daily (WebRTC)
- **Pipeline**: Cascade
  - **STT**: Whisper (Local)
  - **LLM**: Ollama (local, e.g. `qwen2.5:7b`, `qwen3:4b`)
  - **TTS**: VieNeu-TTS (Vietnamese, CPU-compatible)
- **Features**:
  - Vietnamese language support
  - Web search via Ollama Cloud API
  - Audio Recording
  - Transcription
  - Observability (Whisker + Tail)

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) running locally
- Node.js 18+ (for client)

## Setup

### 1. Install Ollama and pull a model

```bash
ollama pull qwen2.5:7b
# or for faster CPU performance:
ollama pull qwen3:4b
```

### 2. Server

```bash
cd server
```

**Install dependencies:**

```bash
uv sync
```

**Install VieNeu-TTS (choose one based on your platform):**

```bash
# Default install
pip install vieneu

# Windows users — CPU pre-built (required for llama-cpp on Windows)
pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/

# macOS users — ARM64/Apple Silicon (enables Metal GPU acceleration)
pip install vieneu --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal/
```

**Configure environment variables:**

```bash
cp .env.example .env
# Edit .env and fill in your values
```

**Run the bot:**

```bash
uv run bot.py
```

### 3. Client

```bash
cd client
npm install
cp env.example .env.local
npm run dev
```

Open http://localhost:5173

## Environment Variables

All options are documented in `server/.env.example`. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model to use |
| `OPENAI_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `VIENEU_MODE` | `turbo` | TTS mode: `standard`, `turbo` (CPU), `turbo_gpu` (CUDA) |
| `VIENEU_VOICE_INDEX` | `0` | Voice preset index (see table below) |
| `OLLAMA_API_KEY` | — | Ollama Cloud API key for web search (get at ollama.com/settings/keys) |
| `DAILY_API_KEY` | — | Daily.co API key (only needed for Daily transport) |

### VieNeu Voice Presets (Turbo mode)

| Index | Name | Gender | Region |
|-------|------|--------|--------|
| 0 | Bích Ngọc | Female | Northern |
| 1 | Phạm Tuyên | Male | Northern |
| 2 | Thục Đoan | Female | Southern |
| 3 | Xuân Vĩnh | Male | Southern |

### GPU Acceleration (Ollama)

`OLLAMA_NUM_GPU` is an **Ollama server-side** variable — set it before starting Ollama, not in `.env`:

```powershell
# Windows PowerShell — temporary (current session only)
$env:OLLAMA_NUM_GPU="-1"; ollama serve   # -1 = all layers to GPU, 0 = CPU only
```

```bash
# macOS / Linux
OLLAMA_NUM_GPU=-1 ollama serve
```

**Windows — set permanently (persists across reboots):**

```powershell
# Run PowerShell as Administrator
[System.Environment]::SetEnvironmentVariable("OLLAMA_NUM_GPU", "-1", "Machine")

# Verify
[System.Environment]::GetEnvironmentVariable("OLLAMA_NUM_GPU", "Machine")
# Expected output: -1

# Then restart Ollama (kill and reopen, or restart tray icon)
Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
Start-Process ollama -ArgumentList "serve"
```

Verify GPU is active after starting a model:

```powershell
ollama ps
# PROCESSOR column should show "100% GPU"
```

### GPU Acceleration (VieNeu TTS — `turbo_gpu` mode)

`turbo_gpu` mode requires PyTorch with CUDA. Check your CUDA version first:

```powershell
nvidia-smi
```

Then install the matching PyTorch build:

```powershell
# CUDA 12.1 (most common for RTX 30xx / 40xx)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then set in `server/.env`:

```env
VIENEU_MODE=turbo_gpu
```

> **Note:** PyTorch CUDA is ~2–3 GB. CPU modes (`turbo`, `standard`) do not require PyTorch.

> **Important — Python version:** PyTorch only supports Python 3.11–3.13. If you recreate the venv (e.g. switching from Python 3.14 to 3.11), PyTorch is **not carried over** — you must reinstall it manually after `uv sync`.

After installing, verify PyTorch is present before running the bot:

```powershell
uv pip list | Select-String "torch"
# Must show: torch, torchvision, torchaudio
```

## Model Files

Model files are cached locally in `server/models/huggingface/` (git-ignored). On first run, VieNeu will download the required models automatically. Subsequent runs load from cache instantly.

Whisper models are cached in the default HuggingFace cache (`~/.cache/huggingface/`).

## Web Search

The bot uses Ollama's web search API so the LLM can look up real-time information (interest rates, current events, etc.). The LLM decides autonomously when a search is needed.

To enable:
1. Get a free API key at [ollama.com/settings/keys](https://ollama.com/settings/keys)
2. Set `OLLAMA_API_KEY=<your-key>` in `server/.env`

## Project Structure

```
VoiceBotLocal/
├── server/
│   ├── bot.py                # Main bot pipeline
│   ├── vieneu_service.py     # VieNeu TTS pipecat service
│   ├── web_search_tools.py   # Ollama web search tool handlers
│   ├── pyproject.toml        # Python dependencies
│   ├── .env.example          # Environment variables template
│   ├── .env                  # Your config (git-ignored)
│   └── models/               # Local model cache (git-ignored)
│       └── huggingface/      # VieNeu model files
├── client/
│   ├── src/
│   ├── package.json
│   └── ...
├── .gitignore
└── README.md
```

## Observability

### Whisker — Live Pipeline Debugger

Visualize the pipeline and debug frames in real time.

1. Run an ngrok tunnel: `ngrok http 9090`
2. Go to [whisker.pipecat.ai](https://whisker.pipecat.ai/) and enter your ngrok URL
3. Start the bot and press connect

### Tail — Terminal Dashboard

Monitor sessions, logs, audio levels, and metrics in real time.

```bash
# In a second terminal while the bot is running:
pipecat tail
```

## Learn More

- [Pipecat Documentation](https://docs.pipecat.ai/)
- [VieNeu-TTS GitHub](https://github.com/pnnbao97/VieNeu-TTS)
- [Ollama Web Search](https://docs.ollama.com/capabilities/web-search)
- [Pipecat GitHub](https://github.com/pipecat-ai/pipecat)
- [Discord Community](https://discord.gg/pipecat)
