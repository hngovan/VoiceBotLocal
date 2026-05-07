import huggingface_hub.constants as _hf_constants
import huggingface_hub
import asyncio
import os
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.tts_service import TTSService

# Store HuggingFace model cache inside the project so teammates can reuse it
_MODELS_DIR = Path(__file__).parent / "models" / "huggingface"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Patch huggingface_hub constants directly — os.environ.setdefault is too late
# because huggingface_hub reads HF_HUB_CACHE at import time, before dotenv loads.

_hf_constants.HF_HUB_CACHE = str(_MODELS_DIR)
huggingface_hub.constants.HF_HUB_CACHE = str(_MODELS_DIR)

# VIENEU_MODE options:
#   standard   — GGUF via llama-cpp, CPU-compatible (default)
#   turbo      — GGUF via llama-cpp, CPU-compatible, faster backbone
#   turbo_gpu  — PyTorch/LMDeploy, requires CUDA GPU


class VieNeuTTSService(TTSService):
    """Vietnamese TTS using VieNeu-TTS-v2.

    Env vars:
        VIENEU_MODE        standard | turbo | turbo_gpu  (default: turbo)
        VIENEU_VOICE_INDEX 0-based preset voice index    (default: 0)
    """

    SAMPLE_RATE = 24000

    def __init__(
        self,
        *,
        voice_index: int = 0,
        **kwargs,
    ):
        kwargs.setdefault("sample_rate", self.SAMPLE_RATE)
        super().__init__(push_start_frame=True, push_stop_frames=True, **kwargs)
        self._voice_index = voice_index
        self._mode = os.getenv("VIENEU_MODE", "standard").lower()
        self._tts = None
        self._voice_dict = None

    def can_generate_metrics(self) -> bool:
        return True

    def _load_model(self):
        from vieneu import Vieneu

        logger.debug(
            f"Loading VieNeu TTS model (mode={self._mode}, cache={_MODELS_DIR})...")

        match self._mode:
            case "turbo":
                tts = Vieneu(mode="turbo", device="cpu")
            case "turbo_gpu":
                tts = Vieneu(mode="turbo_gpu", device="cuda")
            case _:
                tts = Vieneu(mode="standard")

        voices = tts.list_preset_voices()
        if voices:
            idx = min(self._voice_index, len(voices) - 1)
            desc, voice_name = voices[idx]
            voice_dict = tts.get_preset_voice(voice_name)
            logger.debug(f"VieNeu voice selected: {desc} ({voice_name})")
        else:
            voice_dict = None
            logger.warning(
                "VieNeu: no preset voices found, using model default")

        logger.debug(f"VieNeu TTS model loaded (mode={self._mode})")
        return tts, voice_dict

    async def start(self, frame):
        await super().start(frame)
        if self._tts is None:
            loop = asyncio.get_event_loop()
            self._tts, self._voice_dict = await loop.run_in_executor(None, self._load_model)

    def _run_infer(self, text: str) -> np.ndarray:
        kwargs = {"text": text}
        if self._voice_dict is not None:
            kwargs["voice"] = self._voice_dict
        audio = self._tts.infer(**kwargs)
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        return np.asarray(audio, dtype=np.float32)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_tts_usage_metrics(text)

            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(None, lambda: self._run_infer(text))

            # float32 [-1, 1] → int16 PCM
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            pcm_bytes = pcm.tobytes()

            await self.stop_ttfb_metrics()

            CHUNK_SIZE = 8192
            for i in range(0, len(pcm_bytes), CHUNK_SIZE):
                chunk = pcm_bytes[i: i + CHUNK_SIZE]
                if chunk:
                    yield TTSAudioRawFrame(chunk, self.SAMPLE_RATE, 1, context_id=context_id)

        except ImportError as e:
            msg = f"Thiếu dependency cho mode '{self._mode}': {e}"
            logger.error(f"{self} {msg}")
            yield ErrorFrame(error=msg)
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
            yield ErrorFrame(error=str(e))
