#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""VoiceBotLocal - Pipecat Voice Agent

Cascade pipeline: Speech-to-Text → LLM → Text-to-Speech

Services:
- Whisper  (STT, local)
- Ollama   (LLM, local)
- VieNeu   (TTS, Vietnamese, CPU-compatible)

Run the bot::

    uv run bot.py
"""


from pathlib import Path
from dotenv import load_dotenv
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.runner.types import SmallWebRTCRunnerArguments
try:
    from pipecat.transports.daily.transport import DailyTransport, DailyParams
    DAILY_AVAILABLE = True
except Exception:
    DAILY_AVAILABLE = False
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat_whisker import WhiskerObserver
import os
from pipecat.transports.base_transport import BaseTransport
import datetime
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
import wave
from pipecat.runner.types import DailyRunnerArguments
import aiofiles
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_response_universal import AssistantTurnStoppedMessage, UserTurnStoppedMessage
import io
from vieneu_service import VieNeuTTSService
from web_search_tools import TOOLS, handle_web_fetch, handle_web_search
from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.runner.types import RunnerArguments
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat_tail.observer import TailObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frameworks.rtvi import RTVIProcessor

load_dotenv(override=True)

MODELS_DIR = Path(__file__).parent / "models"


async def save_audio_file(audio: bytes, filename: str, sample_rate: int, num_channels: int):
    """Save audio data to a WAV file."""
    if len(audio) > 0:
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Audio saved to {filename}")


async def run_bot(transport: BaseTransport):
    """Main bot logic."""
    logger.info("Starting bot")

    # Speech-to-Text service
    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(
            model=os.getenv("OPENAI_MODEL"),
        ),
    )

    # Text-to-Speech service
    tts = VieNeuTTSService(
        voice_index=int(os.getenv("VIENEU_VOICE_INDEX", "0")),
    )

    # LLM service
    llm = OLLamaLLMService(
        settings=OLLamaLLMService.Settings(
            model=os.getenv("OLLAMA_MODEL"),
            system_instruction=(
                    "Bạn là chuyên viên tư vấn ngân hàng chuyên nghiệp và thân thiện. "
                    "Nhiệm vụ của bạn là hỗ trợ khách hàng về các vấn đề ngân hàng như: "
                    "tài khoản thanh toán, tiết kiệm, vay vốn, thẻ tín dụng, chuyển tiền, "
                    "lãi suất, sản phẩm tài chính và các dịch vụ ngân hàng khác. "
                    "Luôn trả lời bằng tiếng Việt, lịch sự và dễ hiểu. "
                    "Nếu câu hỏi cần thông tin cập nhật như lãi suất hoặc chính sách mới nhất, "
                    "hãy nói ngắn gọn 'Để có thông tin chính xác, tôi sẽ tra cứu ngay.' rồi dùng công cụ tìm kiếm. "
                    "Sau khi có kết quả, tóm tắt trong 2 đến 3 câu ngắn, không liệt kê toàn bộ dữ liệu. "
                    "Câu trả lời sẽ được đọc to nên tránh dùng ký hiệu, emoji, dấu đầu dòng hoặc bảng biểu. "
                    "Luôn trả lời bằng tiếng Việt, ngắn gọn, súc tích và đi thẳng vào vấn đề."
                ),
        ),
    )

    context = LLMContext()
    context.set_tools(TOOLS)
    llm.register_function("web_search", handle_web_search)
    llm.register_function("web_fetch", handle_web_fetch)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),),
    )

    # Audio recording
    audio_buffer = AudioBufferProcessor()

    rtvi = RTVIProcessor()

    # Pipeline - assembled from reusable components
    pipeline = Pipeline([
        transport.input(),

        rtvi,

        stt,

        user_aggregator,

        llm,

        tts,


        transport.output(),

        audio_buffer,

        assistant_aggregator,

    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[
            WhiskerObserver(pipeline),
            TailObserver(),
        ],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi_proc):
        # Kick off the conversation
        context.add_message(
            {"role": "user", "content": "Xin hãy giới thiệu bản thân với tư cách là chuyên viên tư vấn ngân hàng."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Start recording audio
        await audio_buffer.start_recording()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}user: {message.content}"
        logger.info(f"Transcript: {line}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}assistant: {message.content}"
        logger.info(f"Transcript: {line}")

    @audio_buffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/merged_{timestamp}.wav"
        os.makedirs("recordings", exist_ok=True)
        await save_audio_file(audio, filename, sample_rate, num_channels)

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""

    transport = None

    match runner_args:
        case DailyRunnerArguments():
            if not DAILY_AVAILABLE:
                logger.error(
                    "Daily transport is not available on this platform (Windows). Use SmallWebRTC instead.")
                return
            transport = DailyTransport(
                runner_args.room_url,
                runner_args.token,
                "Pipecat Bot",
                params=DailyParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                ),
            )
        case SmallWebRTCRunnerArguments():
            webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection

            transport = SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                ),
            )
        case _:
            logger.error(
                f"Unsupported runner arguments type: {type(runner_args)}")
            return

    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
