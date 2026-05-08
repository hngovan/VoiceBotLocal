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
from pipecat.audio.vad.vad_analyzer import VADParams
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
        ttfs_p99_latency=float(os.getenv("WHISPER_TTFS_P99", "1.5")),
        settings=WhisperSTTService.Settings(
            model=os.getenv("OPENAI_MODEL"),
            language=os.getenv("WHISPER_LANGUAGE", "vi"),
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
                "Bạn là trợ lý ảo nữ của ngân hàng, giọng nói thân thiện, chuyên nghiệp và tự nhiên. "
                "Khi bắt đầu cuộc hội thoại, hãy chủ động chào khách hàng bằng câu ngắn gọn như: "
                "'Xin chào, em là trợ lý ảo của ngân hàng. Anh/chị cần em hỗ trợ vấn đề gì ạ?' "
                "Nhiệm vụ của bạn là hỗ trợ khách hàng về các vấn đề ngân hàng như: "
                "tài khoản thanh toán, tiết kiệm, vay vốn, thẻ tín dụng, chuyển tiền, "
                "lãi suất, sản phẩm tài chính và các dịch vụ ngân hàng khác. "
                "Luôn trả lời bằng tiếng Việt, lịch sự, tự nhiên và dễ hiểu. "
                "Nếu cần tra cứu thông tin cập nhật như lãi suất hoặc chính sách mới nhất, "
                "hãy nói tự nhiên như: "
                "'Anh/chị chờ em một chút nhé, em kiểm tra giúp mình ngay ạ.' "
                "Sau đó mới dùng công cụ tìm kiếm. "
                "Sau khi có kết quả, chỉ tóm tắt ngắn gọn trong 2 đến 3 câu dễ nghe. "
                "QUAN TRỌNG - Định dạng trả lời: "
                "Câu trả lời sẽ được đọc bằng giọng nói, vì vậy TUYỆT ĐỐI KHÔNG dùng: "
                "danh sách đánh số (1. 2. 3.), gạch đầu dòng, emoji, markdown, bảng biểu. "
                "Chỉ viết thành câu văn liền mạch, tự nhiên như đang nói chuyện. "
                "Ví dụ SAI: '1. Tổng tiền vay là 5 tỷ. 2. Lãi suất 8%.' "
                "Ví dụ ĐÚNG: 'Tổng tiền vay là 5 tỷ với lãi suất 8% mỗi năm.' "
                "Ưu tiên câu ngắn, hội thoại tự nhiên, giống nhân viên hỗ trợ thật. "
                "Không trả lời quá dài. "
                "Luôn đi thẳng vào vấn đề nhưng vẫn giữ sự thân thiện."
            ),
            extra={
                "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
            },
        ),
    )

    context = LLMContext()
    context.set_tools(TOOLS)
    llm.register_function("web_search", handle_web_search)
    llm.register_function("web_fetch", handle_web_fetch)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                stop_secs=float(os.getenv("VAD_STOP_SECS", "0.5")),
            )),),
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
