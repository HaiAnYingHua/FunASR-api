"""
FastAPI 路由定义
提供语音识别的HTTP接口
"""
import os
import tempfile
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import asyncio

from ..core.asr_singleton import get_asr_instance

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ASR"])


class RecognizeRequest(BaseModel):
    """识别请求模型"""
    language: Optional[str] = Field(None, description="语言代码")
    batch_size_s: int = Field(300, description="批处理大小（秒）")
    return_timestamp: bool = Field(False, description="是否返回时间戳")


class RecognizeResponse(BaseModel):
    """识别响应模型"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_info: Optional[Dict[str, Any]] = None


def parse_asr_result(result: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    解析ASR结果为统一格式

    Args:
        result: ASR原始结果

    Returns:
        解析后的结果字典
    """
    if not result or len(result) == 0:
        return {
            "text": "",
            "timestamp": [],
            "sentence_info": []
        }

    first_result = result[0]

    return {
        "text": first_result.get("text", ""),
        "timestamp": first_result.get("timestamp", []),
        "sentence_info": first_result.get("sentence_info", []),
        "key": first_result.get("key", ""),
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查接口
    检查服务状态和模型信息
    """
    try:
        asr = get_asr_instance()
        model_info = asr.get_model_info()
        return HealthResponse(status="healthy", model_info=model_info)
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return HealthResponse(status="unhealthy", model_info=None)


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize_file(
    file: UploadFile = File(..., description="音频文件"),
    language: Optional[str] = Form(None),
    batch_size_s: int = Form(300),
    return_timestamp: bool = Form(False)
):
    """
    音频文件识别接口

    支持的音频格式: wav, mp3, m4a, flac, ogg 等

    Args:
        file: 上传的音频文件
        language: 语言代码（可选）
        batch_size_s: 批处理大小
        return_timestamp: 是否返回时间戳

    Returns:
        RecognizeResponse: 识别结果
    """
    # 检查文件扩展名
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件名")

    ext = os.path.splitext(file.filename)[1].lower()
    supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.pcm', '.opus']
    if ext not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式: {ext}，支持的格式: {', '.join(supported_formats)}"
        )

    # 保存临时文件
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"处理文件: {file.filename}, 大小: {len(content)} bytes")

        # 获取ASR实例并识别
        asr = get_asr_instance()

        # 在线程池中执行CPU密集型任务
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            asr.recognize_file,
            temp_file_path,
            batch_size_s,
            language
        )

        parsed_result = parse_asr_result(result)

        logger.info(f"识别成功: {parsed_result['text'][:50]}...")

        return RecognizeResponse(
            success=True,
            message="识别成功",
            data=parsed_result
        )

    except Exception as e:
        logger.error(f"识别失败: {e}")
        return RecognizeResponse(
            success=False,
            message=f"识别失败: {str(e)}",
            data=None
        )
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")


@router.post("/recognize/batch")
async def recognize_batch(
    files: List[UploadFile] = File(..., description="多个音频文件"),
    language: Optional[str] = Form(None),
    batch_size_s: int = Form(300)
):
    """
    批量音频文件识别接口

    Args:
        files: 多个音频文件
        language: 语言代码
        batch_size_s: 批处理大小

    Returns:
        批量识别结果
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="最多支持10个文件同时识别")

    results = []
    temp_files = []

    try:
        asr = get_asr_instance()
        loop = asyncio.get_event_loop()

        for file in files:
            # 保存临时文件
            ext = os.path.splitext(file.filename or ".wav")[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_files.append(temp_file.name)

            # 异步识别
            result = await loop.run_in_executor(
                None,
                asr.recognize_file,
                temp_file.name,
                batch_size_s,
                language
            )

            parsed_result = parse_asr_result(result)
            results.append({
                "filename": file.filename,
                "success": True,
                "data": parsed_result
            })

        return {
            "success": True,
            "message": f"成功识别 {len(results)} 个文件",
            "results": results
        }

    except Exception as e:
        logger.error(f"批量识别失败: {e}")
        return {
            "success": False,
            "message": f"批量识别失败: {str(e)}",
            "results": results
        }
    finally:
        # 清理所有临时文件
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")


class RecognizeBufferRequest(BaseModel):
    """音频缓冲区识别请求"""
    audio_data: str = Field(..., description="Base64编码的音频数据")
    format: str = Field("wav", description="音频格式")
    sample_rate: int = Field(16000, description="采样率")
    language: Optional[str] = Field(None, description="语言代码")


@router.post("/recognize/buffer")
async def recognize_buffer(request: RecognizeBufferRequest):
    """
    音频缓冲区识别接口

    接收Base64编码的音频数据进行识别

    Args:
        request: 包含音频数据和参数的请求

    Returns:
        识别结果
    """
    import base64
    import io

    try:
        # 解码Base64音频数据
        audio_bytes = base64.b64decode(request.audio_data)

        # 根据格式处理
        if request.format.lower() == "wav":
            # WAV格式，需要提取PCM数据
            import wave
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_bytes = frames

        # 获取ASR实例并识别
        asr = get_asr_instance()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            asr.recognize_buffer,
            audio_bytes,
            request.sample_rate
        )

        parsed_result = parse_asr_result(result)

        return RecognizeResponse(
            success=True,
            message="识别成功",
            data=parsed_result
        )

    except Exception as e:
        logger.error(f"缓冲区识别失败: {e}")
        return RecognizeResponse(
            success=False,
            message=f"识别失败: {str(e)}",
            data=None
        )
