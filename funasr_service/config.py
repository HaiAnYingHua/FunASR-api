"""
FunASR 服务配置
可以通过环境变量覆盖这些配置
"""
import os
from typing import Optional


class Config:
    """服务配置类"""

    # 模型配置
    MODEL_NAME: str = os.getenv("FUNASR_MODEL", "paraformer-zh")
    DEVICE: str = os.getenv("FUNASR_DEVICE", "cpu")
    VAD_MODEL: Optional[str] = os.getenv("FUNASR_VAD_MODEL", None)
    PUNC_MODEL: Optional[str] = os.getenv("FUNASR_PUNC_MODEL", None)

    # 常用模型配置
    COMMON_MODELS = {
        "paraformer-zh": "paraformer-zh",
        "paraformer-zh-streaming": "paraformer-zh-streaming",
        "speech_paraformer-large": "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "speech_paraformer-vad-punc": "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "sense-voice": "iic/sense_voice_small"
    }

    # 服务配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("RELOAD", "False").lower() == "true"

    # 识别参数
    DEFAULT_BATCH_SIZE_S: int = int(os.getenv("DEFAULT_BATCH_SIZE_S", "300"))

    # 文件上传限制
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", "100"))  # MB
    SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".pcm", ".opus"]

    @classmethod
    def get_model_full_name(cls, model_name: str) -> str:
        """获取模型完整名称"""
        return cls.COMMON_MODELS.get(model_name, model_name)


config = Config()
