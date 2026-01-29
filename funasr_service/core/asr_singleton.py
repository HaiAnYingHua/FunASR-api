"""
FunASR ASR Singleton Core
单例模式的语音识别核心类，确保全局只加载一次模型
"""
import logging
import threading
from typing import Optional, List, Dict, Any, Union
from funasr import AutoModel

logger = logging.getLogger(__name__)


class ASRSingleton:
    """
    ASR单例类
    使用线程安全的单例模式，确保模型只加载一次
    """
    _instance: Optional['ASRSingleton'] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = "paraformer-zh",
        device: str = "cpu",
        vad_model: Optional[str] = None,
        punc_model: Optional[str] = None,
        spk_model: Optional[str] = None,
        **kwargs
    ):
        """
        初始化ASR模型（只执行一次）

        Args:
            model_name: 主模型名称，如 "paraformer-zh", "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            device: 运行设备，"cpu" 或 "cuda"
            vad_model: VAD模型名称（可选）
            punc_model: 标点模型名称（可选）
            spk_model: 说话人模型名称（可选）
            **kwargs: 其他模型参数
        """
        if ASRSingleton._initialized:
            return

        with self._lock:
            if ASRSingleton._initialized:
                return

            logger.info(f"正在加载ASR模型: {model_name}")
            logger.info(f"运行设备: {device}")

            model_kwargs = {
                "model": model_name,
                "device": device,
                "disable_update": True,
                **kwargs
            }

            if vad_model:
                logger.info(f"加载VAD模型: {vad_model}")
                model_kwargs["vad_model"] = vad_model

            if punc_model:
                logger.info(f"加载标点模型: {punc_model}")
                model_kwargs["punc_model"] = punc_model

            if spk_model:
                logger.info(f"加载说话人模型: {spk_model}")
                model_kwargs["spk_model"] = spk_model

            self.model = AutoModel(**model_kwargs)
            self.model_name = model_name
            self.device = device

            ASRSingleton._initialized = True
            logger.info("ASR模型加载完成")

    def recognize_file(
        self,
        audio_path: str,
        batch_size_s: int = 300,
        language: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        识别音频文件

        Args:
            audio_path: 音频文件路径
            batch_size_s: 批处理大小（秒）
            language: 语言代码（可选）
            **kwargs: 其他参数

        Returns:
            识别结果列表，每个元素包含:
            - text: 识别文本
            - timestamp: 时间戳（如果模型支持）
            - sentence_info: 句子信息（如果有标点模型）
        """
        if not self.model:
            raise RuntimeError("ASR模型未初始化")

        inference_kwargs = {
            "batch_size_s": batch_size_s,
            **kwargs
        }

        if language:
            inference_kwargs["language"] = language

        try:
            result = self.model.generate(input=audio_path, **inference_kwargs)
            return result
        except Exception as e:
            logger.error(f"识别失败: {e}")
            raise

    def recognize_buffer(
        self,
        audio_bytes: bytes,
        fs: int = 16000,
        batch_size_s: int = 300,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        识别音频缓冲区

        Args:
            audio_bytes: 音频字节数据
            fs: 采样率
            batch_size_s: 批处理大小（秒）
            **kwargs: 其他参数

        Returns:
            识别结果列表
        """
        if not self.model:
            raise RuntimeError("ASR模型未初始化")

        import numpy as np

        # 假设输入是PCM 16位数据
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0

        inference_kwargs = {
            "batch_size_s": batch_size_s,
            **kwargs
        }

        try:
            result = self.model.generate(input=audio_data, **inference_kwargs)
            return result
        except Exception as e:
            logger.error(f"识别失败: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "initialized": ASRSingleton._initialized
        }

    @classmethod
    def reset(cls):
        """重置单例（用于测试或重新加载模型）"""
        with cls._lock:
            if cls._instance:
                cls._instance.model = None
            cls._instance = None
            cls._initialized = False


# 全局获取ASR实例的函数
def get_asr_instance(
    model_name: str = "paraformer-zh",
    device: str = "cpu",
    vad_model: Optional[str] = None,
    punc_model: Optional[str] = None,
    **kwargs
) -> ASRSingleton:
    """
    获取ASR单例实例

    Args:
        model_name: 模型名称
        device: 运行设备
        vad_model: VAD模型
        punc_model: 标点模型
        **kwargs: 其他参数

    Returns:
        ASRSingleton实例
    """
    return ASRSingleton(
        model_name=model_name,
        device=device,
        vad_model=vad_model,
        punc_model=punc_model,
        **kwargs
    )
