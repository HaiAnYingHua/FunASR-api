"""
FunASR FastAPI 服务主入口
"""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from funasr_service.api.routes import router
from funasr_service.core import get_asr_instance

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 从环境变量获取配置
MODEL_NAME = os.getenv("FUNASR_MODEL", "paraformer-zh")
DEVICE = os.getenv("FUNASR_DEVICE", "cpu")
VAD_MODEL = os.getenv("FUNASR_VAD_MODEL", None)
PUNC_MODEL = os.getenv("FUNASR_PUNC_MODEL", None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    启动时加载模型，关闭时清理资源
    """
    # 启动时初始化模型
    logger.info("=" * 50)
    logger.info("FunASR FastAPI 服务启动中...")
    logger.info(f"模型配置:")
    logger.info(f"  - 主模型: {MODEL_NAME}")
    logger.info(f"  - 设备: {DEVICE}")
    if VAD_MODEL:
        logger.info(f"  - VAD模型: {VAD_MODEL}")
    if PUNC_MODEL:
        logger.info(f"  - 标点模型: {PUNC_MODEL}")
    logger.info("=" * 50)

    try:
        # 预加载模型
        get_asr_instance(
            model_name=MODEL_NAME,
            device=DEVICE,
            vad_model=VAD_MODEL,
            punc_model=PUNC_MODEL
        )
        logger.info("模型加载完成，服务就绪")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

    yield

    # 关闭时清理
    logger.info("FunASR FastAPI 服务关闭中...")


# 创建FastAPI应用
app = FastAPI(
    title="FunASR语音识别服务",
    description="基于FunASR的语音识别HTTP API服务",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 根路径
@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "FunASR语音识别服务",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/v1/health",
            "recognize": "/api/v1/recognize",
            "recognize_batch": "/api/v1/recognize/batch",
            "recognize_buffer": "/api/v1/recognize/buffer",
            "docs": "/docs"
        }
    }


# 注册路由
app.include_router(router)


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": f"服务器内部错误: {str(exc)}",
            "data": None
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "funasr_service.main:app",
        host=host,
        port=port,
        reload=False,  # 生产环境关闭热重载
        log_level="info"
    )
