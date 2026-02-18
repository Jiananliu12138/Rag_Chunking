"""
Meta-Chunking RAG Server — 应用入口
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from app.api.v1.router import api_v1_router
from app.config import get_settings
from app.core.exceptions import RAGBaseException, to_http_exception
from app.core.logging_config import logger
from app.core.path_setup import ensure_paths

OPENAPI_TAGS = [
    {
        "name": "分块 / Chunking",
        "description": (
            "对原始文档执行文本分块，支持四种策略：\n\n"
            "- **token** — 基于 tiktoken 的固定 Token 大小滑动窗口分块\n"
            "- **semantic** — 基于 HuggingFace 嵌入模型语义相似度的自适应分块\n"
            "- **llamaindex** — LlamaIndex `SimpleNodeParser` 固定窗口分块\n"
            "- **lumber** — 调用 vLLM 服务，由 LLM 识别主题边界进行内容感知分块"
        ),
    },
    {
        "name": "向量索引 / Index",
        "description": (
            "管理 Milvus Lite 本地向量数据库。\n\n"
            "每个 **collection** 对应 `milvus_data/` 目录下一个独立的 `.db` 文件。\n"
            "构建索引前需准备好嵌入模型（本地 HuggingFace 路径）。"
        ),
    },
    {
        "name": "检索与生成 / Retrieval & Generation",
        "description": (
            "执行向量相似度检索，或完整的 RAG 生成流程。\n\n"
            "RAG 流程：**查询向量化** → **Milvus top-k 检索** → **拼接上下文** → **vLLM 生成答案**。"
        ),
    },
    {
        "name": "端到端评估 / End-to-End Evaluation",
        "description": (
            "对 RAG 系统的整体输出质量进行评估，提供两条路径：\n\n"
            "- **传统指标**：F1、ROUGE-L、BLEU-1/2/3/4、BERTScore（可选）\n"
            "- **RAGAS**：Faithfulness、Answer Relevancy、Context Recall/Precision、"
            "Context Entity Recall、Noise Sensitivity"
        ),
    },
    {
        "name": "组件评估 / Component Evaluation",
        "description": (
            "对**分块组件本身**进行质量评估，不依赖下游检索/生成结果：\n\n"
            "- **chunk-quality**：边界清晰度（BC）+ 语义不相似度\n"
            "- **chunk-stickiness**：基于块间依赖图的结构熵黏连度评估"
        ),
    },
    {
        "name": "系统 / System",
        "description": "服务健康检查与基础信息接口。",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    ensure_paths()
    logger.info("RAG Server 启动，Milvus 数据目录: %s", settings.MILVUS_DATA_DIR)
    yield
    logger.info("RAG Server 关闭")


def custom_openapi():
    """自定义 OpenAPI Schema 生成函数，添加 Swagger UI 高级配置"""
    if app.openapi_schema:
        return app.openapi_schema
    
    settings = get_settings()
    openapi_schema = get_openapi(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        routes=app.routes,
        tags=OPENAPI_TAGS,
    )
    
    # 添加服务器配置
    openapi_schema["servers"] = [
        {"url": "http://localhost:8080", "description": "本地开发环境"},
        {"url": "http://0.0.0.0:8080", "description": "容器内环境"},
    ]
    
    # 添加联系信息和许可证
    openapi_schema["info"]["contact"] = {
        "name": "Meta-Chunking Project",
        "url": "https://github.com/your-repo/meta-chunking",
    }
    openapi_schema["info"]["license"] = {
        "name": "MIT",
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,  # 默认不展开 Models
            "docExpansion": "list",          # 默认展开接口列表但不展开详情
            "filter": True,                  # 启用搜索过滤
            "tryItOutEnabled": True,         # 默认启用 Try it out
            "persistAuthorization": True,    # 持久化 Authorization
            "displayRequestDuration": True,  # 显示请求耗时
        },
        lifespan=lifespan,
    )
    
    # 使用自定义 OpenAPI Schema
    app.openapi = custom_openapi

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 全局异常处理 ──────────────────────────────────────────────────────────
    @app.exception_handler(RAGBaseException)
    async def rag_exception_handler(request: Request, exc: RAGBaseException):
        http_exc = to_http_exception(exc)
        return JSONResponse(
            status_code=http_exc.status_code,
            content={"success": False, "message": exc.message, "data": None},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("未捕获异常: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(exc), "data": None},
        )

    # ── 路由 ──────────────────────────────────────────────────────────────────
    app.include_router(api_v1_router)

    # ── 健康检查 ──────────────────────────────────────────────────────────────
    @app.get("/health", tags=["系统 / System"], summary="健康检查")
    async def health():
        return {"status": "ok", "version": settings.APP_VERSION}

    return app


app = create_app()
