from fastapi import APIRouter

from app.api.v1.chunk_controller import router as chunk_router
from app.api.v1.component_eval_controller import router as component_eval_router
from app.api.v1.eval_controller import router as eval_router
from app.api.v1.file_browser_controller import router as file_browser_router
from app.api.v1.index_controller import router as index_router
from app.api.v1.retrieval_controller import router as retrieval_router
from app.api.v1.testset_controller import router as testset_router

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(chunk_router)
api_v1_router.include_router(index_router)
api_v1_router.include_router(retrieval_router)
api_v1_router.include_router(eval_router)
api_v1_router.include_router(component_eval_router)
api_v1_router.include_router(file_browser_router)
api_v1_router.include_router(testset_router)
