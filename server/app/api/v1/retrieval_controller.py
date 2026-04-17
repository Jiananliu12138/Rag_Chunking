from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.exceptions import (
    CollectionNotFoundException,
    ModelLoadException,
    RetrievalException,
    to_http_exception,
)
from app.repositories.milvus_repository import MilvusRepository
from app.schemas.common import BaseResponse
from app.schemas.retrieval_schema import (
    RAGGenerateFileRequest,
    RAGGenerateFileResult,
    RAGRequest,
    RAGResult,
    SearchRequest,
    SearchResult,
)
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["Retrieval & Generation"])

_ERR_404 = {
    "description": "The target collection does not exist. Build the index first.",
    "content": {
        "application/json": {
            "example": {
                "success": False,
                "message": "Collection 'wiki_chunks' does not exist. Build the index first.",
                "data": None,
            }
        }
    },
}

_ERR_500 = {
    "description": "Retrieval or LLM generation failed.",
    "content": {
        "application/json": {
            "example": {
                "success": False,
                "message": "LLM generation failed: Connection refused",
                "data": None,
            }
        }
    },
}


def _get_retrieval_service() -> RetrievalService:
    return RetrievalService(MilvusRepository())


@router.post(
    "/search",
    response_model=BaseResponse[SearchResult],
    summary="Vector similarity search",
    description=(
        "Search the target collection and return the top-k most relevant chunks. "
        "The collection must already be built, and the embedding model/dimension "
        "should match the index configuration."
    ),
    responses={404: _ERR_404, 500: _ERR_500},
)
def search(
    request: SearchRequest,
    service: Annotated[RetrievalService, Depends(_get_retrieval_service)],
) -> BaseResponse[SearchResult]:
    try:
        result = service.search(request)
        return BaseResponse.ok(result)
    except (CollectionNotFoundException, ModelLoadException, RetrievalException) as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/generate",
    response_model=BaseResponse[RAGResult],
    summary="RAG generation",
    description=(
        "Full RAG flow: embed query, retrieve top-k chunks from Milvus, assemble context, "
        "and call the configured LLM chat endpoint (`/chat/completions`) to generate the answer."
    ),
    responses={404: _ERR_404, 500: _ERR_500},
)
def rag_generate(
    request: RAGRequest,
    service: Annotated[RetrievalService, Depends(_get_retrieval_service)],
) -> BaseResponse[RAGResult]:
    try:
        result = service.rag_generate(request)
        return BaseResponse.ok(result)
    except (CollectionNotFoundException, ModelLoadException, RetrievalException) as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/generate-file",
    response_model=BaseResponse[RAGGenerateFileResult],
    summary="Batch RAG generation from file",
    description=(
        "Read a jsonl input file, run retrieval plus chat-based generation for each record, "
        "and write the aggregated JSON result file."
    ),
    responses={404: _ERR_404, 500: _ERR_500},
)
def rag_generate_file(
    request: RAGGenerateFileRequest,
    service: Annotated[RetrievalService, Depends(_get_retrieval_service)],
) -> BaseResponse[RAGGenerateFileResult]:
    try:
        result = service.rag_generate_file(request)
        return BaseResponse.ok(result, message="RAG file generation completed")
    except (CollectionNotFoundException, ModelLoadException, RetrievalException) as exc:
        raise to_http_exception(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
