from fastapi import HTTPException, status


class RAGBaseException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ResourceNotFoundException(RAGBaseException):
    pass


class ChunkingException(RAGBaseException):
    pass


class IndexBuildException(RAGBaseException):
    pass


class RetrievalException(RAGBaseException):
    pass


class EvaluationException(RAGBaseException):
    pass


class ModelLoadException(RAGBaseException):
    pass


class CollectionNotFoundException(RAGBaseException):
    pass


def to_http_exception(exc: RAGBaseException) -> HTTPException:
    mapping = {
        ResourceNotFoundException: status.HTTP_404_NOT_FOUND,
        CollectionNotFoundException: status.HTTP_404_NOT_FOUND,
        ChunkingException: status.HTTP_422_UNPROCESSABLE_ENTITY,
        IndexBuildException: status.HTTP_500_INTERNAL_SERVER_ERROR,
        RetrievalException: status.HTTP_500_INTERNAL_SERVER_ERROR,
        EvaluationException: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ModelLoadException: status.HTTP_503_SERVICE_UNAVAILABLE,
    }
    status_code = mapping.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)
    return HTTPException(status_code=status_code, detail=exc.message)
