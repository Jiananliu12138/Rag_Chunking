from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class BaseResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str = "ok"
    data: Optional[T] = None

    @classmethod
    def ok(cls, data: T, message: str = "ok") -> "BaseResponse[T]":
        return cls(success=True, message=message, data=data)

    @classmethod
    def fail(cls, message: str) -> "BaseResponse[None]":
        return cls(success=False, message=message, data=None)


class PaginatedResponse(BaseModel, Generic[T]):
    total: int
    items: list[T]
