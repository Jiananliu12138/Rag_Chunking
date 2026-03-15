from pydantic import BaseModel, Field


class FileSystemEntry(BaseModel):
    name: str = Field(..., description="Entry display name")
    path: str = Field(..., description="Absolute path")
    is_dir: bool = Field(..., description="Whether this entry is a directory")
    size_bytes: int | None = Field(None, description="File size in bytes if available")


class FileBrowserResult(BaseModel):
    current_path: str = Field(..., description="Current absolute directory path")
    parent_path: str | None = Field(None, description="Parent directory path if available")
    roots: list[str] = Field(default_factory=list, description="Available filesystem roots")
    entries: list[FileSystemEntry] = Field(default_factory=list, description="Entries in the current directory")
