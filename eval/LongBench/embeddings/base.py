"""
Hugging Face 嵌入模型封装
支持 Bi-encoder 和 Cross-encoder 模型
"""

# ============================================================
# 标准库导入
# ============================================================
from __future__ import annotations
import os
import typing as t
from dataclasses import field
from typing import List

# ============================================================
# 第三方库导入
# ============================================================
import numpy as np
from pydantic.dataclasses import dataclass

# LangChain 导入
from langchain_core.embeddings import Embeddings

# Transformers 和 SentenceTransformers
import sentence_transformers
from transformers import AutoConfig
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
)
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from torch import Tensor


# ============================================================
# 默认配置
# ============================================================
DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"


# ============================================================
# 主类定义
# ============================================================
@dataclass
class HuggingfaceEmbeddings(Embeddings):
    """
    Hugging Face 嵌入模型封装类
    
    功能：
    1. 自动识别模型类型（Bi-encoder 或 Cross-encoder）
    2. 支持在线下载和本地加载
    3. 提供向量化接口
    
    支持的模型类型：
    - Bi-encoder: 用于语义检索（如 BGE, E5, MiniLM）
    - Cross-encoder: 用于结果重排序
    """
    
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    
    cache_folder: t.Optional[str] = None
    """Path to store models. 
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    
    model_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    
    encode_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        初始化模型
        自动识别模型类型并加载
        """
        # 加载模型配置
        config = AutoConfig.from_pretrained(self.model_name)
        
        # 检查模型架构是否为序列分类模型（Cross-encoder）
        intersection = np.intersect1d(
            list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values()),
            config.architectures if config.architectures else [],
        )
        self.is_cross_encoder = len(intersection) > 0

        # 根据模型类型加载对应的模型
        if self.is_cross_encoder:
            self.model = sentence_transformers.CrossEncoder(
                self.model_name, **self.model_kwargs
            )
        else:
            self.model = sentence_transformers.SentenceTransformer(
                self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
            )

        # 确保输出为张量格式
        if "convert_to_tensor" not in self.encode_kwargs:
            self.encode_kwargs["convert_to_tensor"] = True

    def embed_query(self, text: str) -> List[float]:
        """
        对单个查询文本进行向量化
        
        Args:
            text: 查询文本
            
        Returns:
            向量（列表形式）
        """
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对多个文档进行批量向量化
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        assert isinstance(
            self.model, SentenceTransformer
        ), "Model is not of the type Bi-encoder"
        
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, **self.encode_kwargs
        )

        assert isinstance(embeddings, Tensor)
        return embeddings.tolist()

    def predict(self, texts: List[List[str]]) -> List[List[float]]:
        """
        Cross-encoder 预测（用于重排序）
        
        Args:
            texts: 文本对列表 [["query", "doc1"], ["query", "doc2"], ...]
            
        Returns:
            相似度分数列表
        """
        assert isinstance(
            self.model, CrossEncoder
        ), "Model is not of the type CrossEncoder"

        predictions = self.model.predict(texts, **self.encode_kwargs)

        assert isinstance(predictions, Tensor)
        return predictions.tolist()
