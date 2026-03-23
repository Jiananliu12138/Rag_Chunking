from __future__ import annotations

from typing import Iterable

import grpc
from google.protobuf import struct_pb2
from pydantic import ValidationError

from app.core.exceptions import (
    EvaluationException,
    ModelLoadException,
    RAGBaseException,
    ResourceNotFoundException,
)
from app.rpc.generated import evaluation_service_pb2 as pb2
from app.rpc.generated import evaluation_service_pb2_grpc as pb2_grpc
from app.schemas.eval_schema import (
    ChunkQualityRequest,
    ChunkStickinessRequest,
    RAGASEvalRequest,
    TraditionalEvalRequest,
    RetrievalEvalRequest,
)
from app.services.component_eval_service import ComponentEvalService
from app.services.eval_service import EvalService


def _abort_from_exception(context: grpc.ServicerContext, exc: Exception) -> None:
    if isinstance(exc, ValidationError):
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, exc.json())

    if isinstance(exc, ResourceNotFoundException):
        context.abort(grpc.StatusCode.NOT_FOUND, exc.message)

    if isinstance(exc, ModelLoadException):
        context.abort(grpc.StatusCode.UNAVAILABLE, exc.message)

    if isinstance(exc, EvaluationException):
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, exc.message)

    if isinstance(exc, RAGBaseException):
        context.abort(grpc.StatusCode.INTERNAL, exc.message)

    context.abort(grpc.StatusCode.INTERNAL, str(exc))


def _value_to_python(value: struct_pb2.Value):
    kind = value.WhichOneof("kind")
    if kind == "null_value":
        return None
    if kind == "number_value":
        return value.number_value
    if kind == "string_value":
        return value.string_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "struct_value":
        return {key: _value_to_python(item) for key, item in value.struct_value.fields.items()}
    if kind == "list_value":
        return [_value_to_python(item) for item in value.list_value.values]
    return None


def _python_to_value(data) -> struct_pb2.Value:
    value = struct_pb2.Value()
    _fill_value(value, data)
    return value


def _fill_value(value: struct_pb2.Value, data) -> None:
    if data is None:
        value.null_value = struct_pb2.NullValue.NULL_VALUE
        return

    if isinstance(data, bool):
        value.bool_value = data
        return

    if isinstance(data, (int, float)):
        value.number_value = float(data)
        return

    if isinstance(data, str):
        value.string_value = data
        return

    if isinstance(data, dict):
        struct_value = struct_pb2.Struct()
        for key, item in data.items():
            _fill_value(struct_value.fields[str(key)], item)
        value.struct_value.CopyFrom(struct_value)
        return

    if isinstance(data, (list, tuple)):
        list_value = struct_pb2.ListValue()
        for item in data:
            child = list_value.values.add()
            _fill_value(child, item)
        value.list_value.CopyFrom(list_value)
        return

    value.string_value = str(data)


def _optional_scalar(message: object, field_name: str):
    return getattr(message, field_name) if message.HasField(field_name) else None


def _optional_value(message: object, field_name: str):
    if not message.HasField(field_name):
        return None
    return _value_to_python(getattr(message, field_name))


def _build_metric_summary(summary) -> pb2.RagasMetricSummary:
    return pb2.RagasMetricSummary(
        mean=summary.mean,
        min=summary.min,
        max=summary.max,
    )


def _build_chunk_pair_result(details: Iterable) -> list[pb2.ChunkPairResult]:
    items: list[pb2.ChunkPairResult] = []
    for detail in details:
        item = pb2.ChunkPairResult()
        if detail.semantic_dissimilarity is not None:
            item.semantic_dissimilarity = detail.semantic_dissimilarity
        if detail.boundary_clarity is not None:
            item.boundary_clarity = detail.boundary_clarity
        items.append(item)
    return items


def _field_default(model_cls, field_name: str):
    return model_cls.model_fields[field_name].default


class EvalRpcServicer(pb2_grpc.EvalRpcServiceServicer):
    def __init__(self, eval_service: EvalService | None = None) -> None:
        self._eval_service = eval_service or EvalService()

    def TraditionalEval(
        self,
        request: pb2.TraditionalEvalRequest,
        context: grpc.ServicerContext,
    ) -> pb2.TraditionalEvalResponse:
        try:
            payload = TraditionalEvalRequest(
                predictions=list(request.predictions) or None,
                answers=[list(item.items) for item in request.answers] or None,
                test=_optional_value(request, "test"),
                enable_bert_score=_optional_scalar(request, "enable_bert_score"),
                bert_score_model=_optional_scalar(request, "bert_score_model"),
                bert_score_device=_optional_scalar(request, "bert_score_device"),
            )
            result = self._eval_service.evaluate_traditional(payload)
            response = pb2.TraditionalEvalResponse(
                f1=result.f1,
                rouge_l=result.rouge_l,
                bleu_1=result.bleu_1,
                bleu_2=result.bleu_2,
                bleu_3=result.bleu_3,
                bleu_4=result.bleu_4,
                sample_count=result.sample_count,
            )
            if result.bert_score_f1 is not None:
                response.bert_score_f1 = result.bert_score_f1
            return response
        except (ValidationError, RAGBaseException, Exception) as exc:
            _abort_from_exception(context, exc)

    def RagasEval(
        self,
        request: pb2.RagasEvalRequest,
        context: grpc.ServicerContext,
    ) -> pb2.RagasEvalResponse:
        try:
            payload = RAGASEvalRequest(
                test=_optional_value(request, "test"),
                vllm_api_base=_optional_scalar(request, "vllm_api_base"),
                vllm_api_key=_optional_scalar(request, "vllm_api_key"),
                vllm_model_name=_optional_scalar(request, "vllm_model_name"),
                embedding_model_path=_optional_scalar(request, "embedding_model_path"),
                device=_optional_scalar(request, "device"),
                enable_cache=_optional_scalar(request, "enable_cache"),
                cache_dir=_optional_scalar(request, "cache_dir"),
            )
            result = self._eval_service.evaluate_ragas(payload)
            return pb2.RagasEvalResponse(
                summary=pb2.RagasSummary(
                    ragas_score=_build_metric_summary(result.summary.ragas_score),
                    faithfulness=_build_metric_summary(result.summary.faithfulness),
                    answer_relevancy=_build_metric_summary(result.summary.answer_relevancy),
                    context_recall=_build_metric_summary(result.summary.context_recall),
                    context_precision=_build_metric_summary(result.summary.context_precision),
                    context_entity_recall=_build_metric_summary(result.summary.context_entity_recall),
                    noise_sensitivity_relevant=_build_metric_summary(result.summary.noise_sensitivity_relevant),
                    noise_sensitivity_irrelevant=_build_metric_summary(result.summary.noise_sensitivity_irrelevant),
                ),
                sample_count=result.sample_count,
                samples=_python_to_value(result.samples),
            )
        except (ValidationError, RAGBaseException, Exception) as exc:
            _abort_from_exception(context, exc)

    def RetrievalEval(
        self,
        request: pb2.RetrievalEvalRequest,
        context: grpc.ServicerContext,
    ) -> pb2.RetrievalEvalResponse:
        try:
            payload = RetrievalEvalRequest(
                test=_optional_value(request, "test"),
                cuts=list(request.cuts) or None,
                skip_empty_gold=_optional_scalar(request, "skip_empty_gold"),
            )
            result = self._eval_service.evaluate_retrieval(payload)
            return pb2.RetrievalEvalResponse(
                meta=_python_to_value(result.meta),
                aggregated=_python_to_value(result.aggregated),
                per_query=_python_to_value(result.per_query),
                diagnostics=_python_to_value(result.diagnostics),
            )
        except (ValidationError, RAGBaseException, Exception) as exc:
            _abort_from_exception(context, exc)


class ComponentEvalRpcServicer(pb2_grpc.ComponentEvalRpcServiceServicer):
    def __init__(self, component_eval_service: ComponentEvalService | None = None) -> None:
        self._component_eval_service = component_eval_service or ComponentEvalService()

    def ChunkQualityEval(
        self,
        request: pb2.ChunkQualityEvalRequest,
        context: grpc.ServicerContext,
    ) -> pb2.ChunkQualityEvalResponse:
        try:
            payload = ChunkQualityRequest(
                chunks=_optional_value(request, "chunks"),
                enable_semantic_similarity=_optional_scalar(request, "enable_semantic_similarity"),
                enable_boundary_clarity=_optional_scalar(request, "enable_boundary_clarity"),
                score_temperature=_optional_scalar(request, "score_temperature"),
                sim_model_path=_optional_scalar(request, "sim_model_path"),
                vllm_api_base=_optional_scalar(request, "vllm_api_base"),
                vllm_model_name=_optional_scalar(request, "vllm_model_name"),
            )
            result = self._component_eval_service.evaluate_chunk_quality(payload)
            return pb2.ChunkQualityEvalResponse(
                avg_semantic_dissimilarity=result.avg_semantic_dissimilarity,
                avg_boundary_clarity=result.avg_boundary_clarity,
                num_pairs=result.num_pairs,
                details=_build_chunk_pair_result(result.details),
            )
        except (ValidationError, RAGBaseException, Exception) as exc:
            _abort_from_exception(context, exc)

    def ChunkStickinessEval(
        self,
        request: pb2.ChunkStickinessEvalRequest,
        context: grpc.ServicerContext,
    ) -> pb2.ChunkStickinessEvalResponse:
        try:
            threshold = _optional_scalar(request, "threshold")
            delta = _optional_scalar(request, "delta")
            score_temperature = _optional_scalar(request, "score_temperature")
            payload = ChunkStickinessRequest(
                chunks=_optional_value(request, "chunks"),
                threshold=threshold if threshold is not None else _field_default(ChunkStickinessRequest, "threshold"),
                delta=delta if delta is not None else _field_default(ChunkStickinessRequest, "delta"),
                score_temperature=(
                    score_temperature
                    if score_temperature is not None
                    else _field_default(ChunkStickinessRequest, "score_temperature")
                ),
                vllm_api_base=_optional_scalar(request, "vllm_api_base"),
                vllm_model_name=_optional_scalar(request, "vllm_model_name"),
            )
            result = self._component_eval_service.evaluate_chunk_stickiness(payload)
            return pb2.ChunkStickinessEvalResponse(
                structural_entropy_complete=result.structural_entropy_complete,
                structural_entropy_incomplete=result.structural_entropy_incomplete,
                normalized_structural_entropy_complete=result.normalized_structural_entropy_complete,
                normalized_structural_entropy_incomplete=result.normalized_structural_entropy_incomplete,
                graph_complete=_python_to_value(result.graph_complete),
                graph_incomplete=_python_to_value(result.graph_incomplete),
            )
        except (ValidationError, RAGBaseException, Exception) as exc:
            _abort_from_exception(context, exc)
