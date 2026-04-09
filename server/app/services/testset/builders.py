"""Factories: LLM, embedding, transforms, query distribution."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

from openai import AsyncOpenAI
from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.testset.transforms import (
    CosineSimilarityBuilder,
    EmbeddingExtractor,
    OverlapScoreBuilder,
    Parallel,
    SummaryExtractor,
)
from ragas.testset.transforms.extractors.llm_based import (
    NERExtractor,
    ThemesExtractor,
)

from .prompts import StrictNERPrompt, StrictSummaryPrompt, StrictThemesPrompt
from .synthesizers import ArticleSummarySynthesizer

QueryDistribution = List[Tuple[object, float]]


def build_llm(
    *,
    base_url: str,
    api_key: str = "EMPTY",
    model: str,
    provider: str = "openai",
    adapter: str = "auto",
    max_tokens: int = 2048,
    timeout: int = 3600,
):
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    kwargs = {}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return llm_factory(
        model=model,
        provider=provider,
        client=client,
        adapter=adapter,
        **kwargs,
    )


def build_embedding_model(
    *,
    model_path: str,
    device: str = "cuda",
):
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Local embedding model path not found: {path.resolve()}")
    return HuggingFaceEmbeddings(
        model=str(path),
        device=device,
        normalize_embeddings=True,
    )


def build_transforms(llm, embedding_model):
    def filter_chunks(node):
        return node.type.name == "CHUNK"

    summary_extractor = SummaryExtractor(
        llm=llm,
        prompt=StrictSummaryPrompt(),
        filter_nodes=filter_chunks,
    )
    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=embedding_model,
        property_name="summary_embedding",
        embed_property_name="summary",
        filter_nodes=filter_chunks,
    )
    theme_extractor = ThemesExtractor(
        llm=llm,
        prompt=StrictThemesPrompt(),
        max_num_themes=10,
        filter_nodes=filter_chunks,
    )
    ner_extractor = NERExtractor(
        llm=llm,
        prompt=StrictNERPrompt(),
        max_num_entities=10,
        filter_nodes=filter_chunks,
    )
    cosine_sim_builder = CosineSimilarityBuilder(
        property_name="summary_embedding",
        new_property_name="summary_similarity",
        threshold=0.7,
        filter_nodes=filter_chunks,
    )
    ner_overlap_sim = OverlapScoreBuilder(threshold=0.01, filter_nodes=filter_chunks)

    return [
        summary_extractor,
        Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
        Parallel(cosine_sim_builder, ner_overlap_sim),
    ]


def build_query_distribution(llm) -> Union[QueryDistribution, None]:
    from ragas.testset.synthesizers.single_hop.specific import (
        SingleHopSpecificQuerySynthesizer,
    )
    from ragas.testset.synthesizers.multi_hop import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
    )

    return [
        (SingleHopSpecificQuerySynthesizer(llm=llm), 0.3),
        (MultiHopAbstractQuerySynthesizer(llm=llm), 0.3),
        (MultiHopSpecificQuerySynthesizer(llm=llm), 0.2),
        (ArticleSummarySynthesizer(llm=llm), 0.2),
    ]
