"""
Testset generation configuration: LLM / embedding / transform / prompt / distribution.

All customization points are in this file. The main script
(run_generate_with_chunks.py) imports and uses them directly.
"""

import logging
import random
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

from openai import AsyncOpenAI
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.prompt import PydanticPrompt, StringIO
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.base import (
    BaseScenario,
    BaseSynthesizer,
    QueryLength,
    QueryStyle,
    Scenario,
)
from ragas.testset.transforms import (
    CosineSimilarityBuilder,
    EmbeddingExtractor,
    OverlapScoreBuilder,
    Parallel,
    SummaryExtractor,
)
from ragas.testset.transforms.extractors.llm_based import (
    NERExtractor,
    NERPrompt,
    SummaryExtractorPrompt,
    ThemesAndConceptsExtractorPrompt,
    ThemesExtractor,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static config
# ---------------------------------------------------------------------------

LLM_BASE_URL = "http://127.0.0.1:8001/v1"
LLM_API_KEY = "EMPTY"
LLM_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
LLM_PROVIDER = "openai"
LLM_ADAPTER = "auto"
LLM_TIMEOUT_SECONDS = 120
LLM_MAX_TOKENS = 4096

EMBEDDING_MODEL_PATH = Path(r"/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5")
EMBEDDING_DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Custom prompts  (override ragas defaults to prevent repetitive output)
# ---------------------------------------------------------------------------

class _StrictSummaryPrompt(SummaryExtractorPrompt):
    instruction: str = (
        "Summarize the given text in at most 3 sentences.\n"
        "Rules:\n"
        "- Capture only the most important points. Be concise.\n"
        "- Do NOT repeat sentences or paraphrase the same idea multiple times.\n"
        "- If the text is very short, a single sentence is acceptable.\n"
        "- Output ONLY the summary text, nothing else.\n"
        "- Your entire output must be under 200 tokens."
    )
    examples: t.List[t.Tuple[StringIO, StringIO]] = [
        (
            StringIO(
                text=(
                    "Artificial intelligence\n\n"
                    "Artificial intelligence is transforming various industries by "
                    "automating tasks that previously required human intelligence. "
                    "From healthcare to finance, AI is being used to analyze vast "
                    "amounts of data quickly and accurately. This technology is also "
                    "driving innovations in areas like self-driving cars and "
                    "personalized recommendations."
                )
            ),
            StringIO(
                text=(
                    "AI is revolutionizing industries by automating tasks, analyzing "
                    "data, and driving innovations like self-driving cars and "
                    "personalized recommendations."
                )
            ),
        ),
        (
            StringIO(
                text=(
                    "The Apollo program was a series of space missions run by NASA "
                    "between 1961 and 1972. Its primary goal was to land humans on "
                    "the Moon and return them safely to Earth. Apollo 11, launched on "
                    "July 16, 1969, was the first mission to achieve this goal when "
                    "astronauts Neil Armstrong and Buzz Aldrin walked on the lunar "
                    "surface. The program involved extensive development of spacecraft, "
                    "including the Saturn V rocket, the Command Module, and the Lunar "
                    "Module. Over the course of the program, twelve astronauts walked "
                    "on the Moon across six successful landing missions. The Apollo "
                    "program had lasting impacts on science, technology, and "
                    "international space policy."
                )
            ),
            StringIO(
                text=(
                    "The Apollo program (1961-1972) was NASA's effort to land humans "
                    "on the Moon. Apollo 11 achieved the first lunar landing in 1969. "
                    "Twelve astronauts walked on the Moon across six missions, leaving "
                    "a lasting impact on science and space policy."
                )
            ),
        ),
    ]


class _StrictNERPrompt(NERPrompt):
    instruction: str = (
        "Extract the most important named entities from the given text.\n"
        "Rules:\n"
        "- Return AT MOST max_num entities. Fewer is fine if the text has fewer.\n"
        "- Each entity must be UNIQUE — never repeat the same entity or a trivial variant.\n"
        "- Only include proper nouns, specific terms, or clearly defined concepts.\n"
        "- Do NOT pad the list with generic words, descriptions, or rephrased duplicates.\n"
        "- Keep each entity name short (1-5 words).\n"
        "- Your entire output must be under 300 tokens.\n"
        "\n"
        "BAD output (duplicates — NEVER do this):\n"
        '  {"entities": ["taxable income", "taxable income", "taxable income"]}\n'
        "GOOD output (unique, concise):\n"
        '  {"entities": ["taxable income", "IRS", "Form 1040"]}'
    )


class _StrictThemesPrompt(ThemesAndConceptsExtractorPrompt):
    instruction: str = (
        "Extract the main themes and concepts from the given text.\n"
        "Rules:\n"
        "- Return AT MOST max_num themes. Fewer is fine if the text covers fewer topics.\n"
        "- Each theme must be UNIQUE — do NOT repeat the same theme in different wording.\n"
        "- Use short, specific phrases (1-5 words each).\n"
        "- Do NOT pad the list with vague or overlapping terms.\n"
        "- Your entire output must be under 300 tokens.\n"
        "\n"
        "BAD output (overlapping — NEVER do this):\n"
        '  {"output": ["machine learning", "ML techniques", "machine learning methods"]}\n'
        "GOOD output (distinct, specific):\n"
        '  {"output": ["machine learning", "neural networks", "data preprocessing"]}'
    )


SUMMARY_PROMPT = _StrictSummaryPrompt()
NER_PROMPT = _StrictNERPrompt()
THEMES_PROMPT = _StrictThemesPrompt()

MULTIHOP_QA_INSTRUCTION = (
    "Generate a multi-hop query and answer based on the specified conditions "
    "(persona, themes, style, length) and the provided context. "
    "The themes represent phrases extracted or generated from the context, "
    "highlighting the suitability of the selected context for multi-hop query creation. "
    "Ensure the query explicitly incorporates these themes.\n"
    "### Instructions:\n"
    "1. **Generate a Multi-Hop Query**: Use the provided context segments and themes "
    "to form a query that requires combining information from ALL provided segments "
    "(e.g., `<1-hop>`, `<2-hop>`, `<3-hop>`, etc.). The query MUST require information "
    "from EVERY segment to be fully answered — not just some of them.\n"
    "2. **Generate an Answer**: Use only the content from the provided context to create "
    "a detailed and faithful answer. The answer MUST reference information from ALL "
    "provided context segments. Do not add information not present in the context.\n"
    "3. **Multi-Hop Context Tags**:\n"
    "   - Each context segment is tagged as `<1-hop>`, `<2-hop>`, etc.\n"
    "   - The query MUST use information from ALL tagged segments and connect them "
    "meaningfully.\n"
    "   - If 3 segments are provided, all 3 must contribute to answering the query.\n"
    "4. **Additional Context** (if provided): If llm_context is provided, use it as "
    "guidance for what type of question to generate and how to structure the answer. "
    "Still ensure the content comes only from the provided context.\n"
    "5. Your entire response must be under 500 tokens."
)

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def build_llm():
    client = AsyncOpenAI(
        api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT_SECONDS
    )
    llm_kwargs = {}
    if LLM_MAX_TOKENS is not None:
        llm_kwargs["max_tokens"] = LLM_MAX_TOKENS
    return llm_factory(
        model=LLM_MODEL,
        provider=LLM_PROVIDER,
        client=client,
        adapter=LLM_ADAPTER,
        **llm_kwargs,
    )


def build_embedding_model():
    model_path = EMBEDDING_MODEL_PATH.expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local embedding model path not found: {model_path.resolve()}"
        )
    return HuggingFaceEmbeddings(
        model=str(model_path),
        device=EMBEDDING_DEVICE,
        normalize_embeddings=True,
    )


def build_transforms(llm, embedding_model):
    def filter_chunks(node):
        return node.type.name == "CHUNK"

    summary_extractor = SummaryExtractor(
        llm=llm,
        prompt=SUMMARY_PROMPT,
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
        prompt=THEMES_PROMPT,
        max_num_themes=10,
        filter_nodes=filter_chunks,
    )
    ner_extractor = NERExtractor(
        llm=llm,
        prompt=NER_PROMPT,
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


# ---------------------------------------------------------------------------
# Custom Synthesizer: article-level summarization questions
# ---------------------------------------------------------------------------


def _get_article_id(node: Node) -> t.Optional[t.Any]:
    """Extract source_article_id from a CHUNK node's metadata."""
    meta = node.get_property("document_metadata")
    if isinstance(meta, dict):
        return meta.get("source_article_id")
    return None


class ArticleSummaryCondition(BaseModel):
    persona: Persona
    query_style: str
    query_length: str
    contexts: t.List[str]
    num_chunks: int


class GeneratedQueryAnswer(BaseModel):
    query: str
    answer: str


class ArticleSummaryQAPrompt(
    PydanticPrompt[ArticleSummaryCondition, GeneratedQueryAnswer]
):
    instruction: str = (
        "You are given multiple text chunks that ALL belong to the SAME article. "
        "Generate a summarization or overview question that can only be answered "
        "by reading across ALL the provided chunks, and provide a comprehensive answer.\n"
        "### Instructions:\n"
        "1. **Generate a Query**: Create a question that asks about the overall "
        "topic, main argument, key findings, or comprehensive summary of the article. "
        "The question should require synthesizing information from ALL chunks.\n"
        "2. **Generate an Answer**: Use ONLY the content from the provided chunks. "
        "The answer must cover key points from EVERY chunk.\n"
        "3. Each chunk is labeled [Chunk 1], [Chunk 2], etc.\n"
        "4. Your entire response must be under 500 tokens."
    )
    input_model: t.Type[ArticleSummaryCondition] = ArticleSummaryCondition
    output_model: t.Type[GeneratedQueryAnswer] = GeneratedQueryAnswer
    examples: t.List[t.Tuple[ArticleSummaryCondition, GeneratedQueryAnswer]] = [
        (
            ArticleSummaryCondition(
                persona=Persona(
                    name="Researcher",
                    role_description="Interested in understanding full research papers.",
                ),
                query_style="Formal",
                query_length="Medium",
                contexts=[
                    "[Chunk 1] The study examines the effects of climate change on coral reefs in the Pacific Ocean.",
                    "[Chunk 2] Rising sea temperatures have led to widespread coral bleaching events since 2015.",
                    "[Chunk 3] The authors propose marine protected areas as a mitigation strategy.",
                ],
                num_chunks=3,
            ),
            GeneratedQueryAnswer(
                query="What are the main findings and proposed solutions in this study on climate change and coral reefs?",
                answer=(
                    "The study examines climate change impacts on Pacific coral reefs, "
                    "finding that rising sea temperatures have caused widespread bleaching "
                    "since 2015. The authors propose establishing marine protected areas "
                    "as a key mitigation strategy."
                ),
            ),
        )
    ]


class ArticleSummaryScenario(BaseScenario):
    article_id: t.Any


@dataclass
class ArticleSummarySynthesizer(BaseSynthesizer):
    """Generate summarization questions from chunks belonging to the same article.

    Groups CHUNK nodes by ``source_article_id`` in their ``document_metadata``,
    then asks the LLM to produce overview questions that require reading all
    chunks of an article.
    """

    name: str = "article_summary_query_synthesizer"
    generate_prompt: PydanticPrompt = ArticleSummaryQAPrompt()
    min_chunks_per_article: int = 2
    max_chunks_per_article: int = 10

    def get_node_clusters(
        self, knowledge_graph: KnowledgeGraph
    ) -> t.Dict[t.Any, t.List[Node]]:
        groups: t.Dict[t.Any, t.List[Node]] = defaultdict(list)
        for node in knowledge_graph.nodes:
            if node.type.name != "CHUNK":
                continue
            aid = _get_article_id(node)
            if aid is not None:
                groups[aid].append(node)
        return {
            aid: nodes
            for aid, nodes in groups.items()
            if len(nodes) >= self.min_chunks_per_article
        }

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: t.List[Persona],
        callbacks: t.Any,
    ) -> t.List[ArticleSummaryScenario]:
        clusters = self.get_node_clusters(knowledge_graph)
        if not clusters:
            raise ValueError(
                "No articles with enough chunks found. "
                "Check that document_metadata contains source_article_id."
            )

        article_ids = list(clusters.keys())
        random.shuffle(article_ids)
        scenarios: t.List[ArticleSummaryScenario] = []
        styles = list(QueryStyle)
        lengths = list(QueryLength)

        for aid in article_ids:
            if len(scenarios) >= n:
                break
            nodes = clusters[aid][: self.max_chunks_per_article]
            persona = random.choice(persona_list) if persona_list else Persona(
                name="General Reader",
                role_description="Wants to understand the full article.",
            )
            scenarios.append(
                ArticleSummaryScenario(
                    nodes=nodes,
                    article_id=aid,
                    style=random.choice(styles),
                    length=random.choice(lengths),
                    persona=persona,
                )
            )

        return scenarios

    async def _generate_sample(
        self, scenario: Scenario, callbacks: t.Any
    ) -> SingleTurnSample:
        if not isinstance(scenario, ArticleSummaryScenario):
            raise TypeError("Expected ArticleSummaryScenario")

        contexts = []
        raw_contexts = []
        for i, node in enumerate(scenario.nodes):
            text = node.get_property("page_content") or ""
            contexts.append(f"[Chunk {i + 1}] {text}")
            raw_contexts.append(text)

        prompt_input = ArticleSummaryCondition(
            persona=scenario.persona,
            query_style=scenario.style.value,
            query_length=scenario.length.value,
            contexts=contexts,
            num_chunks=len(contexts),
        )
        response = await self.generate_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        return SingleTurnSample(
            user_input=response.query,
            reference=response.answer,
            reference_contexts=raw_contexts,
        )


# ---------------------------------------------------------------------------
# Query distribution
# ---------------------------------------------------------------------------

QueryDistribution = List[Tuple[object, float]]


def build_query_distribution(llm) -> Union[QueryDistribution, None]:
    """Build custom query distribution.

    Returns a list of ``(synthesizer, probability)`` tuples, or ``None``
    to let ragas use its (patched) default distribution.
    """
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


# ---------------------------------------------------------------------------
# Monkey-patch: strengthen multi-hop prompt
# ---------------------------------------------------------------------------


def patch_multihop_prompt() -> None:
    """Replace multi-hop QA instruction to require ALL context segments."""
    from ragas.testset.synthesizers.multi_hop.base import MultiHopQuerySynthesizer
    import ragas.testset.synthesizers as synth_pkg
    import ragas.testset.synthesizers.generate as gen_module

    if getattr(synth_pkg, "_mc_multihop_patched", False):
        return

    _orig_default_qd = gen_module.default_query_distribution

    def _patched_default_qd(llm, kg=None, llm_context=None):
        distribution = _orig_default_qd(llm, kg, llm_context)
        for synthesizer, _ in distribution:
            if isinstance(synthesizer, MultiHopQuerySynthesizer):
                synthesizer.generate_query_reference_prompt.instruction = (
                    MULTIHOP_QA_INSTRUCTION
                )
        return distribution

    gen_module.default_query_distribution = _patched_default_qd
    synth_pkg.default_query_distribution = _patched_default_qd
    synth_pkg._mc_multihop_patched = True
