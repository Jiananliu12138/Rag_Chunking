"""Custom RAGAS synthesizer: article-level summarization questions."""

import random
import typing as t
from collections import defaultdict
from dataclasses import dataclass

from pydantic import BaseModel
from ragas.dataset_schema import SingleTurnSample
from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.base import (
    BaseScenario,
    BaseSynthesizer,
    QueryLength,
    QueryStyle,
    Scenario,
)


def _get_article_id(node: Node) -> t.Optional[t.Any]:
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
