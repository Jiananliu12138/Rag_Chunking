"""Idempotent monkey-patches for RAGAS testset generation error tolerance."""

import logging
import math
from typing import Any, Dict, List

from .prompts import MULTIHOP_QA_INSTRUCTION

_logger = logging.getLogger("testset_patches")


def _is_nan_like(value: Any) -> bool:
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def _safe_repr(value: Any, max_len: int = 1000) -> str:
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<repr failed: {type(exc).__name__}: {exc}>"
    if len(text) > max_len:
        return text[:max_len] + "...<truncated>"
    return text


def _build_failed_generation_record(
    index: int,
    sample: Any,
    additional_info: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "index": index,
        "type": type(sample).__name__,
        "repr": _safe_repr(sample),
        "reason": reason,
    }
    for key, value in additional_info.items():
        if key not in record:
            record[key] = value
    return record


def patch_transforms_error_handling() -> None:
    """Patch Extractor and RelationshipBuilder to skip failures gracefully."""
    from ragas.testset.transforms.base import Extractor, RelationshipBuilder

    ext_logger = logging.getLogger("ragas.testset.transforms.base")

    if not getattr(Extractor.generate_execution_plan, "_mc_fault_tolerant", False):

        def _fault_tolerant_generate_execution_plan(self, kg):
            async def safe_apply_extract(node):
                try:
                    property_name, property_value = await self.extract(node)
                    if node.get_property(property_name) is None:
                        node.add_property(property_name, property_value)
                    else:
                        ext_logger.warning(
                            "Property '%s' already exists in node '%.6s'. Skipping!",
                            property_name,
                            node.id,
                        )
                except Exception as exc:
                    ext_logger.warning(
                        "[%s] Extraction failed for node '%.6s', skipping. %s: %s",
                        self.__class__.__name__,
                        node.id,
                        type(exc).__name__,
                        str(exc)[:300],
                    )

            filtered = self.filter(kg)
            plan = [safe_apply_extract(node) for node in filtered.nodes]
            ext_logger.debug(
                "Created %d coroutines for %s (fault-tolerant)",
                len(plan),
                self.__class__.__name__,
            )
            return plan

        _fault_tolerant_generate_execution_plan._mc_fault_tolerant = True
        Extractor.generate_execution_plan = _fault_tolerant_generate_execution_plan
        ext_logger.info("Patched Extractor.generate_execution_plan for fault tolerance")

    if not getattr(RelationshipBuilder.generate_execution_plan, "_mc_fault_tolerant", False):

        def _fault_tolerant_rb_execution_plan(self, kg):
            async def safe_apply_build_relationships(filtered_kg, original_kg):
                try:
                    relationships = await self.transform(filtered_kg)
                    original_kg.relationships.extend(relationships)
                except Exception as exc:
                    ext_logger.warning(
                        "[%s] Relationship building failed, skipping. %s: %s",
                        self.__class__.__name__,
                        type(exc).__name__,
                        str(exc)[:300],
                    )

            filtered_kg = self.filter(kg)
            plan = [safe_apply_build_relationships(filtered_kg=filtered_kg, original_kg=kg)]
            ext_logger.debug(
                "Created %d coroutines for %s (fault-tolerant)",
                len(plan),
                self.__class__.__name__,
            )
            return plan

        _fault_tolerant_rb_execution_plan._mc_fault_tolerant = True
        RelationshipBuilder.generate_execution_plan = _fault_tolerant_rb_execution_plan
        ext_logger.info("Patched RelationshipBuilder.generate_execution_plan for fault tolerance")

    try:
        from ragas.testset.transforms.relationship_builders.traditional import (
            OverlapScoreBuilder,
            JaccardSimilarityBuilder,
        )
    except ImportError:
        return

    if not getattr(OverlapScoreBuilder, "_mc_patched_transform", False):

        async def _safe_overlap_transform(self, kg):
            from ragas.testset.graph import Relationship as _Rel

            distance_measure = self.distance_measure_map[self.distance_measure]
            noisy_items = self._get_noisy_items(kg.nodes, self.property_name)
            relationships = []
            skipped = 0
            for i, node_x in enumerate(kg.nodes):
                for j, node_y in enumerate(kg.nodes):
                    if i >= j:
                        continue
                    node_x_items = node_x.get_property(self.property_name)
                    node_y_items = node_y.get_property(self.property_name)
                    if node_x_items is None or node_y_items is None:
                        skipped += 1
                        continue
                    if self.key_name is not None:
                        node_x_items = node_x_items.get(self.key_name, [])
                        node_y_items = node_y_items.get(self.key_name, [])

                    overlaps = []
                    overlapped_items = []
                    for x in node_x_items:
                        if x not in noisy_items:
                            for y in node_y_items:
                                if y not in noisy_items:
                                    similarity = 1 - distance_measure.distance(
                                        x.lower(), y.lower()
                                    )
                                    verdict = similarity >= self.distance_threshold
                                    overlaps.append(verdict)
                                    if verdict:
                                        overlapped_items.append((x, y))

                    similarity = self._overlap_score(overlaps)
                    if similarity >= self.threshold:
                        relationships.append(
                            _Rel(
                                source=node_x,
                                target=node_y,
                                type=f"{self.property_name}_overlap",
                                properties={
                                    f"{self.property_name}_{self.new_property_name}": similarity,
                                    "overlapped_items": overlapped_items,
                                },
                                bidirectional=True,
                            )
                        )

            if skipped:
                ext_logger.info(
                    "OverlapScoreBuilder: skipped %d node pairs with missing '%s'",
                    skipped,
                    self.property_name,
                )
            return relationships

        OverlapScoreBuilder.transform = _safe_overlap_transform
        OverlapScoreBuilder._mc_patched_transform = True
        ext_logger.info("Patched OverlapScoreBuilder.transform to skip missing properties")

    if not getattr(JaccardSimilarityBuilder, "_mc_patched_find", False):

        def _safe_jaccard_find(self, kg):
            import itertools

            similar_pairs = set()
            skipped = 0
            for (i, node1), (j, node2) in itertools.combinations(enumerate(kg.nodes), 2):
                items1 = node1.get_property(self.property_name)
                items2 = node2.get_property(self.property_name)
                if items1 is None or items2 is None:
                    skipped += 1
                    continue
                if self.key_name is not None:
                    items1 = items1.get(self.key_name, [])
                    items2 = items2.get(self.key_name, [])
                similarity = self._jaccard_similarity(set(items1), set(items2))
                if similarity >= self.threshold:
                    similar_pairs.add((i, j, similarity))

            if skipped:
                ext_logger.info(
                    "JaccardSimilarityBuilder: skipped %d node pairs with missing '%s'",
                    skipped,
                    self.property_name,
                )
            return list(similar_pairs)

        JaccardSimilarityBuilder._find_similar_embedding_pairs = _safe_jaccard_find
        JaccardSimilarityBuilder._mc_patched_find = True
        ext_logger.info("Patched JaccardSimilarityBuilder to skip missing properties")


def patch_safe_generate() -> None:
    """Patch TestsetGenerator.generate to tolerate NaN / malformed samples."""
    import ragas.testset.synthesizers.generate as ragas_generate

    if getattr(ragas_generate.TestsetGenerator.generate, "_mc_safe_patched", False):
        return

    def _safe_generate(
        self,
        testset_size: int,
        query_distribution: Any = None,
        num_personas: int = 3,
        run_config: Any = None,
        batch_size: Any = None,
        callbacks: Any = None,
        token_usage_parser: Any = None,
        with_debugging_logs: bool = False,
        raise_exceptions: bool = True,
        return_executor: bool = False,
    ) -> Any:
        if run_config is not None and isinstance(self.llm, ragas_generate.BaseRagasLLM):
            self.llm.set_run_config(run_config)

        query_distribution = query_distribution or ragas_generate.default_query_distribution(
            self.llm,
            self.knowledge_graph,
            self.llm_context,
        )
        callbacks = callbacks or []
        ragas_callbacks = {}

        if token_usage_parser is not None:
            from ragas.cost import CostCallbackHandler
            cost_cb = CostCallbackHandler(token_usage_parser=token_usage_parser)
            ragas_callbacks["cost_cb"] = cost_cb
        else:
            cost_cb = None

        for cb in ragas_callbacks.values():
            if isinstance(callbacks, ragas_generate.BaseCallbackManager):
                callbacks.add_handler(cb)
            else:
                callbacks.append(cb)

        testset_generation_rm, testset_generation_grp = ragas_generate.new_group(
            name=ragas_generate.RAGAS_TESTSET_GENERATION_GROUP_NAME,
            inputs={"testset_size": testset_size},
            callbacks=callbacks,
        )

        if with_debugging_logs:
            from ragas.utils import patch_logger
            patch_logger("ragas.experimental.testset.synthesizers", logging.DEBUG)
            patch_logger("ragas.experimental.testset.graph", logging.DEBUG)
            patch_logger("ragas.experimental.testset.transforms", logging.DEBUG)

        if self.persona_list is None:
            self.persona_list = ragas_generate.generate_personas_from_kg(
                llm=self.llm,
                kg=self.knowledge_graph,
                num_personas=num_personas,
                callbacks=callbacks,
            )
        else:
            ragas_generate.random.shuffle(self.persona_list)

        splits, _ = ragas_generate.calculate_split_values(
            [prob for _, prob in query_distribution],
            testset_size,
        )
        scenario_generation_rm, scenario_generation_grp = ragas_generate.new_group(
            name="Scenario Generation",
            inputs={"splits": splits},
            callbacks=testset_generation_grp,
        )

        exec = ragas_generate.Executor(
            desc="Generating Scenarios",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=False,
            batch_size=batch_size,
        )

        splits, _ = ragas_generate.calculate_split_values(
            [prob for _, prob in query_distribution],
            testset_size,
        )
        for i, (scenario, _) in enumerate(query_distribution):
            exec.submit(
                scenario.generate_scenarios,
                n=splits[i],
                knowledge_graph=self.knowledge_graph,
                persona_list=self.persona_list[:num_personas],
                callbacks=scenario_generation_grp,
            )

        try:
            scenario_sample_list = exec.results()
        except Exception as e:
            scenario_generation_rm.on_chain_error(e)
            raise e
        else:
            scenario_generation_rm.on_chain_end(
                outputs={"scenario_sample_list": scenario_sample_list}
            )

        sample_generation_rm, sample_generation_grp = ragas_generate.new_group(
            name="Sample Generation",
            inputs={"scenario_sample_list": scenario_sample_list},
            callbacks=testset_generation_grp,
        )
        exec = ragas_generate.Executor(
            "Generating Samples",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=True,
            batch_size=batch_size,
        )
        additional_testset_info: List[Dict[str, Any]] = []
        for i, (synthesizer, _) in enumerate(query_distribution):
            for sample in scenario_sample_list[i]:
                exec.submit(
                    synthesizer.generate_sample,
                    scenario=sample,
                    callbacks=sample_generation_grp,
                )
                additional_testset_info.append(
                    {
                        "synthesizer_name": synthesizer.name,
                    }
                )

        if return_executor:
            self._mc_last_failed_rows = []
            return exec

        try:
            eval_samples = exec.results()
        except Exception as e:
            sample_generation_rm.on_chain_error(e)
            raise e
        else:
            sample_generation_rm.on_chain_end(outputs={"eval_samples": eval_samples})

        failed_rows: List[Dict[str, Any]] = []
        if len(eval_samples) != len(additional_testset_info):
            failed_rows.append(
                {
                    "index": -1,
                    "type": "length_mismatch",
                    "repr": "",
                    "reason": (
                        "eval_samples and additional_testset_info lengths differ: "
                        f"{len(eval_samples)} vs {len(additional_testset_info)}"
                    ),
                }
            )

        testsets = []
        for index, (sample, additional_info) in enumerate(
            zip(eval_samples, additional_testset_info)
        ):
            if _is_nan_like(sample):
                failed_rows.append(
                    _build_failed_generation_record(
                        index=index,
                        sample=sample,
                        additional_info=additional_info,
                        reason="eval_sample is NaN",
                    )
                )
                continue

            try:
                testsets.append(
                    ragas_generate.TestsetSample(
                        eval_sample=sample,
                        **additional_info,
                    )
                )
            except Exception as exc:
                failed_rows.append(
                    _build_failed_generation_record(
                        index=index,
                        sample=sample,
                        additional_info=additional_info,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )

        self._mc_last_failed_rows = failed_rows

        testset = ragas_generate.Testset(samples=testsets, cost_cb=cost_cb)
        testset_generation_rm.on_chain_end({"testset": testset})
        ragas_generate.track(
            ragas_generate.TestsetGenerationEvent(
                event_type="testset_generation",
                evolution_names=[
                    e.__class__.__name__.lower() for e, _ in query_distribution
                ],
                evolution_percentages=[p for _, p in query_distribution],
                num_rows=len(testsets),
                language="english",
            )
        )
        return testset

    _safe_generate._mc_safe_patched = True
    ragas_generate.TestsetGenerator.generate = _safe_generate


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


def apply_all_patches() -> None:
    """Apply all patches (idempotent)."""
    patch_transforms_error_handling()
    patch_multihop_prompt()
    patch_safe_generate()
