from __future__ import annotations

import json
import re
from dataclasses import dataclass
from threading import Lock
from typing import Any

import requests
from tqdm.auto import tqdm

from public_method.evaluation._parallel import parallel_map


PROMPT_VERSION = "answer_equivalence_v1"

_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = Lock()

_SYSTEM_PROMPT = """You are a strict but fair QA judge.

You must decide whether the candidate answer is equivalent to at least one reference answer for the same question.

Return score=1 only when the candidate answer is fully correct for the question and semantically equivalent to at least one reference answer.

Accept:
- paraphrases
- different word order
- harmless extra detail that does not change the answer

Return score=0 when:
- the candidate contradicts the reference
- the candidate misses a key fact needed to answer the question
- the candidate adds a conflicting claim
- numbers, dates, entities, comparisons, negation, or causal relations differ
- the candidate is too vague to verify
- you are unsure

Keep reason very short, ideally under 30 words.

Output JSON only:
{"score": 0 or 1, "reason": "short explanation"}
"""


@dataclass(frozen=True)
class JudgeSample:
    row_index: int
    question_id: str | int | None
    record_id: str | int | None
    question: str | None
    prediction: str
    references: list[str]


def _normalize_api_base(api_base: str) -> str:
    return str(api_base).rstrip("/")


def _extract_message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"Judge response is missing choices: {data}")

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        if text_parts:
            return "".join(text_parts).strip()

    raise ValueError(f"Judge returned unsupported content: {data}")


def _extract_first_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Judge returned empty content")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError(f"Judge content does not contain JSON: {text}")

    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                break

    raise ValueError(f"Judge content does not contain a valid JSON object: {text}")


def _extract_partial_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    score_match = re.search(r'"score"\s*:\s*(true|false|0|1)', text, flags=re.IGNORECASE)
    if not score_match:
        return None

    score_raw = score_match.group(1)
    reason = None
    reason_match = re.search(r'"reason"\s*:\s*"', text, flags=re.IGNORECASE)
    if reason_match:
        reason = text[reason_match.end() :].strip()
        if reason.endswith("}"):
            reason = reason[:-1].rstrip()
        if reason.endswith('"'):
            reason = reason[:-1].rstrip()
        reason = reason.replace('\\"', '"').strip()

    return {
        "score": score_raw,
        "reason": reason or "Partial JSON recovered from truncated judge output.",
    }


def _build_user_prompt(sample: JudgeSample) -> str:
    payload = {
        "question": sample.question,
        "candidate_answer": sample.prediction,
        "reference_answers": sample.references,
        "decision_rule": (
            "Return score 1 only if the candidate answer is fully correct for the question "
            "and semantically equivalent to at least one reference answer. Otherwise return 0."
        ),
    }
    return (
        "Evaluate the candidate answer against the reference answers.\n"
        "Use the question only for disambiguation.\n"
        "Return JSON only.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _normalize_reason(value: Any) -> str:
    reason = str(value or "").strip()
    if reason:
        return reason
    return "No reason provided."


def _normalize_score(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "correct", "match"}:
            return 1
        if lowered in {"0", "false", "no", "incorrect", "mismatch"}:
            return 0
    raise ValueError(f"Unsupported judge score: {value}")


def _cache_key(
    *,
    api_base: str,
    model_name: str,
    sample: JudgeSample,
) -> str:
    return json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "api_base": api_base,
            "model_name": model_name,
            "question": sample.question,
            "prediction": sample.prediction,
            "references": sample.references,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _post_chat_completion(
    *,
    api_base: str,
    api_key: str | None,
    model_name: str,
    user_prompt: str,
    use_json_mode: bool,
    timeout: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key and str(api_key).strip():
        headers["Authorization"] = f"Bearer {str(api_key).strip()}"

    response = requests.post(
        f"{_normalize_api_base(api_base)}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Judge returned non-object payload: {data}")
    return data


def _judge_one_sample(
    *,
    sample: JudgeSample,
    api_base: str,
    api_key: str | None,
    model_name: str,
    timeout: int,
) -> dict[str, Any]:
    if not sample.references:
        return {
            "row_index": sample.row_index,
            "question_id": sample.question_id,
            "record_id": sample.record_id,
            "question": sample.question,
            "score": 0,
            "matched": False,
            "reason": "No reference answer was provided.",
        }

    cache_key = _cache_key(
        api_base=_normalize_api_base(api_base),
        model_name=model_name,
        sample=sample,
    )
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return dict(cached)

    user_prompt = _build_user_prompt(sample)
    last_error: Exception | None = None
    response_text = ""

    for use_json_mode in (True, False):
        try:
            raw = _post_chat_completion(
                api_base=api_base,
                api_key=api_key,
                model_name=model_name,
                user_prompt=user_prompt,
                use_json_mode=use_json_mode,
                timeout=timeout,
            )
            response_text = _extract_message_content(raw)
            try:
                parsed = _extract_first_json_object(response_text)
            except ValueError:
                parsed = _extract_partial_json_object(response_text)
                if parsed is None:
                    raise
            score = _normalize_score(parsed.get("score"))
            result = {
                "row_index": sample.row_index,
                "question_id": sample.question_id,
                "record_id": sample.record_id,
                "question": sample.question,
                "score": score,
                "matched": bool(score),
                "reason": _normalize_reason(parsed.get("reason")),
            }
            with _CACHE_LOCK:
                _CACHE[cache_key] = dict(result)
            return result
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "LLM judge failed to return valid JSON output. "
        f"Last error: {last_error}. Raw response: {response_text}"
    )


def evaluate_answer_equivalence(
    *,
    samples: list[JudgeSample],
    api_base: str,
    api_key: str | None,
    model_name: str,
    timeout: int = 600,
    show_progress: bool = True,
    max_workers: int = 16,
) -> dict[str, Any]:
    if not api_base or not str(api_base).strip():
        raise ValueError("api_base is required for LLM judge evaluation")
    if not model_name or not str(model_name).strip():
        raise ValueError("model_name is required for LLM judge evaluation")

    def _run(sample: JudgeSample) -> dict[str, Any]:
        return _judge_one_sample(
            sample=sample,
            api_base=api_base,
            api_key=api_key,
            model_name=model_name,
            timeout=timeout,
        )

    details = parallel_map(
        _run,
        samples,
        max_workers=max_workers,
        desc="LLM judge",
        show_progress=show_progress,
    )

    correct_count = sum(int(item["score"]) for item in details)
    evaluated_count = len(details)
    incorrect_count = max(evaluated_count - correct_count, 0)
    success_rate = (correct_count / evaluated_count) if evaluated_count else 0.0

    return {
        "llm_judge_success_rate": success_rate,
        "llm_judge_correct_count": correct_count,
        "llm_judge_incorrect_count": incorrect_count,
        "llm_judge_model": str(model_name).strip(),
        "llm_judge_prompt_version": PROMPT_VERSION,
        "judge_details": details,
    }
