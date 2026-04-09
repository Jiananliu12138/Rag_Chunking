"""Custom RAGAS prompts that prevent repetitive / degenerate LLM output."""

import typing as t

from ragas.prompt import StringIO
from ragas.testset.transforms.extractors.llm_based import (
    NERPrompt,
    SummaryExtractorPrompt,
    ThemesAndConceptsExtractorPrompt,
)


class StrictSummaryPrompt(SummaryExtractorPrompt):
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


class StrictNERPrompt(NERPrompt):
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


class StrictThemesPrompt(ThemesAndConceptsExtractorPrompt):
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
