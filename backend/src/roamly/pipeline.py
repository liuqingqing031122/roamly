"""Orchestration of the extraction steps: prompt, LLM call or cache, post-processing."""

import logging

from roamly.extraction import ExtractedClaims, ExtractionResult, process_extracted_claims
from roamly.extraction_cache import ExtractionCache
from roamly.extraction_prompt import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_prompt
from roamly.llm import LLMClient

logger = logging.getLogger(__name__)


def extract_claims(
    source_text: str, client: LLMClient, cache: ExtractionCache | None = None
) -> ExtractionResult:
    """Extract and validate claims from guide text.

    Only successfully validated LLM output is cached; LLM errors propagate and are
    never stored. Post-processing runs on both cache hits and fresh responses.
    """
    extracted = cache.get(source_text) if cache else None
    if extracted is not None:
        logger.info("Extraction cache hit")
    else:
        response = client.complete_structured(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=build_extraction_user_prompt(source_text),
            response_model=ExtractedClaims,
        )
        extracted = response.output
        logger.info(
            "LLM call used %d input and %d output tokens",
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        if cache:
            try:
                cache.put(source_text, extracted)
            except OSError as exc:
                logger.warning("Could not write extraction cache: %s", exc)

    return process_extracted_claims(source_text, extracted)
