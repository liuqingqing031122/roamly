"""On-disk cache of validated LLM extraction output.

The cache stores `ExtractedClaims` (the validated LLM response), not the final
`ExtractionResult`, so post-processing always runs on cache hits. Entries store a
hash of the source text rather than the full text, but cached claims still contain
verbatim excerpts from it.
"""

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from roamly.extraction import ExtractedClaims
from roamly.extraction_prompt import EXTRACTION_PROMPT_VERSION, EXTRACTION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Bump when the layout of cache entries changes.
CACHE_FORMAT_VERSION = 1


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ExtractionCache:
    """Stores one JSON file per cache key in `directory`."""

    def __init__(
        self,
        directory: Path,
        model_settings: Mapping[str, Any],
        *,
        prompt_version: str = EXTRACTION_PROMPT_VERSION,
        system_prompt: str = EXTRACTION_SYSTEM_PROMPT,
    ) -> None:
        self.directory = directory
        self.model_settings = dict(model_settings)
        self.prompt_version = prompt_version
        self.system_prompt = system_prompt

    def key_fields(self, source_text: str) -> dict[str, Any]:
        """Everything that can change the extraction output for this source text."""
        schema = json.dumps(ExtractedClaims.model_json_schema(), sort_keys=True)
        return {
            "cache_format": CACHE_FORMAT_VERSION,
            "prompt_version": self.prompt_version,
            # Guards against prompt edits made without a version bump.
            "system_prompt_sha256": _sha256(self.system_prompt),
            "response_schema_sha256": _sha256(schema),
            "model_settings": self.model_settings,
            "source_text_sha256": _sha256(source_text),
        }

    def key(self, source_text: str) -> str:
        return _sha256(json.dumps(self.key_fields(source_text), sort_keys=True))

    def path(self, source_text: str) -> Path:
        return self.directory / f"{self.key(source_text)}.json"

    def get(self, source_text: str) -> ExtractedClaims | None:
        """Return the cached output, or None on a miss or an unreadable entry."""
        path = self.path(source_text)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            return ExtractedClaims.model_validate(entry["output"])
        except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
            logger.warning("Ignoring unreadable cache entry %s: %s", path.name, exc)
            return None

    def put(self, source_text: str, output: ExtractedClaims) -> None:
        """Store a validated output; the write is atomic so readers never see partial files."""
        entry = {
            "key_fields": self.key_fields(source_text),
            "created_at": datetime.now(UTC).isoformat(),
            "output": output.model_dump(mode="json"),
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(entry, file, indent=2, ensure_ascii=False)
            os.replace(tmp_name, self.path(source_text))
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
