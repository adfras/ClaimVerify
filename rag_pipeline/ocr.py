from __future__ import annotations

import logging
import shlex
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Sequence


LOGGER = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=4)
def resolve_ocr_executable(spec: str | None) -> Sequence[str] | None:
    """Return the base OCR command as a list suitable for subprocess."""
    if not spec:
        return None

    spec = spec.strip()
    if not spec or spec.lower() in {"none", "disable", "disabled"}:
        return None

    if spec.lower() == "auto":
        direct = shutil.which("ocrmypdf")
        if direct:
            return [direct]
        docker = shutil.which("docker")
        script = _project_root() / "scripts" / "docker_ocrmypdf.sh"
        if docker and script.exists():
            return [str(script)]
        LOGGER.warning(
            "No OCR backend available (neither ocrmypdf nor Docker). Disabling OCR."
        )
        return None

    try:
        parts = shlex.split(spec)
    except ValueError:
        LOGGER.warning("Invalid OCR command specification %r; disabling OCR.", spec)
        return None

    if not parts:
        return None

    first = parts[0]
    candidate_path = Path(first)
    if not candidate_path.is_absolute():
        repo_candidate = _project_root() / candidate_path
        if repo_candidate.exists():
            parts[0] = str(repo_candidate.resolve())
            return parts

    if candidate_path.exists():
        parts[0] = str(candidate_path.resolve())
        return parts

    resolved = shutil.which(first)
    if resolved:
        parts[0] = resolved
        return parts

    LOGGER.warning("OCR command %s not found on PATH; disabling OCR.", first)
    return None


def build_ocr_command(
    spec: str | None, input_path: Path, output_path: Path
) -> Sequence[str] | None:
    base = resolve_ocr_executable(spec)
    if not base:
        return None
    return [
        *base,
        "--skip-text",
        "--deskew",
        "--clean",
        str(input_path),
        str(output_path),
    ]

