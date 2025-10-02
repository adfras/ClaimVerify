#!/usr/bin/env bash
# Run OCRmyPDF inside the official container image.
# Usage: scripts/docker_ocrmypdf.sh INPUT.pdf OUTPUT.pdf [extra OCRmyPDF args]

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 INPUT.pdf OUTPUT.pdf [ocrmypdf options...]" >&2
    exit 1
fi

INPUT=$1
OUTPUT=$2
shift 2

# Resolve paths so Docker sees them correctly.
INPUT_PATH=$(realpath "$INPUT")
OUTPUT_PATH=$(realpath "$OUTPUT")
WORKDIR=$(dirname "$OUTPUT_PATH")

# Ensure output directory exists.
mkdir -p "$WORKDIR"

# Run OCRmyPDF inside the official container. Mount the working directory
# to /data so input/output paths are available.
docker run --rm \
    -v "${WORKDIR}:/data" \
    -v "${INPUT_PATH}:/input.pdf:ro" \
    ghcr.io/jbarlow83/ocrmypdf:latest \
    ocrmypdf "$@" /input.pdf "/data/$(basename "$OUTPUT_PATH")"
