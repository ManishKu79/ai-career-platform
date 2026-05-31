#!/usr/bin/env bash
# scripts/docker_build.sh
# Builds all Docker images for the AI Career Platform
# Usage: ./scripts/docker_build.sh [--no-cache] [--tag VERSION]

set -euo pipefail
# set -e: exit on error
# set -u: treat unset variables as errors
# set -o pipefail: pipe fails if any command fails

# ── Colors for output ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Configuration ──────────────────────────────────────────────────────
PROJECT_NAME="career-platform"
VERSION="${VERSION:-1.0.0}"
NO_CACHE=""

# ── Parse arguments ────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --no-cache)
      NO_CACHE="--no-cache"
      echo -e "${YELLOW}Building without cache...${NC}"
      ;;
    --tag)
      VERSION="$2"
      shift
      ;;
  esac
done

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}AI Career Platform — Docker Build${NC}"
echo -e "${BLUE}Version: ${VERSION}${NC}"
echo -e "${BLUE}================================${NC}"

# ── Verify prerequisites ───────────────────────────────────────────────
echo -e "\n${YELLOW}[1/4] Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
  echo -e "${RED}ERROR: Docker not installed${NC}"
  exit 1
fi

if ! docker info &> /dev/null; then
  echo -e "${RED}ERROR: Docker daemon not running${NC}"
  exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}"

# Verify required files exist
for file in requirements.txt docker/Dockerfile.api docker/Dockerfile.streamlit; do
  if [ ! -f "$file" ]; then
    echo -e "${RED}ERROR: Required file not found: $file${NC}"
    exit 1
  fi
done

echo -e "${GREEN}✓ Required files present${NC}"

# ── Build API image ────────────────────────────────────────────────────
echo -e "\n${YELLOW}[2/4] Building FastAPI image...${NC}"
echo "    Context: . (project root)"
echo "    Dockerfile: docker/Dockerfile.api"

BUILD_START=$(date +%s)

docker build \
  $NO_CACHE \
  --file docker/Dockerfile.api \
  --tag "${PROJECT_NAME}-api:${VERSION}" \
  --tag "${PROJECT_NAME}-api:latest" \
  --label "version=${VERSION}" \
  --label "build-date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-arg PYTHON_VERSION=3.11 \
  . # Build context = project root

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

echo -e "${GREEN}✓ API image built in ${BUILD_TIME}s${NC}"
docker image inspect "${PROJECT_NAME}-api:${VERSION}" \
  --format "    Size: {{.Size | printf \"%.0f\"}} bytes" 2>/dev/null || true

# ── Build Streamlit image ──────────────────────────────────────────────
echo -e "\n${YELLOW}[3/4] Building Streamlit image...${NC}"

BUILD_START=$(date +%s)

docker build \
  $NO_CACHE \
  --file docker/Dockerfile.streamlit \
  --tag "${PROJECT_NAME}-streamlit:${VERSION}" \
  --tag "${PROJECT_NAME}-streamlit:latest" \
  --label "version=${VERSION}" \
  --label "build-date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  .

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

echo -e "${GREEN}✓ Streamlit image built in ${BUILD_TIME}s${NC}"

# ── Summary ────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[4/4] Build Summary${NC}"
echo -e "${BLUE}Images created:${NC}"
docker images | grep "${PROJECT_NAME}" | \
  awk '{printf "  %-40s %-15s %s\n", $1":"$2, $3, $7" "$8}'

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Build complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Next steps:"
echo "  Start:  docker compose up -d"
echo "  Logs:   docker compose logs -f"
echo "  Stop:   docker compose down"