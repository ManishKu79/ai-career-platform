
# Usage: ./scripts/docker_start.sh [--build] [--prod]

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

COMPOSE_FILE="docker-compose.yml"
BUILD_FLAG=""
DETACH="-d"

for arg in "$@"; do
  case $arg in
    --build) BUILD_FLAG="--build" ;;
    --prod)  COMPOSE_FILE="docker-compose.yml -f docker-compose.prod.yml" ;;
    --foreground) DETACH="" ;;
  esac
done

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE} AI Career Intelligence Platform${NC}"
echo -e "${BLUE}=====================================${NC}"

# ── Check .env.docker exists ───────────────────────────────────────────
if [ ! -f ".env.docker" ]; then
  echo -e "${YELLOW}Creating .env.docker from defaults...${NC}"
  cat > .env.docker << 'EOF'
MONGODB_URL=mongodb://mongodb:27017
MONGODB_DB_NAME=career_platform
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
API_BASE_URL=http://api:8000
SPACY_MODEL=en_core_web_lg
MAX_FILE_SIZE_MB=10
ATS_SCORE_THRESHOLD=0.5
EOF
  echo -e "${GREEN}✓ .env.docker created${NC}"
fi

# ── Pull MongoDB image ─────────────────────────────────────────────────
echo -e "\n${YELLOW}[1/4] Pulling MongoDB image...${NC}"
docker pull mongo:7.0 --quiet
echo -e "${GREEN}✓ MongoDB image ready${NC}"

# ── Start services ─────────────────────────────────────────────────────
echo -e "\n${YELLOW}[2/4] Starting services...${NC}"

docker compose \
  -f $COMPOSE_FILE \
  up $BUILD_FLAG $DETACH \
  --remove-orphans

if [ -z "$DETACH" ]; then
  exit 0
fi

echo -e "${GREEN}✓ Containers started${NC}"

# ── Wait for health checks ─────────────────────────────────────────────
echo -e "\n${YELLOW}[3/4] Waiting for services to be healthy...${NC}"
echo "    This may take 60-90 seconds (spaCy model loading)"

MAX_WAIT=120
WAITED=0
INTERVAL=5

services=("career_mongodb" "career_api" "career_streamlit")

for service in "${services[@]}"; do
  echo -n "    Waiting for ${service}..."
  WAITED=0

  while [ $WAITED -lt $MAX_WAIT ]; do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' "$service" 2>/dev/null || echo "not_found")

    if [ "$STATUS" = "healthy" ]; then
      echo -e " ${GREEN}✓ healthy${NC}"
      break
    elif [ "$STATUS" = "unhealthy" ]; then
      echo -e " ${RED}✗ unhealthy${NC}"
      echo "    Check logs: docker logs $service"
      break
    else
      echo -n "."
      sleep $INTERVAL
      WAITED=$((WAITED + INTERVAL))
    fi
  done

  if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e " ${YELLOW}⚠ timeout (may still be starting)${NC}"
  fi
done

# ── Verify API responds ────────────────────────────────────────────────
echo -e "\n${YELLOW}[4/4] Verifying API endpoint...${NC}"
sleep 2

if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
  echo -e "${GREEN}✓ FastAPI is responding${NC}"
else
  echo -e "${YELLOW}⚠ API not responding yet — may still be loading${NC}"
fi

# ── Final output ───────────────────────────────────────────────────────
echo -e "\n${GREEN}=====================================${NC}"
echo -e "${GREEN} Platform Running!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo -e "  📊 Dashboard:   ${BLUE}http://localhost:8501${NC}"
echo -e "  🚀 FastAPI:     ${BLUE}http://localhost:8000${NC}"
echo -e "  📖 API Docs:    ${BLUE}http://localhost:8000/docs${NC}"
echo -e "  🏥 Health:      ${BLUE}http://localhost:8000/health${NC}"
echo -e "  🗄️  MongoDB:    ${BLUE}localhost:27017${NC}"
echo ""
echo -e "  Logs:    ${YELLOW}docker compose logs -f [api|streamlit|mongodb]${NC}"
echo -e "  Stop:    ${YELLOW}docker compose down${NC}"
echo -e "  Rebuild: ${YELLOW}docker compose up --build${NC}"
