
# Verifies all services are healthy
# Usage: ./scripts/docker_healthcheck.sh
# Exit code: 0 = all healthy, 1 = one or more unhealthy

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ALL_HEALTHY=true

echo "AI Career Platform — Health Check"
echo "=================================="

# ── Check Docker containers ────────────────────────────────────────────
echo ""
echo "Container Health:"
containers=(
  "career_mongodb:MongoDB"
  "career_api:FastAPI"
  "career_streamlit:Streamlit"
)

for entry in "${containers[@]}"; do
  container="${entry%%:*}"
  label="${entry##*:}"

  STATUS=$(docker inspect \
    --format='{{.State.Health.Status}}' \
    "$container" 2>/dev/null || echo "not_running")

  if [ "$STATUS" = "healthy" ]; then
    echo -e "  ${GREEN}✓${NC} ${label}: healthy"
  elif [ "$STATUS" = "starting" ]; then
    echo -e "  ${YELLOW}⏳${NC} ${label}: starting"
    ALL_HEALTHY=false
  else
    echo -e "  ${RED}✗${NC} ${label}: ${STATUS}"
    ALL_HEALTHY=false
  fi
done

# ── Check HTTP endpoints ───────────────────────────────────────────────
echo ""
echo "HTTP Endpoints:"

check_endpoint() {
  local url="$1"
  local label="$2"
  local expected_status="${3:-200}"

  HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time 5 "$url" 2>/dev/null || echo "000")

  if [ "$HTTP_STATUS" = "$expected_status" ]; then
    echo -e "  ${GREEN}✓${NC} ${label}: HTTP ${HTTP_STATUS}"
  else
    echo -e "  ${RED}✗${NC} ${label}: HTTP ${HTTP_STATUS} (expected ${expected_status})"
    ALL_HEALTHY=false
  fi
}

check_endpoint "http://localhost:8000/health"          "API Liveness"
check_endpoint "http://localhost:8000/ready"           "API Readiness"
check_endpoint "http://localhost:8000/docs"            "API Docs"
check_endpoint "http://localhost:8501/_stcore/health"  "Streamlit"

# ── Check MongoDB connectivity via API ─────────────────────────────────
echo ""
echo "Database:"

DB_HEALTH=$(curl -s http://localhost:8000/api/v1/admin/health \
  --max-time 10 2>/dev/null || echo '{"status":"unreachable"}')

DB_STATUS=$(echo "$DB_HEALTH" | \
  python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" \
  2>/dev/null || echo "parse_error")

if [ "$DB_STATUS" = "healthy" ]; then
  echo -e "  ${GREEN}✓${NC} MongoDB: healthy (via API health check)"
else
  echo -e "  ${RED}✗${NC} MongoDB: ${DB_STATUS}"
  ALL_HEALTHY=false
fi

# ── Final result ───────────────────────────────────────────────────────
echo ""
echo "=================================="

if $ALL_HEALTHY; then
  echo -e "${GREEN}✓ All systems healthy${NC}"
  echo ""
  echo "  Dashboard: http://localhost:8501"
  echo "  API Docs:  http://localhost:8000/docs"
  exit 0
else
  echo -e "${RED}✗ One or more systems unhealthy${NC}"
  echo ""
  echo "Debug commands:"
  echo "  docker compose logs api"
  echo "  docker compose logs mongodb"
  echo "  docker compose ps"
  exit 1
fi
