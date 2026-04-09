#!/usr/bin/env bash
set -euo pipefail

# Verify required docker-compose services are running, then launch demo app.
# Usage:
#   scripts/run_demo_with_docker_check.sh
#   scripts/run_demo_with_docker_check.sh --start-missing
#   scripts/run_demo_with_docker_check.sh --app-cmd "python demo/app.py"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
APP_CMD=".venv/bin/streamlit run demo/app.py"
START_MISSING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-missing)
      START_MISSING=1
      shift
      ;;
    --app-cmd)
      APP_CMD="${2:-}"
      if [[ -z "${APP_CMD}" ]]; then
        echo "Error: --app-cmd requires a value."
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--start-missing] [--app-cmd \"<command>\"]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Run with --help for usage."
      exit 1
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not in PATH."
  exit 1
fi

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "Error: compose file not found at ${COMPOSE_FILE}"
  exit 1
fi

cd "${ROOT_DIR}"

if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon is not running."
  exit 1
fi

mapfile -t REQUIRED_SERVICES < <(docker compose -f "${COMPOSE_FILE}" config --services)
mapfile -t RUNNING_SERVICES < <(docker compose -f "${COMPOSE_FILE}" ps --services --status running)

if [[ ${#REQUIRED_SERVICES[@]} -eq 0 ]]; then
  echo "No services found in ${COMPOSE_FILE}."
  exit 1
fi

MISSING_SERVICES=()
for svc in "${REQUIRED_SERVICES[@]}"; do
  found=0
  for running in "${RUNNING_SERVICES[@]:-}"; do
    if [[ "${svc}" == "${running}" ]]; then
      found=1
      break
    fi
  done
  if [[ ${found} -eq 0 ]]; then
    MISSING_SERVICES+=("${svc}")
  fi
done

echo "Required services: ${REQUIRED_SERVICES[*]}"
if [[ ${#RUNNING_SERVICES[@]} -gt 0 ]]; then
  echo "Running services: ${RUNNING_SERVICES[*]}"
else
  echo "Running services: (none)"
fi

if [[ ${#MISSING_SERVICES[@]} -gt 0 ]]; then
  echo "Missing services: ${MISSING_SERVICES[*]}"
  if [[ ${START_MISSING} -eq 1 ]]; then
    echo "Starting missing services..."
    docker compose -f "${COMPOSE_FILE}" up -d "${MISSING_SERVICES[@]}"
  else
    echo "Tip: run with --start-missing to auto-start missing services."
    exit 1
  fi
fi

mapfile -t RUNNING_AFTER < <(docker compose -f "${COMPOSE_FILE}" ps --services --status running)
for svc in "${REQUIRED_SERVICES[@]}"; do
  found=0
  for running in "${RUNNING_AFTER[@]:-}"; do
    if [[ "${svc}" == "${running}" ]]; then
      found=1
      break
    fi
  done
  if [[ ${found} -eq 0 ]]; then
    echo "Error: service '${svc}' is still not running."
    exit 1
  fi
done

echo "All required docker services are running."
echo "Launching app command: ${APP_CMD}"
exec bash -lc "${APP_CMD}"
