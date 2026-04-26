#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh                         Start the interactive REPL
  ./run.sh "natural language task"  Run one task and exit
  ./run.sh --session <id>           Start REPL with a specific session id
  ./run.sh --session <id> "task"    Run one task in a specific session
  ./run.sh --test                   Run the no-network test suite
  ./run.sh --cli-help               Show the underlying CLI help

By default this uses qwen3.6:27b via the configured Ollama Funnel endpoint.
EOF
}

find_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    command -v "$PYTHON_BIN"
    return
  fi

  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
      then
        command -v "$candidate"
        return
      fi
    fi
  done

  echo "Python 3.11+ is required. Install Python 3.11 or set PYTHON_BIN=/path/to/python." >&2
  exit 1
}

ensure_venv() {
  if [[ ! -x ".venv/bin/python" ]]; then
    local py
    py="$(find_python)"
    "$py" -m venv .venv
  fi

  # Skip pip install if the editable entrypoint already exists. Two concurrent
  # ./run.sh invocations otherwise race on .venv/bin/adaptive-agent. After
  # changing pyproject dependencies, remove .venv to force a reinstall.
  if [[ ! -x ".venv/bin/adaptive-agent" ]]; then
    PIP_DISABLE_PIP_VERSION_CHECK=1 .venv/bin/python -m pip install -q -e ".[dev]"
  fi
}

run_agent() {
  local env_args=(
    -u AGENT_MODEL
    -u OPENAI_BASE_URL
    -u OPENAI_API_KEY
    -u AGENT_HTTP_TIMEOUT
    -u AGENT_ENV_FILE
    -u AGENT_EMBEDDING_MODEL
    AGENT_STATE_DIR="${AGENT_STATE_DIR:-$ROOT_DIR/.agent_state}"
  )

  env "${env_args[@]}" .venv/bin/python -m adaptive_agent "$@"
}

session_id="${AGENT_SESSION_ID:-local}"

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  --test)
    ensure_venv
    PATH="$ROOT_DIR/.venv/bin:$PATH" .venv/bin/python -m pytest -q
    exit 0
    ;;
  --cli-help)
    ensure_venv
    .venv/bin/python -m adaptive_agent --help
    exit 0
    ;;
  --session)
    if [[ $# -lt 2 ]]; then
      echo "--session requires an id." >&2
      exit 1
    fi
    session_id="$2"
    shift 2
    ;;
esac

ensure_venv

if [[ $# -eq 0 ]]; then
  run_agent repl -s "$session_id"
else
  run_agent run -s "$session_id" "$*"
fi
