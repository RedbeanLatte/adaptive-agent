# Adaptive Agent CLI

자연어 작업을 받아 필요한 Python tool을 즉석에서 생성·실행하고, 사용자 승인 후 저장해 다음 세션에서 재사용·점검·삭제할 수 있는 CLI agent입니다. agent framework나 OpenAI SDK를 쓰지 않고, OpenAI-compatible `/v1/chat/completions` 를 `requests`로 직접 호출합니다.

## 실행

```bash
./run.sh                                    # 대화형 REPL
./run.sh "작업 내용"                          # 1회 실행 후 종료
./run.sh --session <id> "작업 내용"            # 세션 id 지정
./run.sh --cli-help                         # 인자 도움말
./run.sh --test                             # 회귀 테스트 (네트워크 없음)
```

예시:

```bash
./run.sh "fixtures/example1_monsters.json 에서 hp가 100 이상인 몬스터 이름과 평균 hp를 알려줘"
```

`run.sh`가 `.venv` 생성과 `pip install -e ".[dev]"`를 자동으로 처리합니다. Python 3.11+ 가 필요합니다.

## LLM 설정

기본값은 `src/adaptive_agent/llm.py` 안에 둔 Ollama endpoint와 chat model `qwen3:30b`, embedding model `nomic-embed-text` 를 사용합니다. `run.sh`는 재현성을 위해 `AGENT_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `AGENT_HTTP_TIMEOUT`, `AGENT_ENV_FILE`, `AGENT_EMBEDDING_MODEL` 을 실행 환경에서 제거하고 코드의 기본값만 적용합니다.

다른 OpenAI-compatible endpoint나 모델을 쓰려면 `llm.py`의 `DEFAULT_BASE_URL` / `DEFAULT_MODEL`, `embeddings.py`의 `DEFAULT_EMBEDDING_MODEL` 을 직접 수정하거나, `run.sh`에서 위 환경변수 unset 부분을 제거한 뒤 직접 export 하면 됩니다.

## REPL 명령

```text
/help
/tools list
/tools inspect <name>
/tools verify <name>
/tools remove <name>
/exit
```

`/`로 시작하는 명령은 LLM을 호출하지 않고 런타임에서 결정론적으로 처리합니다.

## 동작 요약

- 모델은 매 턴 한 개의 JSON action을 반환합니다 — `answer`, `ask_user`, `run_builtin`, `reuse_tool`, `create_tool`, `request_approval` 6종.
- 런타임이 action을 검증한 뒤 built-in tool 실행, saved tool 재사용, generated tool 생성·실행 중 하나를 수행합니다.
- generated/saved tool은 subprocess에서 실행되며, Python audit-hook guard로 workroot 밖 쓰기와 `.agent_state` / `.git` 쓰기가 차단됩니다.
- 성공한 generated tool은 임시 workspace에서 같은 `example_args`로 1회 replay 검증을 마친 뒤, 사용자 승인(`y/n`)을 받은 경우에만 `.agent_state/tools/<name>/` 에 저장됩니다.
- 다음 세션에서는 catalog lookup을 먼저 시도하고, 적합한 saved tool이 있으면 기본 semantic embedding matching 후 token heuristic fallback으로 재사용을 우선합니다.
- 실행 실패 시 compact failure summary를 만들어 1회 repair retry를 수행하고, 두 번째 실패는 사용자에게 보고하고 종료합니다.
- 세션별 compact JSONL trace를 `.agent_state/traces/<session>.jsonl` 에 기록합니다. raw chain-of-thought는 저장하지 않고 `reasoning_summary` 와 구조화된 trace event만 남깁니다.
- Qwen 계열 모델의 `<think>...</think>` 블록은 응답에서 제거합니다.

## 설계 결정

- **agent framework 미사용**: LLM HTTP 호출, JSON action schema 검증, tool dispatch loop를 직접 구현했습니다.
- **built-in 최소화**: `inspect_file` / `read_text_file` / `write_text_file` / `list_files` 4개로 한정해, 실제 분석·변환·정리 작업은 generated tool이 담당하도록 강제했습니다.
- **subprocess + audit-hook guard**: generated/saved tool 격리는 OS/container 수준이 아니라 Python audit hook으로 workroot 밖 파일 쓰기와 runtime state 디렉터리 쓰기를 차단하는 방식입니다.
- **두 단계 저장 게이트**: 저장은 replay 검증 통과 + 사용자 승인 두 단계를 모두 거친 경우에만 수행합니다. saved tool 패키지는 `tool.py` / `manifest.json` / `verification.json` / `last_run.json` (옵션 `embedding.json`) 로 구성됩니다.
- **slash command 분리**: saved tool 관리는 LLM이 호출하는 built-in tool이 아니라 REPL slash command로 노출해, LLM 의존 없이 확인·검증·삭제가 가능합니다.
- **structured action only**: LLM 응답은 단일 JSON action으로 강제되며, fence 없는 prose 응답은 schema error로 다음 턴에 재시도시킵니다.

세부 구성 다이어그램과 호출 흐름은 [`docs/architecture/system-design.md`](docs/architecture/system-design.md) 를 참고하세요.

## 한계 및 개선 방향

- 실제 품질은 사용하는 LLM의 JSON/action 준수 능력과 코드 생성 품질에 영향을 받습니다.
- sandbox는 Python 실행과 파일 쓰기 제한 중심이며, OS/container 수준 격리는 아닙니다.
- 저장 도구 matching은 기본적으로 OpenAI-compatible `/v1/embeddings` 호출 기반 semantic matching을 먼저 시도하며, 실패하거나 적합한 후보가 없으면 token heuristic으로 fallback합니다. `AGENT_EMBEDDING_MODEL` 로 기본 embedding model을 override할 수 있습니다.
- 시스템 프롬프트는 audit/checklist/scaffold 같은 구조화된 작업도 generated tool로 처리하도록 지시하지만, 모델에 따라 builtin read 후 직접 답변으로 끝낼 수 있습니다. trace의 action 시퀀스로 확인할 수 있습니다.
- 사용자가 명시한 파일이 존재하지 않을 때 프롬프트는 `ask_user` 또는 부재 명시 답변을 강제하지만, 모델이 인접 파일로 대체하는 경우가 드물게 발생할 수 있습니다. tool observation의 `file not found` 메시지로 확인할 수 있습니다.
- 향후에는 더 강한 실행 격리, tool 입출력 schema 검증, 모델별 prompt 튜닝, 실패 케이스별 repair 전략을 확장할 수 있습니다.

## 주요 파일

```text
src/adaptive_agent/llm.py             LLM HTTP client와 기본 Ollama 설정
src/adaptive_agent/runtime.py         agent loop, system prompt, action dispatch
src/adaptive_agent/actions.py         JSON action parsing/validation
src/adaptive_agent/builtins.py        inspect/read/write/list built-ins
src/adaptive_agent/catalog.py         saved tool catalog와 manifest 직렬화
src/adaptive_agent/generated_tools.py generated tool subprocess executor
src/adaptive_agent/runtime_guard.py   generated tool filesystem audit hook
src/adaptive_agent/slash_commands.py  REPL slash command dispatcher
src/adaptive_agent/approval.py        y/n 승인 프롬프트 헬퍼
run.sh                                고정 Ollama 기본값으로 실행
fixtures/                             재현 가능한 데모 입력 파일
tests/                                네트워크 없는 회귀 테스트
docs/architecture/system-design.md    컴포넌트/플로우 다이어그램
```

## 개발

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

테스트는 네트워크를 사용하지 않습니다.
