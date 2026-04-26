# Adaptive Agent — System Design

코드(`src/adaptive_agent/`)를 그대로 매핑한 현재 시스템의 컴포넌트 구성과 요청 흐름.

## 1. Component Overview

```mermaid
flowchart TB
    classDef entry fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef runtime fill:#fff3e0,stroke:#ef6c00,color:#e65100
    classDef action fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    classDef sandbox fill:#ffebee,stroke:#c62828,color:#b71c1c
    classDef storage fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef external fill:#eceff1,stroke:#455a64,color:#263238

    User([User])

    subgraph EntryLayer["Entry (Typer CLI)"]
        RunCmd["<b>run</b><br/>one-shot, non-interactive"]
        ReplCmd["<b>repl</b><br/>interactive loop"]
    end

    subgraph ReplRouter["REPL Input Router"]
        Branch{{"line.startswith('/')<br/>?"}}
    end

    subgraph SlashLayer["Slash Command Dispatcher<br/>(deterministic, no LLM)"]
        SlashHelp["/help"]
        ToolsList["/tools list"]
        ToolsInspect["/tools inspect"]
        ToolsVerify["/tools verify"]
        ToolsRemove["/tools remove"]
    end

    subgraph RuntimeLoop["Agent Runtime Loop (runtime.py)"]
        Session["Session<br/>conversation history"]
        SysPrompt["System Prompt Builder<br/>+ catalog summary"]
        LLM["LLM Client<br/>OpenAI-compatible<br/>/v1/chat/completions"]
        Parser["Action Parser<br/>strict JSON schema"]
        Dispatch{{"Action Dispatcher"}}
        Repair["Repair Policy<br/>max 1 retry per tool"]
    end

    subgraph Actions["6 Action Types"]
        ActAnswer["answer"]
        ActAsk["ask_user"]
        ActBuiltin["run_builtin"]
        ActReuse["reuse_tool"]
        ActCreate["create_tool"]
        ActApproval["request_approval"]
    end

    subgraph ToolLayer["Tool Layer"]
        Builtins["Builtins<br/>inspect / read / write / list"]
        Catalog["Tool Catalog<br/>builtin + saved registry"]
        Matcher["Tool Matching<br/>token heuristic +<br/>semantic embedding"]
        EmbedClient["Embedding Client<br/>/v1/embeddings (optional)"]
    end

    subgraph Sandbox["Execution Sandbox (runtime_guard.py)"]
        Subprocess["subprocess<br/>python -c &lt;code&gt;"]
        Guard["Audit-hook Guard<br/>block writes outside workroot,<br/>.agent_state, .git"]
        Replay["Replay Verification<br/>isolated tempdir copy"]
    end

    subgraph Persist["Persistence (.agent_state/)"]
        ApprovalGate["Approval Gate<br/>explicit y/n"]
        SavedPkg[("tools/&lt;name&gt;/<br/>tool.py<br/>manifest.json<br/>verification.json<br/>last_run.json<br/>embedding.json")]
        Trace["Trace Store<br/>compact JSONL<br/>(no raw CoT)"]
    end

    LLMServer[("OpenAI-compatible<br/>LLM Endpoint<br/>(default: Ollama)")]:::external

    User --> EntryLayer
    RunCmd --> RuntimeLoop
    ReplCmd --> ReplRouter
    Branch -- "yes" --> SlashLayer
    Branch -- "no" --> RuntimeLoop

    SlashLayer --> Catalog
    SlashLayer -. "verify replays code" .-> Sandbox

    Session --> SysPrompt --> LLM
    LLM <--> LLMServer
    LLM --> Parser --> Dispatch
    Dispatch --> Actions

    ActAsk -. "back to user" .-> User
    ActAnswer -. "final reply" .-> User
    ActBuiltin --> Builtins
    ActReuse --> Catalog
    ActCreate --> Sandbox
    ActApproval --> ApprovalGate

    Catalog --> Matcher --> EmbedClient
    EmbedClient <--> LLMServer
    Builtins --> Sandbox
    Catalog -- "saved tool code" --> Sandbox

    Subprocess --> Guard
    Subprocess -- "on success" --> Replay
    Replay --> ApprovalGate
    ApprovalGate -- "approved" --> SavedPkg
    Catalog <--> SavedPkg

    Sandbox -- "stdout / exit / err" --> Dispatch
    Dispatch -- "failure summary" --> Repair --> LLM

    RuntimeLoop --> Trace
    SlashLayer --> Trace

    class RunCmd,ReplCmd entry
    class Session,SysPrompt,LLM,Parser,Dispatch,Repair runtime
    class ActAnswer,ActAsk,ActBuiltin,ActReuse,ActCreate,ActApproval action
    class Subprocess,Guard,Replay sandbox
    class Catalog,Matcher,Builtins,EmbedClient,ApprovalGate,SavedPkg,Trace storage
```

## 2. Request Flow — `create_tool` Happy Path

가장 핵심 경로(generated tool 생성 → 검증 → 승인 → 저장).

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant CLI as CLI (Typer)
    participant RT as Runtime Loop
    participant LM as LLM Client
    participant SB as Sandbox<br/>(subprocess + guard)
    participant RP as Replay (tempdir)
    participant AG as Approval Gate
    participant CT as Catalog
    participant TR as Trace

    U->>CLI: 자연어 작업
    CLI->>RT: run_task(task)
    RT->>LM: messages + system prompt + catalog summary
    LM-->>RT: JSON action
    RT->>TR: event(action)

    alt action = create_tool
        RT->>SB: execute(code, argv)
        SB-->>RT: stdout / stderr / exit
        Note over SB: audit hook blocks writes<br/>outside workroot/.agent_state/.git

        alt 실행 성공
            RT->>RP: replay in isolated copy
            RP-->>RT: verification result
            RT->>RT: pending_save 보관<br/>(answer 이후로 deferred)
        else 실패
            RT->>LM: compact failure summary<br/>(repair, max 1회)
            LM-->>RT: 수정된 code
            RT->>SB: execute (repair)
        end
    end

    RT->>LM: 다음 턴 (answer 유도)
    LM-->>RT: action = answer
    RT-->>U: final_answer 출력

    RT->>AG: 저장 여부 [y/n]
    AG-->>U: prompt
    U-->>AG: y
    AG->>CT: save_generated(spec)
    CT-->>CT: tools/&lt;name&gt;/ 패키지 저장<br/>+ embedding 캐시
    CT->>TR: event(saved_tool)
```

## 3. REPL Slash Command Path

LLM을 거치지 않는 결정론 경로.

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant REPL as REPL Loop
    participant SC as Slash Dispatcher
    participant CT as Catalog
    participant SB as Sandbox
    participant TR as Trace

    U->>REPL: "/tools verify clean_csv"
    REPL->>SC: dispatch(line)
    SC->>CT: lookup(name)
    CT-->>SC: SavedToolSpec (code + example_args)
    SC->>SB: execute_tool_code(...)
    SB-->>SC: result
    SC->>CT: update verification.json + last_run.json
    SC->>TR: event(slash_verify)
    SC-->>U: 검증 요약 출력
```

## 4. Trust & Isolation Boundaries

| 경계 | 신뢰 수준 | 강제 수단 |
|---|---|---|
| User input | untrusted | LLM 입력으로만 사용, action JSON 강제 |
| LLM output | untrusted | `parse_action`이 JSON schema 검증, 알려진 6개 action만 허용 |
| Generated/saved tool code | untrusted | subprocess + Python audit-hook guard, workroot 외 쓰기·`.agent_state`/`.git` 쓰기 차단 |
| Replay 환경 | 분리 | tempdir 복사본에서 실행, 실제 workroot 부수효과 없음 |
| Persistence | gated | explicit `y/n` approval 통과 후에만 `.agent_state/tools/`에 기록 |
| Slash commands | trusted runtime | LLM 우회, deterministic dispatcher만 호출 |

## 5. State Layout

```text
.agent_state/
├── traces/
│   └── <session>.jsonl          # compact event log (no raw CoT)
└── tools/
    └── <name>/
        ├── tool.py              # generated source
        ├── manifest.json        # name/version/desc/example_args/verification_status
        ├── verification.json    # 최근 검증 결과
        ├── last_run.json        # 최근 실행 metric
        └── embedding.json       # semantic matching cache (optional)
```
