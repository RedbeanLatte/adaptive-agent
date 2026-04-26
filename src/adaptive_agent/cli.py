from __future__ import annotations

import typer

app = typer.Typer(
    name="adaptive-agent",
    help="Adaptive AI Agent CLI - 자연어 작업을 생성 도구로 실행합니다.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("run")
def run_cmd(
    task: str = typer.Argument(..., help="실행할 자연어 작업."),
    session: str = typer.Option("default", "--session", "-s", help="세션 id."),
) -> None:
    """단일 작업을 비대화형으로 실행합니다."""
    from adaptive_agent.runtime import run_one_shot

    exit_code = run_one_shot(task=task, session_id=session)
    raise typer.Exit(code=exit_code)


@app.command("repl")
def repl_cmd(
    session: str = typer.Option("default", "--session", "-s", help="세션 id."),
) -> None:
    """대화형 REPL 세션을 시작합니다."""
    from adaptive_agent.runtime import run_repl

    exit_code = run_repl(session_id=session)
    raise typer.Exit(code=exit_code)
