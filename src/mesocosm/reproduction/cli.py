"""Machine-readable research-agent tool dispatcher."""

from __future__ import annotations

import asyncio
import contextlib
import json
import pathlib
import sys

import typer

from mesocosm.reproduction.service import ResearchWorkbench

research_app = typer.Typer(no_args_is_help=True, help="Evidence, factors, learning and candidate lifecycle")


@research_app.command("tools")
def tools(json_: bool = typer.Option(False, "--json")) -> None:
    """List the research tool contracts without starting any jobs."""
    typer.echo(json.dumps(ResearchWorkbench.tools(), ensure_ascii=False, indent=None if json_ else 2))


@research_app.command("call")
def call(tool: str, request: pathlib.Path = typer.Option(..., "--request"),
         root: pathlib.Path = typer.Option(pathlib.Path("runs/research"), "--root"),
         json_: bool = typer.Option(False, "--json"),
         worker: str | None = typer.Option(None, "--worker", help="Operator-approved execution endpoint"),
         worker_token_env: str = typer.Option("MESOCOSM_WORKER_TOKEN", "--worker-token-env"),
         server: str | None = typer.Option(None, "--server", help="Operator-approved vLLM serving endpoint"),
         server_token_env: str | None = typer.Option(None, "--server-token-env"),
         shared_checkpoint_paths: bool = typer.Option(False, "--shared-checkpoint-paths",
             help="Operator confirms the workbench and vLLM share the same checkpoint filesystem"),
         allow_test_evaluation: bool = typer.Option(False, "--allow-test-evaluation",
                                                    help="Independent operator only; never expose to a search agent"),
         ) -> None:
    """Dispatch one JSON tool request; diagnostics never mix with the response."""
    ok = True
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if server and not shared_checkpoint_paths:
                raise ValueError("serving deployment requires shared checkpoint paths; remote upload is unsupported")
            workbench = ResearchWorkbench(root, allow_test_evaluation=allow_test_evaluation,
                                          allowed_workers={worker: worker_token_env} if worker else None,
                                          allowed_servers={server: server_token_env} if server else None)
            result = asyncio.run(workbench.call(tool, json.loads(request.read_text())))
        payload = {"ok": True, "tool": tool, "result": result}
    except Exception as exc:
        ok = False
        # Errors can contain private dataset rows or upstream credentials.
        payload = {"ok": False, "tool": tool, "error_type": type(exc).__name__,
                   "message": "Request failed. Inspect the private store/job diagnostics; no automatic retry occurred."}
    typer.echo(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=None if json_ else 2))
    if not ok:
        raise typer.Exit(1)
