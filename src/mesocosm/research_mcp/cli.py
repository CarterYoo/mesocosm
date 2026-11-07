"""Operator-only configuration and masked corpus preparation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from mesocosm.research_mcp.corpus import Corpus, _read_file, review_source
from mesocosm.research_mcp.service import Settings

designer_app = typer.Typer(no_args_is_help=True, help="Restricted fixed-policy designer MCP")


@designer_app.command()
def serve(config: Path = typer.Option(..., "--config"),
          json_: bool = typer.Option(False, "--json", help="stdio is always the MCP JSON-RPC protocol")) -> None:
    """Run the six-tool stdio server with private operator-owned settings."""
    from mesocosm.research_mcp.server import serve as run

    try:
        settings = Settings.model_validate_json(_read_file(config))
        asyncio.run(run(settings))
    except ImportError as exc:
        typer.echo("Designer MCP requires: uv sync --extra mcp", err=True)
        raise typer.Exit(1) from exc


@designer_app.command("mask-corpus")
def mask_corpus(source: Path = typer.Argument(...),
                store: Path = typer.Option(..., "--store"),
                policy: Path | None = typer.Option(None, "--policy"),
                source_format: str = typer.Option("bundle", "--source-format"),
                review: bool = typer.Option(False, "--review", help="Print private review inventory without importing"),
                json_: bool = typer.Option(False, "--json")) -> None:
    """Inventory a teacher bundle, or build an explicitly reviewed masked projection."""
    try:
        if review:
            result = review_source(source, source_format)
        elif policy:
            result = Corpus(store).import_source(source, policy=json.loads(_read_file(policy)),
                                                 source_format=source_format)
        else:
            raise ValueError("Supply --review, or --policy with a source-hash-bound operator approval.")
    except (ValueError, OSError) as exc:
        typer.echo(json.dumps({"ok": False, "message": str(exc)}))
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, ensure_ascii=False))
