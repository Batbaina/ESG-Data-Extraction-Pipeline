from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from .config import Settings
from .llm import create_llm_backend
from .pipeline import run
from .exporters import finalize_review

app = typer.Typer(no_args_is_help=True)
console = Console()


def _client(settings: Settings):
    return OllamaClient(
        settings.ollama_url, settings.model, settings.generation_options(),
        timeout=settings.request_timeout_seconds,
    )


@app.command()
def config():
    """Show the exact model, prompt and generation/QC settings used by the pipeline."""
    s = Settings()
    table = Table(title="ESG Pipeline Configuration")
    table.add_column("Setting")
    table.add_column("Value")
    for key, value in s.public_dict().items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def check():
    s = Settings()
    data = _client(s).check()
    names = [m.get("name", "") for m in data.get("models", [])]
    console.print(f"[green]Ollama reachable[/green]: {s.ollama_url}")
    console.print(f"Configured model: [bold]{s.model}[/bold]")
    if not any(n.startswith(s.model) for n in names):
        console.print(f"[yellow]Model not found locally. Run: ollama pull {s.model}[/yellow]")
    else:
        console.print("[green]Configured model is installed.[/green]")


@app.command()
def extract(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Path = typer.Option(Path("data/output"), "--output-dir"),
    text_column: str = typer.Option("text", "--text-column"),
    limit: int | None = typer.Option(None, "--limit"),
):
    result = run(input_path, output_dir, text_column, limit)
    t = Table(title="ESG extraction complete")
    t.add_column("Metric"); t.add_column("Value")
    for k in ["records", "candidate_rows", "ready_rows"]:
        t.add_row(k, str(result[k]))
    console.print(t)
    console.print(f"Candidates: {result['candidates']}")
    console.print(f"Auto-QC ready: {result['ready']}")
    console.print(f"Human review: {result['review']}")
    console.print(f"Run manifest: {result['manifest']}")


@app.command()
def finalize(
    review_file: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(Path("data/output/final_finetuning.jsonl"), "--output"),
    min_human_score: int = typer.Option(8, "--min-human-score", min=0, max=10),
):
    output.parent.mkdir(parents=True, exist_ok=True)
    count = finalize_review(review_file, output, min_human_score)
    console.print(f"[green]Finalized {count} human-approved examples[/green] -> {output}")


if __name__ == "__main__":
    app()
