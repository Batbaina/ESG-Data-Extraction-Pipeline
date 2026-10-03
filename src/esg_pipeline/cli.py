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
    return create_llm_backend(settings)


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
    """Check the configured LLM backend and model."""
    s = Settings()
    data = _client(s).check()

    console.print(
        f"Backend: [bold]{s.llm_backend}[/bold]"
    )
    console.print(
        f"Model: [bold]{s.model}[/bold]"
    )

    if s.llm_backend.lower() == "ollama":
        names = [
            m.get("name", "")
            for m in data.get("models", [])
        ]

        console.print(
            f"Ollama API: [green]{s.ollama_url}[/green]"
        )

        if any(n.startswith(s.model) for n in names):
            console.print(
                "[green]Status: model installed and backend ready.[/green]"
            )
        else:
            console.print(
                f"[yellow]Status: model not installed. "
                f"Run: ollama pull {s.model}[/yellow]"
            )

    elif s.llm_backend.lower() == "transformers":
        device = data.get("device", "unknown")

        console.print(
            f"Device: [bold]{device}[/bold]"
        )
        console.print(
            "[green]Status: model loaded and backend ready.[/green]"
        )

    else:
        console.print(
            f"[yellow]Backend response: {data}[/yellow]"
        )


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
