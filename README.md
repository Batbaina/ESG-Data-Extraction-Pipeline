# ESG Data Extraction Pipeline v3

Local-first, reproducible pipeline for converting PDF, Excel/CSV, TXT or Markdown into grounded ESG knowledge units for human validation and later fine-tuning.

## End-to-end architecture

```text
PDF -> Docling -> Markdown -> normalized chunks --+
                                                   |
Excel / CSV / TXT / MD ----------------------------+
                                                   v
                                            Strong ESG Prompt
                                                   v
                                            Qwen3-8B / Ollama
                                                   v
                                      Pydantic structured output
                                         |                  |
                                      invalid             valid
                                         |                  |
                                     retry x N              v
                                                ESG Knowledge Units
                                                         v
                                                  Automatic QC
                                                  /          \
                                               fail          pass
                                                |             |
                                           candidates       ready
                                                              |
                                                              v
                                                        Human review
                                                         5 x (0-2)
                                                              |
                                                              v
                                                           finalize
                                                              |
                                                              v
                                                   final_finetuning.jsonl
```

`ready.jsonl` means **automatic-QC passed**, not human-approved. Only `finalize` creates the human-approved final dataset.

## What v3 adds

- centralized model and generation settings;
- explicit `temperature`, `top_p`, `top_k`, `seed`, `num_ctx`, `num_predict`;
- model parameters recorded with every candidate;
- run manifest with complete settings;
- SHA-256 of input and system prompt;
- Pydantic retry loop;
- configurable automatic QC thresholds;
- human-review workbook with a /10 score;
- final exporter retaining generation metadata;
- included synthetic PDF and Excel examples.

## Included examples

- `data/examples/sample_esg_report.pdf`
- `data/examples/sample_existing_extraction.xlsx`

The example company, metrics and disclosures are synthetic.

## 1. Install Ollama + Qwen3-8B

Ubuntu:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b
ollama run qwen3:8b
```

Use `/bye` to exit the interactive model.

## 2. Install this project

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[pdf,dev]'
```

## 3. Configure the experiment

```bash
cp .env.example .env
```

The Python project reads environment variables. For a shell session you can export them with:

```bash
set -a
. ./.env
set +a
```

Important defaults:

```text
ESG_MODEL=qwen3:8b
ESG_PROMPT_VERSION=v1
ESG_TEMPERATURE=0.0
ESG_TOP_P=0.9
ESG_TOP_K=40
ESG_SEED=42
ESG_NUM_CTX=8192
ESG_NUM_PREDICT=2048
ESG_MAX_RETRIES=2
ESG_CHUNK_MAX_CHARS=6000
ESG_CHUNK_OVERLAP_CHARS=500
ESG_QC_MIN_CONFIDENCE=0.70
ESG_QC_MIN_LEXICAL_GROUNDING=0.45
```

Always verify the effective configuration:

```bash
esg-extract config
esg-extract check
```

## 4. Test the included PDF

Start with a small sample:

```bash
esg-extract extract data/examples/sample_esg_report.pdf --limit 5
```

The run produces:

```text
*_candidates.jsonl
*_ready.jsonl
*_human_review.xlsx
*_run_manifest.json
```

The manifest records the model, all generation parameters, prompt hash, input hash, chunk/QC settings and output paths.

## 5. Test the Excel route

```bash
esg-extract extract \
  data/examples/sample_existing_extraction.xlsx \
  --text-column text \
  --limit 5
```

Your existing columns such as `char_count`, `token_estimate` and `esg_category` can remain in the workbook. `text` is the required source column by default.

## 6. Human evaluation

Open `*_human_review.xlsx` and score each useful candidate from 0 to 2 on:

1. ESG relevance
2. factual faithfulness
3. completeness
4. self-contained quality
5. fine-tuning usefulness

`human_total` is calculated out of 10.

Do not score an extraction as faithful if it introduces knowledge that is absent from the source.

## 7. Finalize approved examples

```bash
esg-extract finalize \
  data/output/YOUR_human_review.xlsx \
  --min-human-score 8 \
  --output data/output/final_finetuning.jsonl
```

Only automatic-QC-passed rows with five valid human scores and a total >= threshold are exported.

## 8. Prompt experimentation

The system prompt is:

```text
prompts/extraction_v1.txt
```

Create a new version without overwriting the previous experiment:

```bash
cp prompts/extraction_v1.txt prompts/extraction_v2.txt
export ESG_PROMPT_VERSION=v2
```

Then rerun the **same representative sample** and compare human scores. The run manifest stores a SHA-256 hash of the exact prompt text.

## 9. Full dataset

After the prompt is stable on your representative evaluation sample:

```bash
esg-extract extract path/to/your_dataset.xlsx --text-column text
```

or:

```bash
esg-extract extract path/to/new_esg_report.pdf
```

## Quality-control limitation

The automatic QC is intentionally transparent and conservative. It checks model confidence, the `source_supported` flag, lexical grounding of `source_facts`, and a basic self-contained heuristic. It cannot prove semantic factual entailment. Human validation therefore remains the gate before final fine-tuning export.
