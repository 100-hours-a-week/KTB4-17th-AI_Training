# Observable checks before model training

`Model pre-training CI` (`.github/workflows/pretraining-ci.yml`) runs on relevant pull requests, pushes to `main` and `codex/**`, and manual dispatch. It uses Python 3.12 on an Ubuntu CPU runner, read-only repository access, SHA-pinned official Actions, pinned lightweight test tools, a 20-minute timeout and cancellation of superseded runs. Fork pull requests need no secrets.

The workflow stops before model training: `training_executed=false`, `external_model_downloads=0`, `gpu_used=false`. Repository checkout may include existing data and weights; no model/tokenizer fetch, teacher call, GPU invocation, checkpoint write, training requirements installation, deployment or AWS operation occurs. Only `train_lora.py --help` is run; its argparse exits before ML imports. Syntax uses AST parsing and does not import application files.

## Local commands

From the repository root, create a Python 3.12 virtual environment and install `ci/requirements.txt`. Do not install `Lena-Training/training/requirements.txt` for this CI.

```sh
python -m pip install -r ci/requirements.txt
mkdir -p reports
python -m pytest tests/pretraining --junitxml=reports/unit-junit.xml --cov=ci.validators --cov-report=json:reports/coverage.json --cov-report=xml:reports/coverage.xml --cov-report=term-missing
python -m ci.contracts --output-dir reports
python -m ci.preflight --root . --output-dir reports
# Supply the actual exit status of each command above (zero means passed):
python -m ci.report --output-dir reports --unit-exit 0 --contracts-exit 0 --preflight-exit 1
```

`ci.preflight` exits 1 for invalid/missing saved inputs, corrupt JSON, bad manifests/messages/completions, duplicate IDs/content, cross-split repeated prompts or family leakage; otherwise 0. Both JSON and JUnit are written when validation fails, including when the validator has a syntax/import failure. The independent standard-library evidence helpers, contracts and report entrypoints do not import the validator; unavailable pytest/JUnit/coverage remains explicitly unavailable in retained aggregate JSON/Markdown. `ci.report` exits 1 when a stage fails or evidence is missing/invalid, and always writes `summary.json` and `summary.md` for ordinary validation failures. Never replace a failed preflight status with zero. The workflow uses actual step outcomes, preserving the red gate while running reporting and artifact upload with `always()`.

## Checks and report scope

The standard-library validator checks the tagging/extraction train and validation JSONL, manifest types/counts, fixed in-repository source paths, input SHA256, every saved system message against its manifest prompt SHA256, message roles/content, finite strict JSON object completions, tagging label structure and extraction score structure. It hashes every audited file and reports bytes, row/family counts, joins, duplicates, overlaps and check durations. Every parsed saved completion is canonically compared with the label belonging to its manifest-source ID: tagging uses primary/secondary/off_topic and extraction uses output. Only compared/mismatched counts are published. IDs and family membership are resolved through teacher source IDs; missing optional `group` in tagging input does not skip this join. Conflicting/duplicate/missing source IDs and missing family membership fail closed. Diagnostic codes/counts omit raw IDs, prompts, completions and source examples. Repeated defects produce one issue per code with the true occurrence count.

JSONL is streamed in binary lines and SHA256 is updated while reading. Limits are 128 MiB per JSONL file, 2 MiB per line, and 100,000 nonblank records per file; manifests are limited to 64 KiB, and each declared train/validation/skip/drop count is capped at the same 100,000-record bound. Oversized or nonregular inputs fail closed, without fingerprinting an input rejected for size. These limits exceed the current largest saved file (about 31 MB) and teacher source (10,279 records). Only ID-to-family/label hashes, duplicate/prompt hashes and aggregate counters survive each parsed row; complete source/teacher rows are not retained.

The prompt hash check verifies saved-file consistency. It cannot establish the current application's serving prompt or business schema: this clone does not contain the companion `app` package. The original nine files under `Lena-Training/tests/` depend on that package and/or omitted application scripts and are **not verified** by this workflow. No fake app modules replace them. Fixture tests exercise the standalone validators and report commands; the real standalone `assert_fits` boundary is tested with a deterministic length-only tokenizer double. This is not a real tokenizer or sequence-length audit.

Coverage is explicitly **`ci/validators.py` only**, from fixture execution. Shared parsing/evidence helpers and report/contract modules are exercised by the suite but excluded from this stated coverage scope. It does not describe legacy application tests, trainer/model execution, data quality, or model accuracy. Current application semantics, tokenizer lengths, trainer dependencies/GPU compatibility and training hyperparameters are outside the verified scope. `training_ready` therefore remains false even for a structurally valid fixture. An unreviewed manifest adds an explicit warning/readiness reason; it can never be reported ready.

## Actual baseline, separate from fixture tests

At base commit `7d2208f6577791ce9c8dc879e4b0c647f3f79c54`:

| Saved artifact | Train rows | Validation rows | Train families | Validation families | Overlapping families |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tagging | 6915 | 828 | 997 | 575 | **575 (fails)** |
| Extraction | 851 | 84 | 851 | 84 | 0 |

Tagging resolves all 7743 SFT IDs through the 10279-row teacher source, exposing family leakage even though the 7743-row intermediate tagging input omits `group`. This baseline must fail CI until the source split/artifact is corrected in a separately reviewed data change. It is not an allowed historical exception.

Both saved manifests set `unreviewed_allowed=true`; neither dataset is ready for training. The base commit also explicitly says extraction output predates the newest filtering code and must be rebuilt before use. `saved_extraction_rebuild_required` remains an explicit provenance/readiness limitation. This CI does not rebuild or edit data, infer readiness from historical training metrics, or prove extraction outputs satisfy unavailable current application filters.

## Evidence

The aggregate requires exactly tagging and extraction, exactly python_syntax and train_lora_argparse_help, matching original preflight JUnit, fixture JUnit and actual validator-scoped coverage before it can pass. Missing checks and contradictory statuses/metrics fail aggregation; valid failed contract checks retain their timings/metrics. All dataset rows are rendered before warning/error bullets.

The Actions summary and `model-pretraining-<run>-<attempt>` artifact are published after failed checks. Retention is 14 days. Evidence includes schema version, repository/commit/run/UTC identity, statuses, durations, sanitized issue codes/counts, input hashes/bytes, readiness reasons, fixture JUnit, real-data preflight JUnit, scoped coverage JSON/XML, contracts JSON, and aggregate JSON/Markdown. Artifacts contain no environment dumps, raw prompts/data or secrets. Cancellation or runner/setup failure can prevent evidence generation; a missing artifact is not a successful audit.
