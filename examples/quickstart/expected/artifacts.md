# Expected Artifacts

This file documents the files a quickstart run should produce. It is intentionally descriptive, not generated output.

## Mock mining

Run:

```bash
bash run_mock_mining.sh
```

Expected artifacts under the chosen output directory:

- `factor_library.json`
- `factor_library/` if the library serializer emits the directory form before final JSON normalization

## Real-data-shaped mining

Run:

```bash
bash run_real_data_shape.sh
```

Expected artifacts:

- `factor_library.json`
- any session or log files created by the CLI in the same output directory

## Evaluation

Run:

```bash
bash run_evaluation.sh
```

Expected behavior:

- console summary for strict recomputation
- no repo-local artifacts

## Benchmark table1

Run:

```bash
bash run_benchmark.sh
```

The helper is a dry run unless called with `--run`. Expected artifacts under the chosen
benchmark output directory after a live run:

- `benchmark/table1/<baseline>.json`
- `benchmark/table1/<baseline>_manifest.json`
- `benchmark/table1/<baseline>/runtime/factor_library.json`
- `benchmark/table1/<baseline>/runtime/run_manifest.json`
- `benchmark/table1/<baseline>/runtime/session.json`
- `benchmark/table1/<baseline>/runtime/session_log.json`

If you later run the broader benchmark surface, the runtime suite also writes:

- `benchmark/ablation/memory_ablation.json`
- `benchmark/ablation/strategy_grid.json`
- `benchmark/cost_pressure/<baseline>.json`
- `benchmark/efficiency/`
- `benchmark/suite.json`
