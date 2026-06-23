# Linear DFlash Lossy Spec Decode

This experiment only targets the linear DFlash/spec-v2 path:

- `--speculative-algorithm STANDALONE`
- `SGLANG_ENABLE_SPEC_V2=True`
- `--speculative-eagle-topk 1`

It does not modify tree speculative sampling, EAGLE tree verification, NGRAM
verification, or the speculative CUDA sampling kernel.

## Calibrate

Launch a local server from this checkout and collect perplexity score quantiles:

```bash
python lossy-spec-dec/eval_sweep.py \
  --mode calibrate \
  --model-path <TARGET_MODEL> \
  --speculative-draft-model-path <DFLASH_DRAFT_MODEL> \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --calibration-requests 128 \
  --output-dir lossy-spec-dec/runs/calibrate
```

Attach to an already-running endpoint:

```bash
python lossy-spec-dec/eval_sweep.py \
  --mode calibrate \
  --base-url http://127.0.0.1:30000 \
  --output-dir lossy-spec-dec/runs/calibrate
```

Calibration writes JSONL records with chunk perplexity scores, normal accept
length, and whether the configured threshold would pass. Lower perplexity is
better; a threshold of `1.0` only passes perfect chunks, and larger thresholds
accept more chunks.

## Sweep

Use NeMo Evaluator through a command template. The harness substitutes
`{task}`, `{openai_base_url}`, `{model_id}`, and `{output_dir}`.

```bash
python lossy-spec-dec/eval_sweep.py \
  --mode sweep \
  --base-url http://127.0.0.1:30000 \
  --model-id <SERVED_MODEL_NAME> \
  --thresholds 1.0,1.05,1.1,1.25 \
  --tasks gsm8k,humaneval \
  --output-dir lossy-spec-dec/runs/sweep \
  --nemo-command-template 'nemo-evaluator-launcher run --task {task} --endpoint-type openai --base-url {openai_base_url} --model {model_id} --output-dir {output_dir}'
```

The sweep writes `runs.jsonl`, `summary.csv`, and `summary.md` with quality
metrics, normalized `quality_metric`/`throughput`/`latency` columns when those
can be inferred from NeMo output, average speculative accept length, score
quantiles, average lossy score, and forced full-accept rate. Each task run also
writes its raw DFlash score records to `dflash_scores.jsonl` under the
threshold/task output directory.
