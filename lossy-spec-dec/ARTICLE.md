# Lossy Speculative Decoding

![normal-specdec](./images/normal-specdec.png)
- Normal speculative decoding passes in the speculated tokens to the target model, and then you run rejection sampling to figure out which tokens to accept.

![lossy-specdec](./images/specdec+liklihood.png)
- During the forward pass, we can also calculate the perplexity of the set of passed tokens.
- I want to add an additional step in the speculative decoding step, which would involve calculating the token-normalized perplexity of the set of tokens, and if it is at or below some predetermined threshold, I want to accept the entire chunk of tokens + its final bonus token.
- The threshold starts at 1.0 for perfect chunks and can be raised to accept more chunks.

## Evaluation
- We will use [Nemotron Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) as the evaluation harness.
- We will test on GSM8K (Math) and Humaneval (coding), both built into the framework.
- We will create a 'calibrate' mode which will run the model with a speculator, but only log all of the perplexity scores so that we can pick a threshold.
- Then we need an actual run mode where all scores at or below that threshold accept the whole chunk of tokens, and all higher scores fall back to normal speculative decoding.
- Then I want to repeatedly raise the threshold and show quality decreasing while speed increases.

## Guessed results
![tok-threshold](./images/tok-vs-threshold.svg)
![benchmark-threshold](./images/benchmark-vs-threshold.svg)
![tok-benchmark](./images/benchmark-vs-tok.svg)

- The first plot shows throughput as a function of the perplexity threshold. A threshold of `1.0` only force-accepts perfect chunks; raising the threshold makes the lossy accept rule more permissive, so throughput should move from normal speculative decoding toward the draft model's speed.
- The second plot shows benchmark quality as a function of the same threshold. As the threshold rises, benchmark scores should move from verifier-like quality toward speculator-like quality.
- The third plot puts benchmark score directly against throughput. This is the tradeoff curve: useful thresholds are the points that keep benchmark score close to baseline while buying a large tokens-per-second gain.

Collect the data with:

```bash
.venv-lossy-calibrate/bin/python lossy-spec-dec/run_threshold_eval_sweep.py \
  --thresholds 1,1.001,1.01,1.1,2,10,100 \
  --speculative-num-draft-tokens 5 \
  --server-args '--disable-cuda-graph' \
  --nemo-command-template 'YOUR_EVAL_COMMAND_WITH_{task}_{openai_base_url}_{model_id}_{output_dir}'
```

Then create the figures with:

```bash
.venv-lossy-calibrate/bin/python lossy-spec-dec/plot_threshold_eval_sweep.py \
  --input lossy-spec-dec/runs/threshold-eval/<run-id>/runs.jsonl \
  --output-dir lossy-spec-dec/images
```
