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
![tok-threshold](./images/tok-vs-threshold.png)
![benchmark-threshold](./images/benchmark-vs-threshold.png)
![tok-benchmark](./images/benchmark-vs-tok.png)

- As you decrease threshold (make it more likely to accept), your speed to smoothly interpolate from speculative decoding speed to speculator speed
- As you decrease threshold, your benchmarks should smoothly interpolate from the verifier to the speculator
- You hopefully have some threshold with small benchmarks drops and large tok/s gains.