# Lossy Speculative Decoding

![normal-specdec](./normal-specdec.png)
- Normal speculative decoding passes in the speculated tokens to the target model, and then you run rejection sampling to figure out which tokens to accept.

![lossy-specdec](./specdec+liklihood.png)
- During the forward pass, we can also calculate the liklihood of the set of passed tokens
- I want to add an additional step in the speculative decoding step, which would involve calculating the log liklihood of the set of tokens, and if it passes some predetermined threshold, I want to acept the entire chunk of tokens + its final bonus token.
- The threshold will be scaled to between 0 and 1. #

## Evaluation
- We will use [Nemotron Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) as the evaluation harness.
- We will test on GSM8K (Math) and Humaneval (coding), both built into the framework.
- We will create a 'calibrate' mode which will run the model with a speculator, but only log all of the log-liklihood scores so that we can pick a threshold. 
- Then we need an actual run mode where all score better than that threshold accept the whole chunk of tokens, and all scores less falllback to normal speculative decoding.
- Then I want to repeatedly lower the threshold and show scores decreasing while speed increases.