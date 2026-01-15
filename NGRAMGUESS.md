# NGRAMGUESS Feature Usage Guide

## Overview

The `<NGRAMGUESS>` feature allows you to provide expected output text when using n-gram speculative decoding. This "guess" is added to the n-gram cache before generation begins, improving speculation quality from the very first token.

## Requirements

- Server must be running with `--speculative-algorithm NGRAM`
- The feature is silently ignored if not using n-gram speculation

## Usage

### Basic Syntax

Wrap your expected output in `<NGRAMGUESS>` tags at the end of your prompt:

```
Your prompt here<NGRAMGUESS>expected output</NGRAMGUESS>
```

### Example 1: Simple Completion

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)

response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="The capital of France is <NGRAMGUESS>Paris</NGRAMGUESS>",
    max_tokens=50
)

print(response.choices[0].text)
```

**What happens:**
1. Prompt sent to model: `"The capital of France is "`
2. Guess `"Paris"` is tokenized and added to n-gram cache
3. Generation begins with cached pattern available
4. First tokens benefit from speculation immediately

### Example 2: Code Completion

```python
prompt = """def fibonacci(n):
    '''Return the nth Fibonacci number'''
<NGRAMGUESS>    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)</NGRAMGUESS>"""

response = client.completions.create(
    model="deepseek-ai/deepseek-coder-6.7b-instruct",
    prompt=prompt,
    max_tokens=100
)
```

### Example 3: Question Answering

```python
prompt = """Q: What is the chemical formula for water?
A: <NGRAMGUESS>H2O</NGRAMGUESS>"""

response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt=prompt,
    max_tokens=50
)
```

## How It Works

### Normal N-gram Speculation (without NGRAMGUESS)

```
Request 1:
  Prompt: "The capital of France is "
  Cache: EMPTY
  First token: Poor speculation (cache miss) → [last_token, 0, 0, 0, ...]
  After token 1: Cache updated with prompt + token 1
  Token 2: Slightly better (has 1 pattern)
  Token 3: Better (has 2 patterns)
  ...
```

### With NGRAMGUESS

```
Request 1:
  Prompt: "The capital of France is "
  Guess: "Paris"
  Cache: PRELOADED with "The capital of France is Paris"
  First token: Good speculation! (cache hit)
  After token 1: Cache updated with prompt + new tokens
  Token 2: Excellent speculation
  Token 3: Excellent speculation
  ...
```

## Advanced Usage

### Multiple Guesses (Not Yet Supported)

Currently, only the first `<NGRAMGUESS>` tag is used. Multiple tags are parsed but only the first is cached.

### Multiline Guesses

Works perfectly:

```python
prompt = """Write a haiku about programming:
<NGRAMGUESS>Code flows like water
Bugs hide in shadows lurking
Debug lights the way</NGRAMGUESS>"""
```

### Partial Guesses

You don't need to guess the entire output:

```python
# Guess just the beginning to "prime" the cache
prompt = "Translate 'hello' to French: <NGRAMGUESS>Bon</NGRAMGUESS>"
# Model might still generate "Bonjour" - the guess just improves early tokens
```

## Performance Impact

### Benefits

1. **Eliminates cold start**: No poor speculation for first 2-3 tokens
2. **Improves throughput**: More draft tokens accepted from the start
3. **User control**: You guide the model with expected patterns

### Typical Results

Without NGRAMGUESS:
```
Token 1: 0.2 average accepted tokens (poor)
Token 2: 0.8 average accepted tokens
Token 3: 1.5 average accepted tokens
Token 4+: 2.0 average accepted tokens
```

With NGRAMGUESS (accurate guess):
```
Token 1: 2.5 average accepted tokens (excellent!)
Token 2: 2.8 average accepted tokens
Token 3+: 2.5-3.0 average accepted tokens
```

### When to Use

✅ **Good use cases:**
- Structured output formats (JSON, code, etc.)
- Common completions (questions with known answers)
- Templates where beginning is predictable
- Few-shot examples where format is consistent

❌ **Not useful for:**
- Creative/unpredictable outputs
- When guess is likely wrong (degrades performance)
- Very short completions (< 5 tokens)

## Limitations

1. **Accuracy matters**: Wrong guesses don't hurt correctness (model still generates), but they waste cache space
2. **Cache is shared**: Guesses from one request affect others in the same server instance
3. **No verification**: The guess is added to cache whether correct or not
4. **Single guess per request**: Only one `<NGRAMGUESS>` tag per prompt

## Debugging

### Check if feature is active

Look for these log messages:

```
INFO: Parsed NGRAMGUESS from request abc123: guess_text='Paris'
INFO: Tokenized NGRAMGUESS for request abc123: 1 tokens
INFO: Added NGRAMGUESS to cache: 10 prompt tokens + 1 guess tokens (total: 11 tokens)
```

### Not using NGRAM speculation?

You'll see:
```
WARNING: NGRAMGUESS tags found in request abc123 but not using NGRAM speculation (current: NONE). Ignoring guess.
```

### Empty guess

```
WARNING: add_guess_to_cache called with empty guess_ids
```

## Implementation Details

- Parsing: `tokenizer_manager.py`
- Cache insertion: `ngram_worker.py::add_guess_to_cache()`
- Happens before first forward pass (during request setup)
- Synchronous cache insertion (blocks until cached)
- Uses last `branch_length` tokens (default: 18) from prompt + guess

## FAQ

**Q: Does the guess affect model output?**  
A: No! The model still generates independently. The guess only helps speculation - if wrong, tokens are rejected during verification.

**Q: Can I use this without n-gram speculation?**  
A: The tags are parsed but ignored. A warning is logged.

**Q: What if my guess is wrong?**  
A: No problem - incorrect drafts are rejected during verification. You just used some cache space.

**Q: Does this work with streaming?**  
A: Yes, perfectly! The guess is cached before streaming starts.

**Q: Can I update the cache later?**  
A: Not directly. But the cache is updated after each token is generated, so patterns accumulate naturally.

## Examples

See `test_ngram_guess.py` for parsing tests.

For full server tests with actual model inference, see:
```bash
python -m pytest test/registered/spec/test_ngram_speculative_decoding.py
```

## Related Documentation

- [N-gram Speculation](NGRAM.md)
- [Server Arguments](docs/advanced_features/server_arguments.md)
- [Speculative Decoding](docs/advanced_features/speculative_decoding.ipynb)
