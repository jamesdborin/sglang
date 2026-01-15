# N-gram Speculative Decoding in SGLang

This document explains the n-gram speculation implementation in SGLang, focusing on how text flows through the system and how draft tokens are generated at runtime.

## Quick Answer: Cache Population

**Q: Where do input tokens get added into the ngram cache before generation begins?**

**A: They don't!** The n-gram cache starts **completely empty** for each new request. Input tokens are NOT pre-populated. Instead:

1. **During prefill (EXTEND mode)**: Cache is not updated at all - speculation is skipped
2. **After first decoded token**: Cache is updated with last 18 tokens from `origin_input_ids + output_ids`  
3. **After each subsequent token**: Cache continues to grow with newly generated patterns
4. **Empty cache behavior**: Returns `[last_input_token, 0, 0, 0, ...]` (zero-padded)

This means the first few tokens have poor speculation (cache misses), but quality improves as the cache learns patterns from the ongoing generation. The cache is shared across all requests in the server, so patterns accumulate over the lifetime of the process.

See [Cache Initialization and Population](#cache-initialization-and-population) for detailed explanation.

---

## Overview

N-gram speculative decoding accelerates inference by predicting multiple tokens at once using cached n-gram patterns, then verifying them in parallel with the target model. This avoids sequential token-by-token generation.

**Key benefit**: Generate multiple draft tokens from cache (cheap) → Verify all at once with target model (one forward pass instead of N).

---

## Architecture

### Core Files

1. **`python/sglang/srt/speculative/ngram_worker.py`**
   - Main orchestrator for n-gram speculation
   - Prepares draft tokens from cache
   - Verifies predictions with target model
   - Updates cache with new sequences

2. **`python/sglang/srt/speculative/ngram_info.py`**
   - `NgramVerifyInput` class for verification
   - Greedy and sampling verification methods
   - Token acceptance logic using tree-based speculation
   - KV cache management

3. **`python/sglang/srt/speculative/cpp_ngram/`**
   - `ngram.cpp`: Core C++ trie-based cache implementation
   - `ngram_cache.py`: Python wrapper
   - `ngram_cache_binding.cpp`: PyBind11 bindings
   - High-performance n-gram storage and matching

4. **`python/sglang/srt/managers/scheduler.py`**
   - Creates and manages the NGRAMWorker
   - Routes batches to the worker for processing

---

## Data Flow: Text to Draft Tokens

### 1. User Request → Tokenizer → Scheduler

```
User text: "The quick brown fox"
    ↓
Tokenizer
    ↓
Token IDs: [464, 2068, 7586, 25341]
    ↓
Stored in Req object:
  - req.origin_input_ids: [464, 2068, 7586, 25341]  # Original prompt
  - req.output_ids: []                               # Generated tokens (empty initially)
```

### 2. Scheduler → NGRAMWorker

During initialization (`scheduler.py`, line 517):
```python
DraftWorkerClass = self.spec_algorithm.create_worker(self.server_args)
self.draft_worker = DraftWorkerClass(...)
self.model_worker = self.draft_worker  # When using NGRAM
```

During inference (`scheduler.py`, lines 2191, 2231):
```python
batch_result = self.model_worker.forward_batch_generation(batch)
```

The `batch` contains multiple `Req` objects, each with:
- `req.origin_input_ids`: Original prompt tokens (List[int])
- `req.output_ids`: Generated tokens so far (List[int])
- Full sequence = `origin_input_ids + output_ids`

---

## Cache Initialization and Population

### When Does the Cache Get Populated?

**The n-gram cache does NOT contain input tokens before generation begins.** This is a key design characteristic:

#### Cache Lifecycle

1. **Initialization (Server Start)**
   ```python
   # ngram_worker.py, line 50
   self.ngram_cache = NgramCache(
       capacity=1_000_000,  # Maximum cache entries
       ...
   )
   # Cache is EMPTY - just an empty trie structure
   ```

2. **Prefill Phase (EXTEND mode)**
   - `forward_mode.is_extend() = True`
   - Speculation is **skipped** (line 143: `return`)
   - Cache is **NOT updated**
   - First token generated normally

3. **First Decode Token**
   - Cache lookup happens but **cache is empty**
   - `matchBFS()` returns: `[last_token, 0, 0, 0, 0, 0, 0, 0]`
   - Target model verifies and generates real token
   - Cache **IS updated** for first time (line 235)
   - Stores: last 18 tokens from `origin_input_ids + output_ids`

4. **Subsequent Tokens**
   - Cache now has patterns from previous tokens
   - Speculation quality improves progressively
   - Cache continues to grow with each verified token

#### What Gets Cached?

From `_update_ngram_cache()` (line 207-210):
```python
put_ids = self._efficient_concat_last_n(
    req.origin_input_ids,  # Original prompt
    req.output_ids,        # All generated tokens so far
    self.branch_length     # Default: 18 tokens
)
```

**Cached sequences include**:
- Tail of input prompt (if needed to reach 18 tokens)
- All generated output tokens
- Updated after EVERY verification step
- Stored as n-grams in trie structure for fast lookup

#### Cache Sharing

The cache is **shared across**:
- All requests in the same worker process
- Different positions within the same request
- Different requests to the same server instance

This means:
- Request A's patterns can help Request B
- Common phrases/patterns accumulate over time
- Long-running servers have richer caches
- Cache persists until `clear_cache_pool()` is called or server restarts

#### Why Not Pre-populate?

The commented code (line 202-205) shows this was considered:
```python
# FIXME: Whether to insert 'extend' into the cache or not, after testing,
# there is not much difference, so we will not insert it for now.
# if batch.forward_mode.is_extend():
#     put_ids = req.origin_input_ids + req.output_ids
```

**Reasons not to pre-populate**:
- Testing showed "not much difference" in accuracy
- Prefill is already efficient (processes all tokens in parallel)
- Speculation is most valuable for decode (sequential token generation)
- Simpler implementation without special-casing prefill

#### Handling Empty Cache

When cache is empty or has no matches (`ngram.cpp`, line 22-61):
```cpp
// fillResult always returns draft_token_num tokens
info.token.emplace_back(last_token);  // First token = last input token

// Try to add children from cache (none if empty)
for (auto [token, next] : tree[root].next) {
    queue.emplace(token, next, 0);
}

// Pad with zeros to reach draft_token_num
while (info.token.size() < draft_token_num) {
    info.token.emplace_back(0);  // Zero padding
}
```

**Result**: First token uses `last_token` (from input), rest are zeros. The model's verification step will generate the real token, and that gets cached for next iteration.

---

## Runtime Logic: How Draft Tokens Are Generated

### Entry Point: `forward_batch_generation()`

Location: `ngram_worker.py`, line 213

```python
def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
    # Step 1: Prepare draft tokens
    self._prepare_for_speculative_decoding(batch)
    
    # Step 2: Verify with target model
    model_worker_batch = batch.get_model_worker_batch()
    batch_result = self.target_worker.forward_batch_generation(
        model_worker_batch, is_verify=True
    )
    
    # Step 3: Accept matching tokens
    verify_input.verify(batch, logits_output, self.page_size)
    
    # Step 4: Update cache with new sequences
    self._update_ngram_cache(batch)
    
    return batch_result
```

---

### Step 1: Prepare Draft Tokens

**Method**: `_prepare_for_speculative_decoding()` (line 142)

**Key behavior**: Only runs in DECODE mode (skips prefill/EXTEND mode)

```python
def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
    if batch.forward_mode.is_extend():
        return  # Skip during prefill
    
    # Get draft tokens from cache
    req_drafts, mask = self._prepare_draft_tokens(batch)
    
    # Build tree structures for parallel verification
    reconstruct_indices_from_tree_mask(...)
    
    # Store in batch for verification
    batch.spec_info = NgramVerifyInput(draft_tokens, tree_mask, ...)
```

---

### Step 2: Extract Recent Token History

**Method**: `_prepare_draft_tokens()` (line 121)

```python
def _prepare_draft_tokens(self, batch: ScheduleBatch) -> tuple[np.ndarray, np.ndarray]:
    self.ngram_cache.synchronize()
    batch_tokens = []
    
    for req in batch.reqs:
        # Extract last N tokens from the sequence
        check_token = self._efficient_concat_last_n(
            req.origin_input_ids,      # Original prompt
            req.output_ids,             # Generated tokens
            self.max_match_window_size  # Default: 10 tokens
        )
        batch_tokens.append(check_token)
    
    # Query cache for continuations
    req_drafts, mask = self.ngram_cache.batch_get(batch_tokens)
    return req_drafts, mask
```

**Helper**: `_efficient_concat_last_n()` (line 63)

Efficiently extracts the last N tokens from a sequence:
- If `output_ids` has ≥N tokens → return `output_ids[-N:]`
- Otherwise → return `origin_input_ids[-(N-len(output_ids)):] + output_ids`

**Example**:
```python
origin_input_ids = [1, 2, 3, 4, 5]
output_ids = [6, 7, 8, 9, 10, 11, 12]
max_match_window_size = 10

# output_ids has 7 tokens, need 3 more from origin_input_ids
check_token = [3, 4, 5] + [6, 7, 8, 9, 10, 11, 12]  # Last 10 tokens
```

---

### Step 3: Query N-gram Cache

**Method**: `ngram_cache.batch_get()` (C++ implementation)

This is the **core speculation logic**:

1. **Cache Structure**: Trie-based data structure storing token sequences
2. **Lookup**: For each request's recent tokens (e.g., last 10), find matching continuations
3. **BFS Tree Building**: Uses Breadth-First Search to create a speculation tree with multiple branches

**Example**:
```
Recent tokens: [1, 2, 3]

Cache contains:
  1 → 2 → 3 → 4 → 5 → 6
  1 → 2 → 3 → 4 → 7 → 8
  1 → 2 → 3 → 9 → 10

Returns speculation tree:
    3
   ↙ ↘
  4   9
 ↙ ↘  ↓
5  7  10
↓  ↓
6  8
```

**Returns**:
- `req_drafts`: Draft token IDs (shape: `[batch_size * draft_token_num]`)
  - Example: `[4, 5, 6, 7, 8, 9, 10, 0]` (padded to draft_token_num=8)
  - **If cache is empty/no matches**: Returns `[last_token, 0, 0, 0, 0, 0, 0, 0]` (zero padding)
- `mask`: Tree attention mask (shape: `[bs, draft_token_num, draft_token_num]`)
  - Binary matrix showing which draft tokens can attend to which
  - Encodes the tree structure for parallel processing

**Important**: The cache starts empty and is populated during generation. Early tokens will have cache misses and return mostly zero-padded results. Speculation improves as more tokens are generated and cached.

---

### Step 4: Build Verification Structures

**Method**: `reconstruct_indices_from_tree_mask()` (line 159)

Converts the tree mask into efficient traversal indices:

```python
reconstruct_indices_from_tree_mask(
    tree_mask,              # Tree structure
    batch.seq_lens,         # Current sequence lengths
    positions,              # Position IDs for each draft token (output)
    retrive_index,          # Tree traversal indices (output)
    retrive_next_token,     # Next token in tree path (output)
    retrive_next_sibling,   # Next sibling in tree (output)
    bs,
    self.draft_token_num,
)
```

**Creates attention mask** (lines 172-184):
```python
for i, req in enumerate(batch.reqs):
    seq_len = len(req.origin_input_ids) + len(req.output_ids)
    # Draft tokens can attend to:
    # 1. All previous tokens in sequence (seq_len - 1)
    # 2. Draft tokens following tree structure
    req_mask = torch.ones((draft_token_num, seq_len - 1))
    req_mask = torch.cat((req_mask, tree_mask[i]), dim=1)
```

**Stores in batch** (line 188):
```python
batch.spec_info = NgramVerifyInput(
    draft_tokens,         # The speculated tokens
    tree_mask,           # Attention mask for tree
    positions,           # Position IDs
    retrive_index,       # Tree traversal indices
    retrive_next_token,  # Next token pointers
    retrive_next_sibling # Sibling pointers
)
```

---

### Step 5: Verify with Target Model

**Method**: `target_worker.forward_batch_generation()` (line 220)

```python
# Forward all draft tokens through target model in one pass
batch_result = self.target_worker.forward_batch_generation(
    model_worker_batch, is_verify=True
)

# Get target model predictions
logits_output = batch_result.logits_output
```

**Verification** (`ngram_info.py`, line 374):

```python
def verify(self, batch, logits_output, page_size):
    # Apply penalties and custom processors
    sampling_info.apply_logits_bias(logits_output)
    
    # Verify tokens (greedy or sampling)
    if sampling_info.is_all_greedy:
        self._greedy_verify(batch, logits_output)
    else:
        self._sampling_verify(batch, logits_output, sampling_info)
    
    # Fill accepted tokens into requests
    self._fill_requests(batch, logits_output)
    
    # Free KV cache for rejected tokens
    self._free_cache(batch, page_size, accept_length_cpu)
    
    return logits_output, verified_ids, num_accepted_tokens
```

**Greedy Verification** (`_greedy_verify`, line 277):
```python
# Get target model predictions
target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
target_predict = target_predict.reshape(bs, self.draft_token_num)

# Compare with draft tokens
candidates = self.draft_token.reshape(bs, self.draft_token_num)

# Use C++ kernel to find longest matching prefix in tree
verify_tree_greedy(
    predicts=self.predict,           # Output: accepted tokens
    accept_index=self.accepted_indices,  # Output: which drafts accepted
    accept_token_num=self.accept_length, # Output: acceptance length
    candidates=candidates,            # Input: draft tokens
    retrive_index=self.retrive_index, # Input: tree structure
    retrive_next_token=...,          # Input: tree traversal
    retrive_next_sibling=...,        # Input: tree traversal
    target_predict=target_predict,   # Input: target predictions
)
```

**Acceptance Logic**:
- Traverse the speculation tree
- At each node, check if target prediction matches draft token
- If match: accept and continue to children
- If mismatch: stop and use target prediction
- Returns longest matching path per request

---

### Step 6: Update Cache

**Method**: `_update_ngram_cache()` (line 199)

**CRITICAL**: This is called ONLY in the `TARGET_VERIFY` branch (line 235), not during prefill!

```python
def _update_ngram_cache(self, batch: ScheduleBatch):
    batch_tokens = []
    for req in batch.reqs:
        # Store last `branch_length` tokens (default: 18)
        put_ids = self._efficient_concat_last_n(
            req.origin_input_ids, 
            req.output_ids, 
            self.branch_length  # Longer than match window
        )
        batch_tokens.append(put_ids)
    
    # Async insert into cache (non-blocking)
    self.ngram_cache.batch_put(batch_tokens)
```

**When cache is updated**:
1. **NOT during prefill (EXTEND mode)**: Line 143-144 returns early
2. **NOT on first decode token**: First decode generates one token, then enters TARGET_VERIFY
3. **Only after TARGET_VERIFY**: Cache updated with prompt + all generated tokens so far
4. **Every subsequent decode step**: Cache updated incrementally

**Why this matters**:
- Cache is built incrementally from the sequence being generated
- First ~2-3 tokens have poor speculation (empty cache)
- Later tokens benefit from patterns established earlier in the sequence
- Cache accumulates patterns across all requests, improving over time
- **This is NOT a pre-built cache** - it starts empty and learns during inference

---

## Configuration Parameters

From `server_args.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--speculative-algorithm` | - | Set to `NGRAM` to enable |
| `--speculative-num-draft-tokens` | 8 | Number of tokens to draft per request |
| `--speculative-ngram-branch-length` | 18 | Length of sequences stored in cache |
| `--speculative-ngram-min-match-window-size` | 1 | Minimum tokens to match for lookup |
| `--speculative-ngram-max-match-window-size` | 10 | Maximum tokens to use for lookup |
| `--speculative-ngram-min-bfs-breadth` | 1 | Minimum branches in speculation tree |
| `--speculative-ngram-max-bfs-breadth` | 8 | Maximum branches in speculation tree |
| `--speculative-ngram-capacity` | 1,000,000 | Maximum cache entries |

---

## Key Insights

### 1. **No Text is Passed**
Only token IDs flow through the system (`origin_input_ids + output_ids`). The ngram worker never sees raw text.

### 2. **Tree-Based Speculation**
Unlike simple n-gram lookup that returns one sequence, this implementation builds a **tree of possible continuations** with multiple branches, allowing parallel exploration of multiple hypotheses.

### 3. **DECODE Mode Only**
Speculation only runs during the decode phase (token-by-token generation). During prefill (EXTEND mode), it's skipped because the full prompt is processed at once anyway.

### 4. **Cache is Populated During Generation**
**The n-gram cache starts EMPTY** for each new request. It is populated incrementally:
- During prefill (EXTEND mode): Cache is NOT updated, speculation is skipped
- During decode (first token): Cache is empty, returns last_token + zero padding
- After verification: Cache is updated with `origin_input_ids + output_ids` (line 235)
- Subsequent tokens: Cache grows and provides better speculation

This means:
- **First few tokens have poor speculation** (cache miss → zero padding)
- **Later tokens benefit from accumulated patterns** from earlier in the same sequence
- **Cache is shared across requests in the same batch/session**, so patterns from one request can help others

### 5. **Verification is Parallel**
All draft tokens (across all branches) are verified in a single forward pass through the target model using tree attention masks.

### 6. **Cache Updates Continuously**
The cache learns from every generated sequence, accumulating patterns that improve future speculation accuracy. Updates happen ONLY after verification in TARGET_VERIFY mode (line 235), not during prefill.

### 7. **Efficient Memory Management**
- Preallocates tensors for maximum batch size (line 71)
- Uses page-aligned KV cache slots
- Frees cache for rejected draft tokens immediately

---

## Performance Characteristics

**Speedup Factors**:
1. **Cache Hit Rate**: Higher hit rates → more accurate drafts → more accepted tokens
2. **Accept Length**: Longer accepted sequences → fewer target model calls
3. **Tree Breadth**: More branches → higher chance of finding correct path
4. **Batch Size**: Larger batches amortize overhead

**Typical Results** (from tests):
- GSM8K accuracy: ~79% (same as without speculation)
- Average accept length: ~1.8-2.5 tokens per verification
- Works with multiple attention backends (FlashAttention, Triton, FlashInfer)

---

## Example Execution Trace

```
Request: "The capital of France is"
Tokens: [464, 3139, 286, 4881, 318]

Prefill Phase (EXTEND mode):
  - forward_mode.is_extend() = True
  - Speculation skipped (line 143-144)
  - Process full prompt, generate first token: 2547 ("Paris")
  - output_ids: [2547]
  - Cache: EMPTY (not updated during prefill)

Generation Step 1 (First DECODE):
  - output_ids: [2547]
  - check_token: [286, 4881, 318, 2547] (last 10, or available)
  - Cache lookup: EMPTY CACHE → returns [2547, 0, 0, 0, 0, 0, 0, 0]
  - Speculation has mostly zeros (poor quality)
  - Verify: target model generates "."
  - output_ids: [2547, 13]
  - Cache updated: stores [464, 3139, 286, 4881, 318, 2547, 13] (last 18 tokens)

Generation Step 2:
  - output_ids: [2547, 13]
  - check_token: [4881, 318, 2547, 13] (last 10)
  - Cache lookup: Finds partial match from step 1
  - Returns some speculation (quality improving)
  - Verify and accept tokens
  - output_ids: [2547, 13, 383, ...]
  - Cache updated: stores [...318, 2547, 13, 383, ...] (last 18 tokens)

Generation Step 3 and beyond:
  - Cache grows richer with patterns
  - Speculation quality improves
  - More draft tokens accepted
  - Cache learns common continuations specific to this sequence

... continues until finished
```

**Key Observation**: The first few tokens have poor speculation because the cache is empty. Speculation quality improves as the cache accumulates patterns from the current generation sequence.

---

## Testing

Tests located in: `test/registered/spec/test_ngram_speculative_decoding.py`

Run tests with different backends:
```bash
# FlashAttention 3
python -m pytest test/registered/spec/test_ngram_speculative_decoding.py::TestNgramSpeculativeDecodingBase

# Triton
python -m pytest test/registered/spec/test_ngram_speculative_decoding.py::TestNgramSpeculativeDecodingTriton

# FlashInfer
python -m pytest test/registered/spec/test_ngram_speculative_decoding.py::TestNgramSpeculativeDecodingFlashinfer

# Paged attention
python -m pytest test/registered/spec/test_ngram_speculative_decoding.py::TestNgramSpeculativeDecodingPaged
```

---

## Related Documentation

- [Speculative Decoding Tutorial](docs/advanced_features/speculative_decoding.ipynb)
- [Server Arguments](docs/advanced_features/server_arguments.md)
- EAGLE speculation: `python/sglang/srt/speculative/eagle_worker.py`
- Standalone speculation: `python/sglang/srt/speculative/standalone_worker.py`
