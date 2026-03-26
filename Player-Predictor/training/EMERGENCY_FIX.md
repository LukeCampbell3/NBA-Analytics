# EMERGENCY FIX: Expert Collapse Root Cause

## The Problem

Your router is producing **extreme confidence** (99.96%) despite all our fixes:
- Load balance loss: 0.9083 (high but ineffective)
- Router entropy: 0.0033 (should be ~2.0)
- Expert 1: 99.96% usage
- All other experts: <0.001%

## Root Cause

The issue is in the **router logit computation**:

```python
# Current code (BROKEN):
query_norm = tf.nn.l2_normalize(router_query, axis=-1)
keys_norm = tf.nn.l2_normalize(expert_keys_matrix, axis=-1)
router_logits = tf.matmul(query_norm, keys_norm, transpose_b=True)  # [B, E]
router_logits_scaled = router_logits / temperature  # Still too confident!
```

**Why this fails**:
1. L2-normalized dot product gives values in [-1, 1]
2. When one expert key is similar to query, you get ~0.9
3. Even with temperature=5.0: 0.9/5.0 = 0.18
4. After softmax: exp(0.18) / sum(...) ≈ 0.99+ (because other logits are much lower)
5. Load balance loss can't overcome this massive confidence gap

## The Fix: Add Gumbel Noise + Reduce Logit Scale

### Option 1: Gumbel-Softmax (RECOMMENDED)

Replace the router with Gumbel-Softmax for exploration:

```python
# In build_model_with_phase2(), replace router section:

# Compute logits (keep as is)
query_norm = tf.nn.l2_normalize(router_query, axis=-1)
keys_norm = tf.nn.l2_normalize(expert_keys_matrix, axis=-1)
router_logits = tf.matmul(query_norm, keys_norm, transpose_b=True)  # [B, E]

# Scale down logits BEFORE temperature
router_logits = router_logits * 0.1  # NEW: Reduce magnitude 10x

# Apply temperature
temperature = self.config.get("router_temperature", 5.0)
router_logits_scaled = router_logits / temperature

# Add Gumbel noise during training (NEW)
def add_gumbel_noise(logits, training):
    if training:
        # Gumbel noise: -log(-log(uniform))
        uniform = tf.random.uniform(tf.shape(logits), minval=1e-10, maxval=1.0)
        gumbel = -tf.math.log(-tf.math.log(uniform))
        return logits + gumbel * 0.5  # Scale noise by 0.5
    return logits

adjusted_router_logits = Lambda(
    lambda x: add_gumbel_noise(x[0] + x[1], x[2]),
    name="router_with_gumbel"
)([router_logits_scaled, spike_routing_bias, training_flag])

# Rest of code continues...
router_probs = Softmax(name="router_probs")(adjusted_router_logits)
```

### Option 2: Explicit Entropy Bonus (SIMPLER)

Add a strong entropy bonus directly to the loss:

```python
# In the loss computation (around line 800-900):

# Current load balance loss
load_balance_loss = ...

# NEW: Add explicit entropy maximization
router_entropy = -tf.reduce_sum(router_probs * tf.math.log(router_probs + 1e-10), axis=-1)
target_entropy = tf.math.log(float(total_experts)) * 0.9  # 90% of max entropy
entropy_bonus = tf.reduce_mean(tf.nn.relu(target_entropy - router_entropy))

# Combine with MUCH HIGHER weight
entropy_weight = 5.0  # Very high!
total_loss = ... + (entropy_weight * entropy_bonus)
```

### Option 3: Top-K Routing (NUCLEAR OPTION)

Force the router to use multiple experts:

```python
# Replace softmax routing with top-k:

# Get top-k experts (e.g., k=3)
k = 3
top_k_logits, top_k_indices = tf.nn.top_k(adjusted_router_logits, k=k)

# Softmax only over top-k
top_k_probs = tf.nn.softmax(top_k_logits / 0.5)  # Lower temp for top-k

# Scatter back to full expert dimension
router_probs = tf.scatter_nd(
    indices=tf.expand_dims(top_k_indices, -1),
    updates=top_k_probs,
    shape=[batch_size, total_experts]
)
```

## Immediate Action

### Quick Test: Increase Entropy Weight 100x

Edit `training/integrate_moe_improvements.py` around line 850-900:

Find this section:
```python
# Entropy regularization
if self.config.get("use_entropy_regularization", True):
    entropy_weight = 0.001  # CURRENT VALUE
```

Change to:
```python
# Entropy regularization (EMERGENCY FIX)
if self.config.get("use_entropy_regularization", True):
    entropy_weight = 1.0  # INCREASED 1000x!
    target_entropy = tf.math.log(float(total_experts)) * 0.85
    current_entropy = -tf.reduce_sum(router_probs * tf.math.log(router_probs + 1e-10), axis=-1)
    entropy_penalty = tf.reduce_mean(tf.nn.relu(target_entropy - current_entropy))
    aux_losses.append(entropy_weight * entropy_penalty)
```

### Alternative: Scale Down Logits

Find this line (around 680):
```python
router_logits = tf.matmul(query_norm, keys_norm, transpose_b=True)
```

Add immediately after:
```python
router_logits = router_logits * 0.05  # Scale down 20x!
```

## Why Load Balance Loss Fails

The load balance loss computes:
```
L_balance = sum(expert_usage^2)
```

When expert 1 has 99.96% usage:
```
L_balance = 0.9996^2 + 10*(0.0004^2) ≈ 0.9992
```

This is only 0.0008 different from perfect balance (1/11 = 0.0909 per expert):
```
L_perfect = 11 * (0.0909^2) ≈ 0.0909
```

So the penalty is: 0.9992 - 0.0909 = 0.9083 (what you see!)

But this 0.9083 penalty is TINY compared to the routing confidence gain of picking expert 1 with 99.96% confidence.

## The Real Solution

You need to either:
1. **Reduce routing confidence** (Gumbel noise, lower logit scale)
2. **Increase entropy penalty** (1000x higher weight)
3. **Force diversity** (top-k routing, explicit constraints)

The current approach of "balance the usage" doesn't work because the router has already decided expert 1 is best, and the penalty for using it isn't strong enough to change that decision.
