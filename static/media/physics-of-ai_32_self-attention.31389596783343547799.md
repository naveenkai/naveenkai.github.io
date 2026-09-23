# Physics of AI — Part VI · Attention

## 32. Self-Attention

> *In Section 31 the decoder looked back at the encoder: one sequence attending to another. But the encoder itself was still an LSTM, passing information along a leaky chain, one position at a time. This section asks the question that led to the Transformer: what if every token in a sequence attends to **every other token in the same sequence**, all at once? No recurrence, no chain, one matrix multiply. We'll build it from scratch, explain the $\sqrt{d}$ in the famous formula by measuring it, and pit it against a bag and an LSTM on a task that needs tokens to find each other.*

---

### 32.1 Queries, keys and values

In Section 31 the decoder state was the **query** ("what do I need?"), and the encoder states were both the **keys** (what gets matched) and the **values** (what gets read). Self-attention gives **every token all three roles**, through three learned linear maps (Section 5):

$$
\mathbf{q}_i = W_Q\,\mathbf{x}_i \qquad \mathbf{k}_i = W_K\,\mathbf{x}_i \qquad \mathbf{v}_i = W_V\,\mathbf{x}_i
$$

- The **query** $\mathbf{q}_i$ is what token $i$ is looking for.
- The **key** $\mathbf{k}_j$ is what token $j$ advertises about itself.
- The **value** $\mathbf{v}_j$ is what token $j$ hands over if someone attends to it.

Why three separate maps instead of using $\mathbf{x}$ directly? Because *what I'm looking for* is usually different from *what I am*. The word "it" is looking for a noun, but it isn't a noun itself. Separate maps let the network learn those roles independently.

Then every token attends to every token, all at once, as matrices:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

This is the central equation of *"Attention Is All You Need"* (Vaswani et al., 2017). With $T$ tokens, $QK^\top$ is a $T \times T$ table of scores: row $i$ says how well token $i$'s query matches every key. The softmax turns each row into weights, and multiplying by $V$ gives each token a weighted blend of everyone's values.

![Self-attention on four tokens: queries, keys and values, the attention matrix, and one output](figures/fig145_qkv.png)

(The numbers in the diagram are illustrative.)

From scratch it's eight lines:

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_k, bias=False)
        self.Wk = nn.Linear(d_model, d_k, bias=False)
        self.Wv = nn.Linear(d_model, d_k, bias=False)
        self.d_k = d_k
    def forward(self, X):                                     # X: (batch, T, d_model)
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)          # every token: ask, advertise, offer
        scores = Q @ K.transpose(1, 2) / math.sqrt(self.d_k)  # (batch, T, T): every token vs every token
        A = scores.softmax(-1)                                # row i: where token i looks
        return A @ V                                          # each token: a blend of values
```

It matches PyTorch's `F.scaled_dot_product_attention` to $9\times10^{-8}$, and a one-head `nn.MultiheadAttention` exactly.

There is **no loop over positions**. Token 1 and token 1,000 are processed in the same matrix multiply, and every token is **one step** from every other. Compare the RNN, where information from token 1 reaches token 1,000 only after 999 sequential updates.

---

### 32.2 Why divide by $\sqrt{d}$?

The scaling factor looks like a detail, but it's Section 15's lesson about keeping signals the right size.

If the entries of $\mathbf{q}$ and $\mathbf{k}$ are roughly independent with variance 1, their dot product $\mathbf{q}\cdot\mathbf{k} = \sum_{i=1}^{d} q_i k_i$ is a sum of $d$ terms, so its variance is about $d$ and its typical size is $\sqrt{d}$. Measured, with random unit-variance vectors and 20 keys per query:

![Unscaled vs scaled scores: their size, the resulting top weight, and the gradient through the softmax](figures/fig146_sqrt_d.png)

| dimension $d$ | score size, unscaled | top attention weight, unscaled | **scaled by $1/\sqrt{d}$** |
|---|---|---|---|
| 4 | 2.0 | 0.40 | score size 1.0, top weight 0.21 |
| 64 | 8.1 | 0.84 | score size 1.0, top weight 0.22 |
| 1024 | 32.0 | **0.96** | score size 1.0, top weight 0.22 |

Without scaling, larger $d$ means larger scores, and a softmax of large numbers is nearly a **hard max** (Section 8.3): one key gets almost all the weight, *before any learning has happened*. That's bad for two reasons:

1. **The network is committed before it has learned anything.** Random initial vectors decide who attends to whom.
2. **The gradient dies.** A saturated softmax is flat, just like a saturated sigmoid (Section 6). At $d = 1024$, the gradient through the unscaled softmax is **13× smaller** than through the scaled one ($3.4\times10^{-4}$ vs $4.5\times10^{-3}$).

Dividing by $\sqrt{d_k}$ keeps the scores at size about 1 whatever the dimension, so attention starts **soft and trainable**. It's He initialisation (Section 15) applied to attention scores.

> 📓 **Notebook rule:** *a dot product grows like $\sqrt{d}$, so divide by $\sqrt{d}$.* Otherwise the softmax saturates into a hard choice before training starts, and its gradient vanishes.

---

### 32.3 A task that needs tokens to find each other

**Associative recall.** The input is $N$ key→value pairs followed by a query key, for example:

```
10→7   0→7   7→9   4→8   13→4   8→2   15→12   11→6   query: 15      answer: 12
```

The answer is the value stored with the queried key. Keys and values are drawn from 16 symbols, with fresh random examples every batch. It's the simplest version of what reading comprehension needs: *find the relevant piece of the input, and read what's attached to it.* Each pair is one token (key embedding + value embedding), and the query is the last token. Three models, same training (3,000 steps):

- **Bag:** the mean of the pair tokens, plus the query token, fed to an MLP (Sections 26–27).
- **LSTM:** reads the pairs and then the query in order (Section 29).
- **Self-attention:** **one** layer. The output at the query position, together with the query token itself, goes to a linear layer.

![Associative recall: accuracy by model and number of pairs; where the query token attends](figures/fig147_recall.png)

| test accuracy (chance 6.25%) | 4 pairs | 8 pairs | 16 pairs | weights |
|---|---|---|---|---|
| bag + MLP | 33.5% | 24.9% | 18.3% | 106,000 |
| LSTM | 99.6% | 96.3% | 83.5% | 104,464 |
| **one self-attention layer** | **100%** | **100%** | **100%** | **17,424** |

- **The bag fails.** Averaging the pairs mixes every key with every value, so the model can't tell which value belonged to which key. (It beats chance only by guessing one of the values that are present.)
- **The LSTM does well, but degrades as the list grows**: it has to hold all 16 pairs in its state and search that memory at the end, the bottleneck of Section 30 again.
- **Self-attention is perfect at every size, with a sixth of the weights.** The right panel shows why. The query token puts weight **1.00** on the single pair whose key matches, and 0 on everything else. The query's $\mathbf{q}$ learned to match the right pair's $\mathbf{k}$, and that pair's $\mathbf{v}$ carries the answer. That's exactly the query/key/value story of 32.1, discovered by gradient descent.

---

### 32.4 The catch: self-attention has no idea about order

Shuffle the input tokens and see what happens to the outputs:

```python
perm = torch.randperm(5)
(sa(X[:, perm]) - sa(X)[:, perm]).abs().max()     # 1.2e-07: shuffling the input just shuffles the output
```

Self-attention is **permutation-equivariant**: move the tokens around and each token's output is unchanged, just in a new place. Every token looks at every other through content alone (queries against keys), and nothing in the calculation knows *where* a token sits.

For associative recall that's a feature, since the pairs have no meaningful order. For language it's fatal. "Dog bites man" and "man bites dog" contain the same three tokens, and pure self-attention treats them identically. It's Section 26's bag of words, just with much smarter mixing. Section 31's model lost its place on long inputs for the same reason. Section 33 fixes this with **positional encodings**.

---

### 32.5 The other catch: every pair costs something

Every token attends to every token, so the score matrix has $T^2$ entries. Timing one layer (width 128, batch 16, forward + backward) against an LSTM on our 2 CPUs:

![Time vs sequence length: LSTM vs self-attention](figures/fig148_cost.png)

| sequence length $T$ | LSTM | self-attention | attention matrices in memory |
|---|---|---|---|
| 64 | 8 ms | **3 ms** | 0.3 MB |
| 512 | 68 ms | 64 ms | 17 MB |
| 1,024 | **157 ms** | 303 ms | 67 MB |
| 2,048 | **323 ms** | 1,449 ms | 268 MB |

On a CPU, attention wins for short sequences and **loses** beyond about 512 tokens: the LSTM's cost grows like $T$, attention's like $T^2$. So why did attention take over?

- **Parallelism.** The LSTM's $T$ steps must run one after another. Attention's $T^2$ scores are one big matrix multiply, exactly what GPUs are built for. On a GPU with thousands of cores, the "one step, all positions" structure wins by a huge margin for the lengths used in practice.
- **Path length.** Every token is one hop from every other, so information and gradients don't leak along a chain (Sections 28–31).

The $T^2$ cost is real, though, and it's why "long context" is still an active research area, with work on memory-efficient kernels such as FlashAttention and on sparse and linear attention.

---

### 📓 Notebook margin: the equation so far

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
\qquad
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

| idea | what we now know |
|---|---|
| self-attention | every token queries every token; one matrix multiply; no recurrence |
| query / key / value | three learned roles: what I seek, what I advertise, what I hand over |
| $\sqrt{d}$ | unscaled scores grow like $\sqrt{d}$ (32 at d = 1024) → saturated softmax, 13× smaller gradient |
| associative recall | one attention layer 100% at 16 pairs (LSTM 83.5%, bag 18.3%), 1/6 the weights |
| order | permutation-equivariant: shuffle in, shuffle out, so it's a bag until we add positions |
| cost | $T^2$: faster than an LSTM at 64 tokens, 4.5× slower at 2,048 on a CPU; wins on GPUs by parallelism |

---

### What comes next

Two things are missing before this becomes a Transformer layer:

1. **Position.** Self-attention can't tell "dog bites man" from "man bites dog". We need to stamp each token with where it is.
2. **More than one question at a time.** One head computes one attention pattern per token. But a word may need to look at its subject, its object and the previous word, all at once.

**Section 33: Many Heads and Positions** adds both: **positional encodings** (the sine-and-cosine waves from the paper, and learned ones), **multi-head attention**, and the **causal mask** that stops a token from looking at the future, which is what lets a Transformer generate text one word at a time. We'll test each piece on tasks that need exactly it, including Section 31's "lost its place" failure.

---

*References: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan Gomez, Łukasz Kaiser & Illia Polosukhin (2017), "Attention Is All You Need" (scaled dot-product attention; the $\sqrt{d_k}$ scaling and its variance argument; queries, keys and values). Jianpeng Cheng, Li Dong & Mirella Lapata (2016), "Long Short-Term Memory-Networks for Machine Reading" (intra-attention, an early form of self-attention). Jimmy Ba et al. (2016), "Using Fast Weights to Attend to the Recent Past" (associative recall as a test of memory). Tri Dao et al. (2022), "FlashAttention". François Chollet, *Deep Learning with Python*, 3rd ed., ch. 15 (self-attention; queries, keys and values; the Transformer encoder). All code in this series is PyTorch.*
