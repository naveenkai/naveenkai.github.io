# Physics of AI — Part VI · Attention

## 33. Many Heads and Positions

> *Section 32's self-attention could match any token with any other, but it had two blind spots. It had no idea **where** a token is, since shuffling the input just shuffled the output, and each token could compute only **one** attention pattern. This section adds the three pieces that turn it into the attention of *"Attention Is All You Need"*: **positional encodings**, **multiple heads**, and the **causal mask** that lets a model generate text. Each one gets a task that fails without it.*

---

### 33.1 Telling tokens where they are

The fix is almost embarrassingly simple: before the first layer, **add a position vector to each token's embedding**.

$$
\mathbf{x}_i = \underbrace{E[\text{token}_i]}_{\text{what}} + \underbrace{\mathbf{p}_i}_{\text{where}}
$$

Now "dog" at position 1 and "dog" at position 3 are different vectors, so queries and keys can match on position as well as content ("attend to the token just before me"). There are two standard choices for $\mathbf{p}_i$.

**Learned** position embeddings are one trainable row per position, exactly like word embeddings (Section 27): `nn.Embedding(max_len, d)`.

**Sinusoidal** encodings are the original paper's fixed choice. Each dimension is a wave with its own wavelength, from a few positions up to thousands:

$$
p_{i,\,2k} = \sin\!\left(\frac{i}{10000^{2k/d}}\right), \qquad p_{i,\,2k+1} = \cos\!\left(\frac{i}{10000^{2k/d}}\right)
$$

```python
def sinusoidal(T, d):
    pos = torch.arange(T)[:, None].float()
    i   = torch.arange(0, d, 2).float()
    ang = pos / (10000 ** (i / d))                 # fast waves in early dimensions, slow waves in later ones
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe
```

![Sinusoidal encoding as a heatmap, and the dot product between positions](figures/fig149_sinusoidal.png)

It's like a clock with many hands. The fast hands tell nearby positions apart, and the slow hands tell far-apart ones apart. The right panel shows the useful property: **nearby positions have similar vectors**. Position 10 has a dot product of 32.0 with itself, 30.9 with positions 9 and 11, 21.1 with position 20, and 17.1 with position 40. The paper's authors also hoped fixed waves would let a model handle **longer** sequences than it was trained on. Let's test that too.

**The test:** reverse a sequence of digits, as in Sections 30–31, but now with **no recurrence at all**: two layers of multi-head self-attention (33.2), each output position predicting its digit directly. Training uses lengths 5 to 20, testing goes up to 40.

![Reversal accuracy vs length for no positions, sinusoidal and learned](figures/fig150_positions_reverse.png)

| digits right | length 5 | 10 | 20 | 25 (unseen) | 40 (unseen) |
|---|---|---|---|---|---|
| no positional information | 36.3% | 28.4% | 22.8% | 20.7% | 18.5% |
| **sinusoidal** | 99.9% | 100% | 100% | 12.9% | 10.5% |
| **learned** | 100% | 100% | 100% | 9.9% | 10.1% |

- **Without positions, reversal is impossible.** Every token sees the same bag of tokens, so output position 3 can't know it should differ from position 7. (It beats 10% chance only by guessing one of the digits that are present.) Exact-match accuracy is 0% at every length.
- **With either kind of position, it's perfect** within the trained lengths, with two attention layers and no LSTM.
- **Neither extrapolates.** At 25 digits both collapse to chance. The sinusoidal waves exist for position 25, but the network never learned what to *do* with them. Section 31's "lost its place" failure is really a **length-generalisation** problem, and it's still open. Later designs such as rotary embeddings (RoPE) and ALiBi, which encode *relative* distance directly into the attention scores, do better. Press et al. (2021) measured the same poor extrapolation for sinusoidal encodings in real language models.

> 📓 **Notebook rule:** *attention knows **what**, positions tell it **where**.* Add a position vector to every token. Just don't expect the model to understand positions it was never trained on.

---

### 33.2 Many heads: several questions at once

One attention head gives each token **one** probability distribution over the other tokens, one "place to look". But a word often needs several things at once: its subject, its object, the previous word.

**Multi-head attention** runs $h$ smaller attention heads in parallel, each with its own $W_Q, W_K, W_V$ of size $d/h$, then concatenates their outputs and mixes them with one more matrix $W_O$:

$$
\text{head}_j = \text{Attention}(XW_Q^{(j)}, XW_K^{(j)}, XW_V^{(j)})
\qquad
\text{MultiHead}(X) = [\text{head}_1; \dots; \text{head}_h]\,W_O
$$

It costs the same as one big head: same total width, same number of weights. It's just split into $h$ independent looks.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d // h
        self.Wqkv = nn.Linear(d, 3 * d, bias=False)     # Q, K, V for all heads in one matmul
        self.Wo   = nn.Linear(d, d, bias=False)          # mixes the heads back together
    def forward(self, X, causal=False):
        B, T, d = X.shape
        q, k, v = self.Wqkv(X).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)   # (B, h, T, dk) each
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dk)                                # h score tables
        if causal:
            s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        out = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, d)                      # concatenate heads
        return self.Wo(out)
```

It matches `nn.MultiheadAttention` (4 heads) to $6\times10^{-8}$.

**First test, and a surprise.** Each token must output the **ordered pair** (left neighbour, right neighbour), which needs two lookups. I expected one head to fail, since a single weighted average of two neighbours seems to blur them together. It didn't: **one head got 100%**. The left panel below shows how. It puts weight **0.3** on the left neighbour and **0.6** on the right. Because the weights are *unequal*, the two digits land in the average with different strengths, and the layer after it can tell which is which. A single head can multiplex a couple of lookups.

**So make it harder: fetch six neighbours at once** (offsets −3 to +3), one layer, same total width (48), different numbers of heads:

![One head multiplexing; accuracy vs number of heads; what each of 3 heads looks at](figures/fig151_multihead.png)

| six-neighbour task (1 layer, width 48) | accuracy (2 seeds) |
|---|---|
| 1 head (48 dims) | 87.5%, 88.1% |
| 2 heads (24 dims each) | 98.5%, 99.7% |
| **3 heads (16 dims each)** | **100%, 100%** |
| 6 heads (8 dims each) | 100%, 100% |

With six things to fetch, one head's single distribution runs out of room. The right panel shows how three heads **divide the work**. Nobody assigned them, but each head settles on its own pair of offsets: one looks at −3 and +1, one at −2 and −1, one at +2 and +3. Together they cover all six.

That's the real argument for multiple heads: several **different** attention patterns per layer, at no extra cost. In trained language models, individual heads have been found that track the previous word, match a verb to its subject, or follow a pronoun back to its noun (Clark et al., 2019).

---

### 33.3 The causal mask: no peeking at the future

To **generate** text, a model predicts the next token from the tokens so far, appends it, and repeats. Training it is efficient: feed a whole sequence and ask *every* position to predict the token after it, all in parallel (the same trick as teacher forcing, Section 30).

But self-attention lets every position see **every** other position, including the one it's supposed to predict. The fix is a **causal mask**: set every score where a query would look at a *later* key to $-\infty$ before the softmax, so position $t$ can only see positions $\le t$.

**The test:** sequences like $0, 7, 4, 1, 8, 5, \dots$, arithmetic progressions mod 10 with a random start and step. After two numbers the rest is fully determined. Train the same 2-layer attention model to predict each next number, with and without the mask, then **generate** 15 numbers from a two-number prompt:

![The causal mask, and what each model generates](figures/fig152_causal.png)

| | next-number accuracy (whole true sequence given) | generating from 2 numbers: numbers right | whole sequences right |
|---|---|---|---|
| no mask | **100%** | 23.9% | 1.8% |
| **causal mask** | 94.5% | **100%** | **100%** |

The unmasked model looks **perfect** during training, because it's cheating. To "predict" position 5 it simply looks at position 6, which is sitting right there in the input. At generation time position 6 doesn't exist yet, and the cheat is worthless: from the prompt "0, 7" it writes "0, 0, 2, 0, 7, 0…".

The masked model's training score is *lower*, 94.5%, and that's the honest number. The very first prediction (from one number alone) can't know the step size, so 1 in 16 positions is a guess ($15/16 \approx 94\%$). But it learned the real rule, and it generates every sequence perfectly.

This is Section 13's lesson in a new form: **a score is only meaningful if the model couldn't see the answer.** The causal mask enforces that during training for every position at once. Every text-generating Transformer, GPT included, uses it.

> 📓 **Notebook rule:** *when training a model to predict the future, hide the future.* Without a causal mask, next-token prediction is a copying exercise, perfect in training and useless at generation.

---

### 📓 Notebook margin: the equation so far

$$
\mathbf{x}_i = E[w_i] + \mathbf{p}_i
\qquad
\text{head}_j = \text{softmax}\!\left(\frac{Q_jK_j^\top}{\sqrt{d/h}} + M\right)V_j
\qquad
\text{MultiHead}(X) = [\text{head}_1;\dots;\text{head}_h]\,W_O
$$

$$
M_{ts} = \begin{cases} 0 & s \le t \\ -\infty & s > t \end{cases} \quad \text{(causal mask, for generation)}
$$

| piece | what it fixes | measured |
|---|---|---|
| positional encoding | attention can't tell order | reversal 22.8% → 100% (length 20) |
| sinusoidal vs learned | – | both perfect within range, both fail beyond (≈10% at 25) |
| multiple heads | one pattern per token | 6 lookups: 1 head 88%, 3 heads 100%, same width |
| causal mask | seeing the answer during training | generation 1.8% → 100% of sequences |

---

### What comes next

We now have every part of attention from the paper. **Section 34: The Transformer Block** assembles the block: multi-head attention and Section 21's MLP, each wrapped in a residual connection with layer norm. Then it trains a small Transformer from scratch, a character-level language model that learns to write text one character at a time, using the causal mask from this section. We'll watch it go from random characters to words, and look inside its attention heads.

---

*References: Ashish Vaswani et al. (2017), "Attention Is All You Need" (sinusoidal positional encodings; multi-head attention; masked decoder self-attention). Jonas Gehring et al. (2017), "Convolutional Sequence to Sequence Learning" (learned position embeddings). Ofir Press, Noah Smith & Mike Lewis (2021), "Train Short, Test Long: Attention with Linear Biases (ALiBi)" (sinusoidal encodings extrapolate poorly). Jianlin Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding" (RoPE). Kevin Clark, Urvashi Khandelwal, Omer Levy & Christopher Manning (2019), "What Does BERT Look At?" (heads with interpretable roles). Elena Voita et al. (2019), "Analyzing Multi-Head Self-Attention" (specialised heads; many can be pruned). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 15 (positional embeddings; multi-head attention; causal masking). All code in this series is PyTorch.*
