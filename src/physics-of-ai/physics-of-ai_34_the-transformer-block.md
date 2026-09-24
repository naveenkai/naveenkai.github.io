# Physics of AI — Part VI · Attention

## 34. The Transformer Block

> *Every piece is now on the table: multi-head self-attention (Sections 32–33), positions and the causal mask (33), and from Part III the residual stream, layer norm and the 4× MLP (Sections 20–21). This section snaps them together into the **Transformer block**, stacks four of them, and trains a small GPT from scratch on Shakespeare, one character at a time. We'll watch it go from random keystrokes to plausible verse, look inside its heads, remove one piece at a time to see what each is worth, and compare it honestly with an LSTM of the same size.*

---

### 34.1 The block

Section 21 ended with a promise: the modern residual block, $\mathbf{h} \leftarrow \mathbf{h} + \text{MLP}(\text{LN}(\mathbf{h}))$, is **half** of a Transformer block. Here's the other half. A block has two residual branches, one after the other:

$$
\mathbf{h} \;\leftarrow\; \mathbf{h} + \text{MultiHeadAttention}\big(\text{LN}(\mathbf{h})\big) \qquad\text{tokens exchange information}
$$
$$
\mathbf{h} \;\leftarrow\; \mathbf{h} + \text{MLP}\big(\text{LN}(\mathbf{h})\big) \qquad\qquad\qquad\;\;\text{each token thinks on its own}
$$

![The GPT stack, and one Transformer block: attention and MLP branches on a residual stream](figures/fig153_transformer_block.png)

It's worth being clear about the division of labour:

- **Attention is the only place where tokens interact.** It moves information *between* positions.
- **The MLP works on each position separately**, with the same weights at every position. It transforms what each token now knows.
- **The residual stream** carries each token's vector from the bottom of the stack to the top. Each branch *adds* to it and never overwrites it (Section 20), and layer norm sits **before** each branch (pre-norm, Section 21.3).

In PyTorch the whole block is a dozen lines, built from pieces we've already written:

```python
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.att = nn.LayerNorm(d), MHA(d, h)              # Section 33, with the causal mask
        self.ln2, self.mlp = nn.LayerNorm(d), nn.Sequential(          # Section 21
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        x = x + self.att(self.ln1(x))      # 1. look at the other tokens
        x = x + self.mlp(self.ln2(x))      # 2. think about what you saw
        return x

class GPT(nn.Module):
    def __init__(self, V, d=128, h=4, L=4, T=128):
        super().__init__()
        self.tok, self.pe = nn.Embedding(V, d), nn.Embedding(T, d)   # what + where (Sections 27, 33)
        self.blocks = nn.Sequential(*[Block(d, h) for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, V)        # final norm, then scores for the next character
    def forward(self, idx):
        x = self.tok(idx) + self.pe(torch.arange(idx.shape[1]))
        return self.head(self.ln(self.blocks(x)))
```

That's a GPT. The large ones differ mostly in size (more layers, wider $d$, more heads, longer context, far more data), not in design.

---

### 34.2 The task: predict the next character

The data is **Tiny Shakespeare** (Karpathy, 2015): 1.1 million characters of the plays, with 65 distinct characters. The first 90% is for training and the last 10% for validation. Each training example is a window of 128 characters, and the model predicts **every** next character in the window at once, which the causal mask (Section 33.3) makes legitimate.

- **Model:** 4 blocks, 4 heads, width 128, context 128 → **826,433 weights**.
- **Training:** AdamW (weight decay 0.1), 100 warm-up steps then cosine decay (Sections 14, 16), batch 32 × 128, 3,000 steps. That's about 12 million characters, roughly 12 passes over the data. It took 11.5 minutes on 2 CPUs.
- **Loss:** cross-entropy per character (Section 8), measured in nats. For scale:

| model | validation loss |
|---|---|
| guess uniformly among 65 characters | 4.17 |
| character frequencies only | 3.35 |
| previous character only (bigram counts) | 2.48 |
| **our GPT** | **1.62** |

A loss of 1.62 nats means the model is, on average, about as uncertain as a choice among $e^{1.62} \approx 5$ characters, down from 65.

![Training and validation loss: Transformer vs LSTM, with simple baselines](figures/fig154_gpt_loss.png)

---

### 34.3 Watching it learn to write

Samples at different points in training, each generated one character at a time from a newline, feeding every choice back in:

**Step 0** (random weights):
```
Caw;suqWPZ&TTNESuOgl,O,tMIvjeX;Z&Kc cUK.msowM PXDcvLLvKzVSKz$ GvJW$Nir.!LIFxZJHXhGS$
```

**Step 250** (loss 2.39): it has learned letter frequencies, spaces, line breaks and the SPEAKER: format.
```
BUMEDNDUELA:
Sart crdeem, iraim.
O, Heofl ngaiee, wees n, gritoofustelor d:
```

**Step 1,000** (loss 1.85): real short words, punctuation, rhythm.
```
Why sent much least at eyery face hose. O's caur tears;
And my in with pooroal nother, whese mage too puty.
```

**Step 3,000** (loss 1.62), a longer sample (temperature 0.8):
```
First Senator:
The put is renief thee so no more dears
To see have my peace is my franish'd solding,
That he haster that I am soldier of thee.

First Gentleman:
As with herso
so have to my heart before anger braint;
What will call no rise Grom of his most
```

It has learned speaker names followed by a colon and a newline, line lengths that look like verse, capital letters at line starts, and archaic forms like *thee*, *'tis* and *franish'd*. **82%** of the words it produces are real words from the training text. It still has no idea what anything *means*.

**Is it just copying?** Of all 20-character stretches in the sample, only **1.9%** appear anywhere in the training text (for 10-character stretches, 22%). It's mostly composing new text in the style, not reciting.

---

### 34.4 Inside the heads

Measure every head on validation text: how much weight does it put on the **previous** character, and how far back does it look on average?

![Average look-back distance per head; two attention maps](figures/fig155_gpt_heads.png)

- **Layer 1, head 1 is a "previous-character head"**: 83% of its attention goes to the character immediately before. The middle panel shows it as a clean off-diagonal line. Nobody designed it. It's the most useful single fact for predicting the next character (it's what the bigram baseline uses), so gradient descent built a head for it.
- **Layer 1, heads 2 and 3 look far back**, 34 and 31 characters on average, with broad, diffuse patterns. They gather context such as "we're in the middle of a word", "a speaker name was just announced" or "this line is getting long".
- **Layer 2's heads are all short-range** (2–3 characters): they work with what layer 1 already assembled.

This matches what Olsson et al. (2022) found in much larger models: early layers build simple "previous token" heads, and later layers combine them into more sophisticated patterns, such as heads that find an earlier occurrence of the current token and copy what followed it (**induction heads**).

---

### 34.5 What each piece is worth

Retrain with one piece removed at a time (1,000 steps each, so each run has its own full schedule):

![Validation loss after 1,000 steps, removing one component at a time](figures/fig156_gpt_ablation.png)

| model (1,000 steps) | validation loss |
|---|---|
| **full Transformer** | **1.98** |
| no MLP (attention only) | 2.28 |
| no position embeddings | 2.08 |
| **no residual connections** | **3.34** |

- **Without residual connections it barely learns**: 3.34 is worse than the bigram baseline (2.48) and almost as bad as knowing only character frequencies (3.35). Only 8 residual layers deep (4 blocks × 2 branches), and it already fails. Sections 20–21 all over again.
- **Without the MLP** it's clearly worse (+0.30). Attention can move information around, but it's mostly averaging. The per-token nonlinear processing matters. (The no-MLP model also has 64% fewer weights, so this isn't a controlled comparison of the idea alone.)
- **Without positions it's only slightly worse** (+0.09), which surprised me after Section 33, where removing positions was fatal. The difference is the **causal mask**. Position 5 can see exactly 5 tokens, position 100 can see 100, so a causal model can work out roughly where it is from how many tokens it can attend to. Haviv et al. (2022) showed exactly this: causal Transformer language models without position encodings still learn positional information. Section 33's reversal task used no mask, so there, nothing gave position away.

---

### 34.6 The honest comparison: an LSTM of the same size

Same data, same number of steps, same optimiser. An LSTM with 823k weights (one layer, 384 units, forget bias +1 as in Section 29):

| | weights | validation loss | training loss | time (2 CPUs) |
|---|---|---|---|---|
| Transformer (4 blocks) | 826,433 | 1.620 | 1.406 | 693 s |
| LSTM (1 layer) | 822,849 | **1.618** | 1.440 | **432 s** |

**At this scale, they tie.** The LSTM even learned faster early on (1.87 vs 2.09 at step 500) and ran faster on our CPU. The Transformer fits its training data a little more tightly (1.41 vs 1.44), but that doesn't carry over to validation.

So why did Transformers replace LSTMs? Not because they win at 800k weights on 2 CPUs. They won for the reasons Section 32 measured:

- **Parallel training.** All 128 positions of a window are computed in one pass. An LSTM must go through them one after another. On GPUs this is the difference between days and weeks.
- **Short paths.** Any character can attend to any earlier one in one step, with no leaky chain (Section 28).
- **Scaling.** Kaplan et al. (2020) found that as models, data and compute grow, Transformers keep improving smoothly, while LSTMs level off earlier, particularly at using long contexts.

This section's tie is at the small end of that curve. It's an honest data point, and a reminder that the Transformer's advantage is something you *earn* with scale.

> 📓 **Notebook rule:** *a Transformer block is attention (mix across tokens) plus an MLP (think within each token), both added to a residual stream.* Remove the residual stream and it can't train. Remove the MLP and it can only average. The design isn't magic at small scale. Its strength is that it keeps getting better as it gets bigger.

---

### 📓 Notebook margin: the equation so far

$$
\mathbf{h}^{(0)}_t = E[c_t] + \mathbf{p}_t
\qquad
\mathbf{h} \leftarrow \mathbf{h} + \text{MHA}_{\text{causal}}(\text{LN}(\mathbf{h}))
\qquad
\mathbf{h} \leftarrow \mathbf{h} + \text{MLP}(\text{LN}(\mathbf{h}))
\qquad
P(c_{t+1}) = \text{softmax}\big(W\,\text{LN}(\mathbf{h}^{(L)}_t)\big)
$$

| idea | what we now know |
|---|---|
| the block | attention (between tokens) + MLP (within each token), each a pre-norm residual branch |
| a character GPT | 826k weights, 11.5 min on 2 CPUs: validation loss 4.29 → 1.62 (bigram 2.48) |
| what it writes | verse-shaped text, 82% real words, 1.9% of 20-char stretches copied |
| heads | a previous-character head (83%); long-range heads (30+ chars back) |
| ablations | no residual 3.34 (fails); no MLP 2.28; no positions 2.08 (the mask leaks position) |
| vs LSTM | a tie at this size (1.620 vs 1.618); Transformers win through parallelism and scale |

---

### What comes next

We've built the decoder half of the Transformer, the half that became GPT. One section remains.

**Section 35: Attention Is All You Need** puts the whole paper together. The **encoder–decoder Transformer** has an encoder stack, a decoder stack, and **cross-attention** connecting them (Section 31's attention, now inside a Transformer). We'll train it on Section 30's reversal task, where every earlier model hit a wall, and take an encoder-only Transformer back to IMDB to face the 90.3% baseline that has stood since Section 26. Then we'll look back at the whole series: one neuron in Section 2, to this.

---

*References: Ashish Vaswani et al. (2017), "Attention Is All You Need" (the Transformer block: attention and position-wise feed-forward sublayers with residual connections and layer norm). Alec Radford et al. (2018, 2019), "Improving Language Understanding by Generative Pre-Training" / "Language Models are Unsupervised Multitask Learners" (GPT: a decoder-only Transformer trained on next-token prediction). Andrej Karpathy (2015), "The Unreasonable Effectiveness of Recurrent Neural Networks" (char-rnn; the Tiny Shakespeare dataset), and nanoGPT. Catherine Olsson et al. (2022), "In-context Learning and Induction Heads". Adi Haviv, Ori Ram, Ofir Press, Peter Izsak & Omer Levy (2022), "Transformer Language Models without Positional Encodings Still Learn Positional Information". Jared Kaplan et al. (2020), "Scaling Laws for Neural Language Models" (Transformers vs LSTMs as scale grows). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 15–16 (the Transformer; text generation with a Transformer). All code in this series is PyTorch.*
