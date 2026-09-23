# Physics of AI — Part VI · Attention

## 31. Looking Back

> *Section 30's encoder–decoder had to squeeze a whole input through one vector, and it broke as the input grew: 50.7% on 30-digit reversal, and 1.4% on copying. This section makes one change. The decoder keeps **all** the encoder's states and, at every output step, decides which ones to look at. That decision is a softmax over similarity scores, and it's called **attention**. Everything in the rest of this series grows out of the few lines of code below.*

---

### 31.1 The idea: a weighted look-back

As the encoder reads, it produces a state at every input position: $\mathbf{h}_1, \dots, \mathbf{h}_T$. Section 30 threw all of them away except the last. Now we keep them all.

When the decoder is about to write output $t$, its own state $\mathbf{s}_t$ says roughly *what it needs next*. Attention does three things with it:

$$
\underbrace{e_{tj} = \mathbf{s}_t^\top W\,\mathbf{h}_j}_{\text{1. score: how well does input } j \text{ match?}}
\qquad
\underbrace{\alpha_{tj} = \frac{e^{e_{tj}}}{\sum_{k} e^{e_{tk}}}}_{\text{2. softmax: turn scores into weights}}
\qquad
\underbrace{\mathbf{c}_t = \sum_j \alpha_{tj}\,\mathbf{h}_j}_{\text{3. read: a weighted average}}
$$

1. **Score** every input position against the decoder's current state. This is a dot product, the same "how aligned are these two vectors?" measure as Section 2's weighted sum, through a learned matrix $W$.
2. **Softmax** the scores into weights that are positive and sum to 1, exactly as Section 8 turned class scores into probabilities. Here the "classes" are input positions.
3. **Read** a weighted average of the encoder states: the **context vector** $\mathbf{c}_t$. If one weight is near 1, the context is essentially that one state.

The prediction then uses both what the decoder knows and what it just read: $\text{softmax}(W_o[\mathbf{s}_t;\, \mathbf{c}_t])$.

![Attention: the decoder scores every encoder state, softmaxes the scores, and reads a weighted average](figures/fig141_attention_diagram.png)

(The weights in the diagram are illustrative. The real ones are shown in 31.3.)

This is **Bahdanau, Cho & Bengio (2014)**, in the simpler "multiplicative" scoring form of Luong, Pham & Manning (2015). Bahdanau's original score was a small MLP, $\mathbf{v}^\top\tanh(W_1\mathbf{s}_t + W_2\mathbf{h}_j)$, but the idea is identical.

Here it is added to Section 30's model. Everything else is unchanged: same encoder, same decoder, same training:

```python
class AttnSeq2Seq(Seq2Seq):
    def __init__(self, H=128, d=32):
        super().__init__(H, d)
        self.Wk  = nn.Linear(H, H, bias=False)              # the W in  s·W·h
        self.out = nn.Linear(2 * H, VOC)                    # predict from [decoder state ; context]
    def attend(self, s, hs, mask):                          # s: (B, T_out, H)   hs: (B, T_in, H)
        scores = s @ self.Wk(hs).transpose(1, 2)            # (B, T_out, T_in): every output step vs every input
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))   # never attend to padding
        alpha  = scores.softmax(-1)                         # each row sums to 1
        return alpha @ hs, alpha                            # context vectors, and the weights themselves
    def forward(self, src, tgt):
        hs, state = self.encode(src)                        # keep ALL encoder states this time
        s, _ = self.dec(self.emb(shift_right(tgt)), state)  # teacher forcing, as in Section 30
        ctx, _ = self.attend(s, hs, src != PAD)
        return self.out(torch.cat([s, ctx], -1))
```

Two details in `attend`:

- **One matrix multiply scores every (output, input) pair at once**, a `(T_out × T_in)` table. That's the first hint of why attention is fast.
- **Masking.** Padding positions get a score of $-\infty$, so the softmax gives them weight exactly 0.

The additions cost **17,920 weights** (185,740 vs 167,820 total, +11%).

---

### 31.2 The bottleneck disappears

Same task, same training budget (5,000 steps), same test sequences:

![Exact-match accuracy vs length: every model from Section 30 vs attention](figures/fig143_attention_length.png)

| exactly right | 20 digits | 25 | 30 | 35 (unseen) | 40 (unseen) |
|---|---|---|---|---|---|
| no attention, $H = 128$ (Section 30) | 99.3% | 92.3% | 50.7% | 0.1% | 0.0% |
| no attention, $H = 256$ (Section 30) | 100% | 99.3% | 89.9% | 0.0% | 0.0% |
| **attention, $H = 128$** | **100%** | **100%** | **100%** | 0.1% | 0.0% |
| **attention, $H = 32$** | **100%** | **100%** | **99.7%** | 0.0% | 0.0% |
| no attention, **copy**, $H = 128$ | 73.1% | 26.1% | 1.4% | 0.0% | 0.0% |
| **attention, copy, $H = 128$** | **100%** | **99.9%** | **99.9%** | **95.3%** | 30.7% |

Three results:

**1. Within the trained lengths, the problem is solved.** Every length from 5 to 30 is now 100% right, on 1,000 test sequences each.

**2. Memory size stops mattering.** With attention, a model with $H = 32$ (64 numbers of state) gets 99.7% on 30 digits. Without attention, eight times as much memory ($H = 256$) got 89.9%. The decoder no longer has to *carry* the input, because it can *look it up*.

**3. Copying goes from impossible to easy**, 1.4% → 99.9% at 30 digits. In Section 30, copying was hard because the first output needed the oldest input, 30 steps back through the encoder. With attention, **every input is one step from every output**, so direction no longer matters. Sutskever's reverse-the-input trick becomes unnecessary.

It also learns much faster:

![Training loss: with and without attention](figures/fig144_attention_loss.png)

| training loss after… | 1,500 steps | 5,000 steps |
|---|---|---|
| no attention | 0.28 | 0.033 |
| attention | **0.0013** | **0.0001** |

That's Section 20's lesson once more, in yet another form. A short path from input to output means a short path for the gradient, and the gradient from output 30 now reaches input 1 through a single softmax instead of 30 LSTM steps.

---

### 31.3 Watching where it looks

The weights $\alpha_{tj}$ are the most interpretable thing a network has given us so far. Plot them as a grid, one row per output digit and one column per input digit:

![Attention weights: reverse (anti-diagonal), copy (diagonal), and reverse on an unseen length](figures/fig142_attention_maps.png)

- **Reverse, 20 digits (left).** The anti-diagonal: output 1 looks at input 20, output 2 at input 19, and so on. On average the largest weight in each row is **0.98**, so the model looks at essentially one position at a time. Nobody told it to reverse by position. It learned to find the right input on its own.
- **Copy, 20 digits (middle).** A perfect diagonal (average top weight **0.997**).
- **A small surprise on the left.** A few rows are shifted by one: two consecutive outputs look at the same input, and one input is skipped, yet every digit is still right. That's because each encoder state $\mathbf{h}_j$ is an LSTM state. It holds a summary of the inputs *up to* $j$, not just digit $j$, so looking at $\mathbf{h}_{14}$ can also recover digit 13. The weights show **where** the model reads, and it's slightly fuzzier about **what** it reads. It's worth remembering whenever attention maps are used as explanations.

**The failure (right): reversing 40 digits, longer than any training input.** For the first dozen outputs the model walks backwards correctly, then it **loses its place**: it jumps back to the end of the input and starts reversing again, and then again. The average top weight drops to 0.67. Only 12% of digits are right.

So attention removes the *bottleneck*, but not the *length limit*. Attention decides where to look by matching **contents**, and nothing in the encoder states says "I am position 35", a position the model never saw during training. The decoder has to keep count in its own LSTM state, and it never learned to count that far. (Copying extrapolates better, 95.3% at 35 digits. My guess is that "the next position after the one I just read" is an easier rule to continue than counting backwards from the end, but I haven't tested that.) How to tell a model **where** each token is will come back in Section 33, as **positional encoding**.

---

### 31.4 What we just built, in general terms

Strip away the translation setting and attention is a general-purpose operation. There's a **query** (what I'm looking for: $\mathbf{s}_t$), a set of **keys** to match it against (here $W\mathbf{h}_j$), and a set of **values** to read out (here $\mathbf{h}_j$ again):

$$
\text{Attention}(\mathbf{q}, K, V) = \text{softmax}(\mathbf{q}\,K^\top)\;V
$$

It's a **soft dictionary lookup**. A Python dictionary returns the value whose key *exactly* equals your query. Attention returns a *blend* of all values, weighted by how well each key matches, and because the blend is smooth, the whole lookup is differentiable and can be trained by backprop (Section 11).

Here's what attention changes in terms of paths:

| | path from input $j$ to output $t$ | what limits it |
|---|---|---|
| encoder–decoder (Section 30) | through every later encoder step, the handoff, and every earlier decoder step | one fixed vector; gradients through long chains |
| **with attention** | **one step**: a softmax weight | nothing between them |

> 📓 **Notebook rule:** *don't carry everything, look it up.* A decoder with attention doesn't need to remember the input. It needs to know what to ask for, and every input is one softmax away.

---

### 📓 Notebook margin: the equation so far

$$
e_{tj} = \mathbf{s}_t^\top W\,\mathbf{h}_j
\qquad
\alpha_{tj} = \text{softmax}_j(e_{tj})
\qquad
\mathbf{c}_t = \sum_j \alpha_{tj}\,\mathbf{h}_j
\qquad
P(y_t) = \text{softmax}\big(W_o[\mathbf{s}_t; \mathbf{c}_t]\big)
$$

| idea | what we now know |
|---|---|
| attention | score every input, softmax, read a weighted average |
| the bottleneck | gone: 30-digit reversal 50.7% → 100% |
| memory size | $H = 32$ with attention (99.7%) beats $H = 256$ without (89.9%) |
| order of output | copying 1.4% → 99.9%: every input is one step away |
| training speed | loss 0.28 → 0.0013 after 1,500 steps |
| what it learns | clean diagonals: top weight 0.98 (reverse), 0.997 (copy) |
| what it doesn't fix | unseen lengths: it loses its place, because nothing says where a token is |

---

### What comes next

There's still an LSTM on each side of our attention. The encoder reads step by step so each $\mathbf{h}_j$ can know about its neighbours, and the decoder steps along to keep its place. Both are the slow, sequential chains that Part V found leaky.

But look at what attention does: it lets one position gather information from **any** other position in one step. So why not let every **input** position attend to every other input position too, instead of passing information along a chain?

**Section 32: Self-Attention** does exactly that. Each token produces a **query**, a **key** and a **value**, and every token attends to every other, all at once, in one matrix multiply. No recurrence at all. We'll build it from scratch, check it against PyTorch, watch it solve tasks that need tokens to find each other across a sequence, and meet the $\sqrt{d}$ in *"Attention Is All You Need"*.

---

*References: Dzmitry Bahdanau, Kyunghyun Cho & Yoshua Bengio (2014), "Neural Machine Translation by Jointly Learning to Align and Translate" (attention; alignment plots). Minh-Thang Luong, Hieu Pham & Christopher Manning (2015), "Effective Approaches to Attention-based Neural Machine Translation" (dot/general scoring). Alex Graves, Greg Wayne & Ivo Danihelka (2014), "Neural Turing Machines" (content-based addressing as a soft lookup). Sarthak Jain & Byron Wallace (2019), "Attention is not Explanation" (caution about reading attention maps as explanations). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 15 (the attention mechanism; queries, keys and values). All code in this series is PyTorch.*
