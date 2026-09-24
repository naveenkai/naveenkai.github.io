# Physics of AI — Part III · Why Depth?

## 21. The Modern Block

> *Section 20 found the key idea: $\mathbf{h} + F(\mathbf{h})$. But "a residual connection around some layers" still leaves choices. Which normalisation? Before the branch or after the addition? What goes inside the branch, and with which activation? This section settles each choice by experiment and assembles the block that modern networks actually stack, the same block that forms the second half of every Transformer. Part III ends with a network 192 layers deep that trains with no special tricks at all.*

---

### 21.1 The test bench

Each network below is a stack of **48 blocks**, and each block has two linear layers inside its branch, so that's **96 hidden layers**. The width is 64, trained with AdamW for 2 epochs on 50,000 digits, same seed each time. We change **one** design choice at a time.

The modern block we'll end up with:

$$
\mathbf{h} \;\leftarrow\; \mathbf{h} + W_2\,\text{GELU}\big(W_1\,\text{LayerNorm}(\mathbf{h})\big)
\qquad W_1: d \to 4d,\;\; W_2: 4d \to d
$$

```python
class Block(nn.Module):
    def __init__(self, d=64, expand=4):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1  = nn.Linear(d, expand * d)      # widen: d → 4d
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(expand * d, d)      # back:  4d → d
    def forward(self, h):
        return h + self.fc2(self.act(self.fc1(self.norm(h))))     # pre-norm residual
```

Four ingredients: **layer norm**, placed **before** the branch, a branch that goes **wide** (4×), and **GELU**. Let's justify each one.

---

### 21.2 Layer norm: normalise each example on its own

Section 17 showed batch norm's weak spot: it normalises each feature *across the batch*, so an example's output depends on its batch-mates. Batches of all-sevens broke it (96.1% → 17.0%).

**Layer norm** flips the direction. It normalises each example *across its own features*: subtract that example's mean over its $d$ numbers, divide by their spread, then apply a learnable scale γ and shift β as before.

![BatchNorm normalises columns, LayerNorm normalises rows](figures/fig92_bn_vs_ln.png)

```python
class MyLayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.gamma, self.beta, self.eps = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d)), eps
    def forward(self, h):
        mean = h.mean(-1, keepdim=True)                          # over this example's features
        var  = h.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (h - mean) / torch.sqrt(var + self.eps) + self.beta
```

This matches `nn.LayerNorm` to $2\times10^{-7}$. Because every example is handled independently, there are no running averages and no train/eval difference: a digit's output is the same whether it's alone or in a batch of 100. Here's Section 17's trap again, on two 48-block networks:

| 48 blocks | accuracy (eval mode) | train mode, batches of one digit |
|---|---|---|
| batch norm | 96.2% | **13.8%** |
| **layer norm** | 96.1% | **96.1%** |

Same accuracy, but with no trap. On MNIST, layer norm doesn't beat batch norm on accuracy (96.1% vs 96.2%, which is noise). It wins on **reliability**, and for sequences of text, where batches contain sentences of different lengths and one sentence's statistics shouldn't affect another's, that reliability is essential. That's why Transformers use it.

---

### 21.3 Where the norm goes: after the add, or before the branch?

The original 2017 Transformer normalised **after** adding the skip ("post-norm"): $\mathbf{h} \leftarrow \text{LN}(\mathbf{h} + F(\mathbf{h}))$. Nearly all modern models normalise **before** the branch ("pre-norm"): $\mathbf{h} \leftarrow \mathbf{h} + F(\text{LN}(\mathbf{h}))$.

![Post-norm vs pre-norm](figures/fig94_post_vs_pre.png)

It looks like a small reshuffle, but look at the skip path. In post-norm, the "untouched copy" of $\mathbf{h}$ is **renormalised at every block**. Across 48 blocks it passes through 48 layer norms, so it isn't untouched any more, and the gradient highway from Section 20.5 has a checkpoint every block. In pre-norm, the skip really is a clean copy all the way through.

| 48 blocks | learning rate | training accuracy | validation accuracy |
|---|---|---|---|
| post-norm | 0.001 | 84.9% | 83.9% |
| **pre-norm** | 0.001 | **96.8%** | **96.1%** |
| post-norm | 0.003 | 11.7% (dead) | 11.0% |
| post-norm + 1 epoch warm-up | 0.003 | 11.7% (dead) | 11.0% |
| **pre-norm** | 0.003 | **96.1%** | **95.5%** |

![Training curves: post-norm vs pre-norm](figures/fig95_placement_curves.png)

At the gentle learning rate post-norm trains, but slowly (84.9% after 2 epochs). At the higher one it doesn't train at all, and even a warm-up (Section 16.6) didn't save it here. Pre-norm trains well at both, with no warm-up at all.

This matches what researchers found with real Transformers (Xiong et al., 2020): post-norm Transformers **need** a careful learning-rate warm-up to train at all, and pre-norm ones are much more forgiving.

> 📓 **Notebook rule:** *keep the skip path clean.* Put the normalisation **inside** the branch, before the layers, so the residual stream from input to output is never touched.

(One consequence: with pre-norm, nothing normalises the stream at the very end, so modern networks add **one final layer norm** before the output layer. Our `Net` does that too.)

---

### 21.4 A wide branch: d → 4d → d

Inside the branch, the first linear layer **widens** the signal to 4× its size, the activation acts there, and the second layer brings it back down. Why widen?

Remember Section 7: each neuron in a layer is one fold, one direction measured. The residual stream carries $d$ numbers from block to block, but *inside* the branch the block can afford $4d$ neurons, four times as many folds, for its correction before squeezing the result back to $d$. It's cheap to carry a narrow stream and think wide in between.

| 48 blocks | parameters | validation accuracy |
|---|---|---|
| branch 1× wide (d → d → d) | 456,522 | 96.0% |
| **branch 4× wide** (d → 4d → d) | 1,645,386 | 96.1% |

Honest result: **on MNIST it makes no difference.** Digits don't need the extra capacity, and the 1× version gets the same score with 3.6× fewer weights. The 4× ratio comes from the original Transformer paper and matters on hard problems like language, where models are usually short of capacity. We keep it because that's where we're heading, not because MNIST asked for it.

---

### 21.5 GELU: a softer ReLU

The last ingredient is the activation. **GELU** (Gaussian Error Linear Unit, Hendrycks & Gimpel, 2016) is

$$
\text{GELU}(z) = z \cdot \Phi(z)
$$

where $\Phi(z)$ is the probability that a standard bell-curve value is below $z$. For large positive $z$, $\Phi \approx 1$, so GELU ≈ $z$, like ReLU. For large negative $z$, $\Phi \approx 0$, so GELU ≈ 0, also like ReLU. In between it bends **smoothly**, and it lets small negative values through a little:

![ReLU vs GELU and their slopes](figures/fig93_gelu.png)

No sharp corner, and, more importantly, **no completely flat zone with zero slope**, so neurons can't die the way ReLU's can (Section 16.6).

| 48 blocks | validation accuracy |
|---|---|
| ReLU | 95.5% |
| **GELU** | 96.1% |

A small gain here, close to the seed noise we measured in Section 18 (about 0.2 points), so treat it as "at least as good". GELU is the standard activation in BERT, GPT and most Transformers since. We use it because that's where we're heading, and because it removes the dead-neuron failure mode for free.

---

### 21.6 Part III's finale: 192 layers, no tricks

Everything together: **96 blocks** (192 linear layers in the branches), pre-norm, layer norm, 4× GELU branches, AdamW at learning rate 0.001, **no warm-up, no special initialisation**:

| 96 blocks (192 layers), same settings | training accuracy | validation accuracy |
|---|---|---|
| post-norm | 10.2% (doesn't train) | 9.6% |
| **modern pre-norm block** | **96.0%** | **95.2%** |

Compare where Part III started. In Section 20.1, a *plain* 50-layer network with every Part II tool reached 17.3%. Now a network four times deeper trains straight away, with nothing special. (It doesn't *beat* a small network on MNIST, which doesn't need 192 layers. The point is that depth has become safe.)

---

### 21.7 You've just built half a Transformer

Put an **attention** layer in front of this block, wrapped in exactly the same pre-norm residual pattern, and you have a Transformer block:

![A Transformer block is two residual branches](figures/fig96_transformer_preview.png)

Everything in the dashed box is this section's block, exactly: layer norm → $d \to 4d$ → GELU → $4d \to d$ → add back. The only missing piece is the top branch, **attention**, which is what lets the model look at *other positions in a sequence* (other words in a sentence, other patches of an image). Building it is the whole of Part VI.

---

### 📓 Notebook margin: Part III in one table

$$
\mathbf{h} \;\leftarrow\; \mathbf{h} + W_2\,\text{GELU}\big(W_1\,\text{LN}(\mathbf{h})\big)
$$

| section | idea | biggest measured effect |
|---|---|---|
| 19 | one hidden layer can approximate anything | but the size can be exponential ($10^{784}$), and training can't find hand-built weights (2,500× worse) |
| 19 | deep beats wide at the same size | 44× lower error (4 layers vs 1) |
| 20 | degradation, and residual connections | 50 plain layers: 17.3% → residual: 98.0% |
| 21 | layer norm | no batch dependence: 13.8% (BN trap) vs 96.1% |
| 21 | pre-norm | 48 blocks at lr 0.003: 11% (post) vs 95.5% (pre) |
| 21 | the modern block | 192 layers train cleanly: 95.2% |

---

## End of Part III

Part III answered "why depth?" with a *yes, but*. Depth is how networks get their efficiency (19), and residual connections plus careful normalisation (20, 21) are how depth becomes trainable.

But every network so far has had a blind spot, noted back in Section 3.5. We **flatten** the image into 784 numbers, and the network has no idea which pixels are neighbours. It would score the same if we shuffled every digit's pixels in the same fixed way. Part IV fixes that.

**Part IV: Seeing** starts with **Section 22: Looking Through a Small Window**, the convolution: one small set of weights slid across the whole image, so that a stroke detector learned in one corner works everywhere. We'll prove the "shuffled pixels" claim with an experiment, build a convolution from scratch, and see what a single layer of learned filters picks out of real digits.

---

*References: Jimmy Lei Ba, Jamie Ryan Kiros & Geoffrey Hinton (2016), "Layer Normalization". Ruibin Xiong et al. (2020), "On Layer Normalization in the Transformer Architecture" (post-norm needs warm-up; pre-norm trains without it). Dan Hendrycks & Kevin Gimpel (2016), "Gaussian Error Linear Units (GELUs)". Ashish Vaswani et al. (2017), "Attention Is All You Need" (the position-wise feed-forward block with inner size 4× the model width, post-norm residuals). Kaiming He et al. (2016), "Identity Mappings in Deep Residual Networks" (keeping the identity path clean). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 15 (the Transformer encoder block: layer normalisation and residual connections). All code in this series is PyTorch.*
