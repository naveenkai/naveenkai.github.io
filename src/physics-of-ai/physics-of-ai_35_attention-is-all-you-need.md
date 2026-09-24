# Physics of AI — Part VI · Attention

## 35. Attention Is All You Need

> *In 2017, eight researchers at Google published a paper whose title was also its argument: you don't need recurrence, and you don't need convolution. Attention, stacked with the residual blocks we built in Part III, is enough. This last section assembles their full model, the **encoder–decoder Transformer**, from the pieces of Sections 31–34. We give it the task that defeated every recurrent model in Section 30, then take it back to IMDB to face the baseline that has stood since Section 26. Then we look back at the whole road, from one neuron deciding whether to watch a movie to this.*

---

### 35.1 The whole model

Section 34 built the **decoder-only** Transformer (a GPT): one stack, masked self-attention, predicting the next token. The original paper was built for translation, so it has **two stacks**:

![The encoder–decoder Transformer: self-attention in both stacks, cross-attention between them](figures/fig157_full_transformer.png)

- **The encoder** reads the whole input. Each block is **self-attention** (every input token looks at every other, no mask) plus an **MLP**, each as a pre-norm residual branch. It replaces Section 30's encoder LSTM.
- **The decoder** writes the output one token at a time. Each block has **three** residual branches:
  1. **masked self-attention**: look at what I've written so far (Section 33's causal mask);
  2. **cross-attention**: queries come from the decoder, keys and values come from the encoder's output. This is *exactly* Section 31's attention, the decoder asking "which input do I need now?", but now inside a Transformer block with multiple heads;
  3. an **MLP**.

In code, the only new piece is letting queries come from a different sequence than keys and values:

```python
class DecBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.self_att = nn.LayerNorm(d), MHA(d, h)
        self.ln2, self.cross    = nn.LayerNorm(d), MHA(d, h)
        self.ln3, self.ff       = nn.LayerNorm(d), mlp(d)
    def forward(self, y, mem, mem_mask):
        y = y + self.self_att(self.ln1(y), causal=True)                 # what have I written?
        y = y + self.cross(self.ln2(y), mem=mem, key_mask=mem_mask)     # what does the input say?
        y = y + self.ff(self.ln3(y))                                    # think about it
        return y
```

Two small differences from the 2017 paper, both taken from later practice and from Section 21: we use **pre-norm** (the paper put layer norm after each addition) and **learned** position embeddings (the paper used sinusoids; Section 33 found no difference within the trained lengths). The paper's base model had 6 + 6 blocks, width 512, 8 heads and 65 million weights. Ours has 2 + 2 blocks, width 64, 4 heads and **239,372** weights.

---

### 35.2 Rematch: reversing digits

This is Section 30's task: reverse (or copy) random digit strings, trained on lengths 5–30 and tested on 1,000 new sequences per length, including lengths 35 and 40 that training never showed.

![Exactly-right sequences vs length: LSTM encoder–decoder, LSTMs with attention, and the Transformer](figures/fig158_reversal_all.png)

| exactly right, 30 digits | reverse | copy |
|---|---|---|
| LSTM encoder–decoder (Section 30) | 50.7% | 1.4% |
| LSTMs + attention (Section 31) | 100% | 99.9% |
| **Transformer, no recurrence at all** | **100%** | **100%** |

Every length from 5 to 30 is **100% right** for both tasks. The loss fell to 0.0001 by step 2,000.

Look at how it reverses:

![Cross-attention of every decoder head while reversing 20 digits](figures/fig159_cross_attention.png)

The two decoder layers split the job:

- **Layer 1's heads are broad and blurry**, concentrated near the *end* of the input. They find where the input stops, which is what reversal needs as a starting point.
- **Layer 2's heads are sharp anti-diagonals**: output 1 reads input 20, output 2 reads input 19, and so on. All four heads agree, each slightly smeared across neighbours.

Nobody told it to split the work this way. It's Section 25's "edges → parts → objects" ladder again, in a different form: early layers locate, later layers retrieve.

**And the honest part: lengths it never saw.** At 35 digits, reversal collapses to **9.5% of digits right**, which is chance. Copying degrades more gracefully: **91.4%** of digits right at 35 and 82.5% at 40, but almost never a perfect sequence (0.9%). That's the same pattern as Sections 31 and 33. The position embeddings for positions 31–40 were never trained, so the model has no idea where those tokens are. The Transformer removed the bottleneck and the leaky chain, but it didn't remove the need to have *seen* something like the test during training (Section 13's lesson, all the way at the end).

---

### 35.3 Back to IMDB: does the Transformer beat the bag?

An **encoder-only** Transformer (like BERT, minus the pre-training) reads the last 200 words of each review. Two blocks, 4 heads, dropout 0.2, the outputs averaged over the words, then one linear layer. We use the same 200 words the RNN, LSTM, GRU and bag saw in Sections 28–29. Two versions: embeddings **from scratch** (width 64), and embeddings **started from Section 27's word2vec** (width 100).

![Best validation accuracy on IMDB, last 200 words: every sequence model in the series](figures/fig160_imdb_final.png)

| model, last 200 words of each review | weights | best validation |
|---|---|---|
| bag of embeddings (order-blind) | 320k | 89.4% |
| plain RNN (Section 28) | 650k | 83.5% |
| LSTM (Section 29) | 670k | 88.8% |
| GRU (Section 29) | 660k | **89.5%** |
| Transformer, from scratch | 1.39M | 84.7% |
| **Transformer, word2vec start** | 2.26M | 88.3% |

(Weights are mostly the 20,000-word embedding table.)

**The Transformer does not beat the bag here.** From scratch, it lands between the plain RNN and the LSTM. Starting its embeddings from word2vec adds 3.6 points in fewer epochs, which is Section 27's pre-training lesson again, but it still ends 1 point short of the bag and the GRU.

It's the final test-set opening of the series, so I checked this model on the held-out test reviews **once**:

> **Transformer (word2vec start): 88.24% test**, against the bag of words and pairs from Section 26: **90.26% test**.

What it does do is **read order**. The same phrase tests as before:

| sentence | bag of words (26) | LSTM (29) | **Transformer** |
|---|---|---|---|
| "i expected it to be great but it was terrible" | 36.6% | 11.4% | 21.9% |
| "i expected it to be terrible but it was great" | 36.6% | 80.8% | 31.5% |
| "this movie was good" | 48.0% | 76.1% | **95.8%** |
| "this movie was not good" | 41.2% | 28.4% | **2.8%** |

It separates "good" from "not good" more sharply than anything before it (95.8% vs 2.8%), but it doesn't get the "expected…but" pair right (both below 50%). It learned negation, not contrast.

**Why didn't the architecture that powers GPT and BERT win?** Every measurement in Part V points the same way:

1. **The task barely needs order.** Sentiment is mostly in which words appear (Section 26: 90% from a bag). There's little for attention's power to add.
2. **40,000 reviews is tiny for a Transformer.** Attention has almost no built-in assumptions, unlike convolution's "nearby pixels matter" or recurrence's "read left to right", so it needs lots of data to learn what the others assume. The word2vec start helped precisely because it brought outside knowledge in.
3. **This is exactly why BERT exists.** Devlin et al. (2019) pre-trained a Transformer encoder on 3.3 billion words of plain text before fine-tuning it on tasks like this one, and reached about **95%** on IMDB-style sentiment, far past any bag. The Transformer's strength isn't a small labelled dataset; it's learning from *everything*, then adapting. Section 27's 7-minute word2vec run was the miniature of that idea.

> 📓 **Notebook rule:** *architecture is not a substitute for data.* The Transformer is the most flexible sequence model we've built, and flexibility has to be paid for with data or with pre-training. On a small, word-driven task, the humble bag still wins.

---

### 35.4 Looking back: one equation, evolving

The series started with a single neuron deciding whether to watch a movie (the prologue and Section 2), and it ends by reading 50,000 movie reviews:

$$
z = w_1\,\text{great reviews} + w_2\,\text{favourite actor} + w_3\,\text{work tomorrow} + b
$$

Every section after that changed this equation a little.

![The whole journey: six parts, one evolving equation](figures/fig161_journey.png)

**Part I · Gears (§1–12).** The weighted sum became a layer, $W\mathbf{x} + \mathbf{b}$. A nonlinearity let layers fold space (Sections 6–7). A loss measured wrongness (8), gradient descent rolled downhill (9–10), and backprop computed every slope in one backward pass (11). That was enough to read handwritten digits: **97.87%**.

**Part II · Making it learn (§13–18).** Validation sets caught memorisation (13). Weight decay, dropout and augmentation held the network back (14). The right initialisation kept signals alive (15), Adam and schedules took better steps (16), and batch norm kept everything in range (17). Together: **99.25%**.

**Part III · Why depth (§19–21).** One hidden layer can approximate anything, but at an absurd cost (19). Deeper was better until it wasn't: 50 plain layers trained to 17%. Then one character, $\mathbf{h} + F(\mathbf{h})$, fixed it (20). Pre-norm, layer norm and GELU turned it into the modern block (21), which, it turned out, was half a Transformer.

**Part IV · Seeing (§22–25).** Convolution shared one small detector across the whole image (22). Pooling widened the view (23), and residual ConvNets handled real photos (24). Looking inside showed edge detectors, part detectors, and shortcuts like "sky means aeroplane", plus an invisible change that fooled it completely (25).

**Part V · Remembering (§26–30).** Words became vectors (26–27). A recurrent network read them in order but forgot within a few dozen steps (28). Gates gave memory a highway (29), and the encoder–decoder squeezed a whole sentence into one vector, which broke (30).

**Part VI · Attention (§31–35).** Instead of carrying everything, the model looked it up (31). Every token asked every other token directly (32). Positions and multiple heads (33), then the full block, trained on Shakespeare (34), and finally the paper's model (35).

The final equation is still recognisably the first one, just used many times over:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
\qquad Q = XW_Q,\;\; K = XW_K,\;\; V = XW_V
$$

Each of $XW_Q$, $XW_K$ and $XW_V$ is Section 2's weighted sum, computed for every token at once. The softmax is Section 8's. Dividing by $\sqrt{d}$ is Section 15's initialisation logic. The residual stream around it is Section 20's, the layer norm is Section 21's, and it's trained by Section 11's backprop with Section 16's Adam. **Nothing in a Transformer is new except the way the pieces are arranged.**

---

### 35.5 What the measurements taught

Across 35 sections, some lessons kept coming back, each time in a new form:

| lesson | where we measured it |
|---|---|
| **Short paths train; long chains don't.** | vanishing signals (15), residuals (20), RNN gradients (28), LSTM highways (29), attention (31) |
| **Keep signals the right size.** | standardising inputs (4, 9), He init (15), batch/layer norm (17, 21), $\sqrt{d}$ scaling (32) |
| **A score only counts on data the model didn't see.** | validation (13), the test envelope (18, 23, 24, 26), the causal mask (33) |
| **Models learn whatever predicts the label.** | "7/10" in reviews (26), sky → aeroplane (25), "not bad" (26, 29) |
| **Build in what you know, or pay for it with data.** | convolution on images (22–24), bag on sentiment (26), the Transformer on small IMDB (35) |
| **Honest baselines are hard to beat.** | the bag of words stood from Section 26 to the end |

---

### 📓 Notebook margin: the final equation

$$
\begin{aligned}
\mathbf{h}^{(0)}_t &= E[w_t] + \mathbf{p}_t \\
\mathbf{h} &\leftarrow \mathbf{h} + \text{MultiHead}\big(\text{LN}(\mathbf{h})\big) \\
\mathbf{h} &\leftarrow \mathbf{h} + W_2\,\text{GELU}\big(W_1\,\text{LN}(\mathbf{h})\big) \\
P(w_{t+1}) &= \text{softmax}\big(W_{\text{out}}\,\text{LN}(\mathbf{h}^{(L)}_t)\big)
\end{aligned}
$$

| Section 35 | result |
|---|---|
| encoder–decoder Transformer | 239k weights, no recurrence |
| reversal and copying, 30 digits | 100% (LSTM encoder–decoder: 50.7% and 1.4%) |
| unseen lengths | reversal at chance; copying 91% of digits: positions must be trained |
| IMDB, from scratch | 84.7% validation |
| IMDB, word2vec start | 88.3% validation, **88.24% test** (bag of pairs: 90.26%) |
| what it reads well | "good" 95.8% vs "not good" 2.8% |

---

## End of the series

This series set out to go from a single neuron to *Attention Is All You Need*, building every piece in PyTorch and measuring every claim. It got there, and a lot of the measurements weren't what the textbooks would lead you to expect. An LSTM tied a Transformer on Shakespeare. A bag of words beat everything on movie reviews. An LSTM's default settings forgot as fast as a plain RNN. One attention head quietly did the work of two.

These results aren't failures of the ideas. They show where each idea's strength actually lies. The Transformer won the world not by being clever at small scale but by being simple, parallel and hungry: it keeps improving as it's given more data and compute, which is the one thing a 2-CPU notebook can't show. Everything else, from the neuron to the residual stream to the softmax over keys, you've now built yourself and seen working.

---

*References: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan Gomez, Łukasz Kaiser & Illia Polosukhin (2017), "Attention Is All You Need" (the encoder–decoder Transformer; base model: 6 + 6 layers, d = 512, 8 heads, 65M parameters). Ruibin Xiong et al. (2020), "On Layer Normalization in the Transformer Architecture" (pre-norm). Jacob Devlin, Ming-Wei Chang, Kenton Lee & Kristina Toutanova (2019), "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (encoder-only pre-training; 3.3B words; strong sentiment results after fine-tuning). Chi Sun, Xipeng Qiu, Yige Xu & Xuanjing Huang (2019), "How to Fine-Tune BERT for Text Classification?" (about 95% on IMDB). Alec Radford et al. (2018), "Improving Language Understanding by Generative Pre-Training" (decoder-only). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 15 (the Transformer encoder and decoder; the sequence-to-sequence Transformer; when bag-of-words models are enough). Michael Nielsen, *Neural Networks and Deep Learning* (the foundations of Parts I–III). All code in this series is PyTorch.*
