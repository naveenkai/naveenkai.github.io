# Physics of AI — Part V · Remembering

## 29. Gates and Highways

> *Section 28's RNN forgot almost everything more than a few dozen words back, because its memory is rewritten at every step and the gradient shrinks at every rewrite. Section 20 met the same disease in deep networks and cured it with one idea: **add** a correction instead of replacing the signal. The **LSTM** (Hochreiter & Schmidhuber, 1997) is that cure applied through time, eighteen years before ResNets. This section builds one from scratch and finds that the cure only works if the highway starts **open**. Then it measures how far gated memory gets us on IMDB.*

---

### 29.1 A second memory, updated by adding

The LSTM keeps **two** vectors. $\mathbf{h}_t$ is the output, as in the RNN. The new one is the **cell state** $\mathbf{c}_t$, a memory lane that runs along the whole sequence. At each word, three **gates** decide how the cell state changes:

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(W_f[\mathbf{x}_t, \mathbf{h}_{t-1}] + \mathbf{b}_f) &&\text{forget gate: how much old memory to keep}\\
\mathbf{i}_t &= \sigma(W_i[\mathbf{x}_t, \mathbf{h}_{t-1}] + \mathbf{b}_i) &&\text{input gate: how much new information to write}\\
\mathbf{g}_t &= \tanh(W_g[\mathbf{x}_t, \mathbf{h}_{t-1}] + \mathbf{b}_g) &&\text{candidate: the new information itself}\\
\mathbf{o}_t &= \sigma(W_o[\mathbf{x}_t, \mathbf{h}_{t-1}] + \mathbf{b}_o) &&\text{output gate: how much memory to reveal}\\[4pt]
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\end{aligned}
$$

($\odot$ means element-by-element multiplication.) A **gate** is just a sigmoid layer (Section 2.5): numbers between 0 and 1, one per memory slot, that act as dimmer switches. A gate of 1 lets everything through and a gate of 0 blocks it. The gates are computed from the current word and the previous output, so the network **learns when to remember and when to forget**.

![The LSTM cell: a cell-state highway along the top, gates underneath](figures/fig133_lstm_cell.png)

Look at the cell-state update, $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t$. There's **no weight matrix and no tanh** acting on $\mathbf{c}_{t-1}$, only a gate and an **addition**. Compare it with the two update rules we already know:

| | update | gradient through one step |
|---|---|---|
| RNN (Section 28) | $\mathbf{h}_t = \tanh(W\mathbf{h}_{t-1} + \dots)$ | $\text{diag}(1-\mathbf{h}_t^2)\,W$ |
| residual block (Section 20) | $\mathbf{h} \leftarrow \mathbf{h} + F(\mathbf{h})$ | $I + \partial F/\partial\mathbf{h}$ |
| **LSTM cell state** | $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t\odot\mathbf{g}_t$ | $\approx \text{diag}(\mathbf{f}_t)$ |

When the forget gate is near 1, the cell state is a **residual stream through time**, and the gradient flows back along it almost untouched. (Strictly, the gates also depend on $\mathbf{h}_{t-1}$, which adds extra terms, but the direct path through $\mathbf{f}_t$ is the one that matters.)

From scratch, all four gates come out of one linear layer:

```python
class MyLSTM(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.d_h = d_h
        self.lin = nn.Linear(d_in + d_h, 4 * d_h)          # all four gates in one matmul
    def forward(self, x):                                   # x: (batch, time, d_in)
        B, T, _ = x.shape
        h = torch.zeros(B, self.d_h); c = torch.zeros(B, self.d_h)
        hs = []
        for t in range(T):
            i, f, g, o = self.lin(torch.cat([x[:, t], h], 1)).chunk(4, 1)
            i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
            c = f * c + i * g                               # forget some old memory, add some new
            h = o * torch.tanh(c)                           # reveal part of it
            hs.append(h)
        return torch.stack(hs, 1), (h, c)
```

It matches `nn.LSTM` to $6\times10^{-8}$. The price is size: with 32-number inputs and 64 hidden units, an LSTM layer has **25,088** weights to the RNN's 6,272, four times as many, one set per gate.

---

### 29.2 The memory test, and a surprise

Back to Section 28.4's task: remember whether the first word was A or B through $T$ random words. This time everything gets a fairer budget (up to 3,000 steps, learning rate 0.003, two seeds, stopping early once it's solved):

![Memory task: plain RNN vs LSTM with three different forget-gate starting points](figures/fig134_memory_grid.png)

| test accuracy (seed 1 / seed 2) | 20 words | 40 | 80 | 160 |
|---|---|---|---|---|
| plain RNN | 100 / 100 | 50 / 100 | 51 / 100 | 50 / 50 |
| **LSTM, PyTorch's default start** | **49 / 50** | 49 / 50 | 51 / 50 | 51 / 50 |
| LSTM, forget-gate bias +1 | 100 / 100 | 48 / 50 | 52 / 50 | 51 / 50 |
| LSTM, forget-gate bias +3 | 100 / 100 | **100 / 100** | **99 / 100** | 50 / 50 |

The surprise is the second row. **An out-of-the-box LSTM fails at every length, even 20 words, where the plain RNN succeeds.** (With the bigger budget the RNN also gets lucky on one seed at 40 and 80 words, which it didn't in Section 28. Plain RNNs aren't *incapable* of long memory. They're **unreliable** at it.)

Why does the default LSTM fail? Its biases start at zero, so every gate starts at $\sigma(0) = 0.5$. A forget gate of 0.5 means the cell state **halves at every step**, the same decay rate we measured for the RNN's gradient. The highway exists, but it starts **half-closed**, and after 20 words only $0.5^{20} \approx$ one millionth of the clue is left.

The fix is one line: start the forget gate's bias at a positive number, so the gate starts near 1 and the highway starts **open**.

```python
H = 64
with torch.no_grad():
    lstm.bias_ih_l0[H:2*H].fill_(1.0)       # PyTorch orders the gates i, f, g, o: this slice is f
    lstm.bias_hh_l0[H:2*H].fill_(0.0)
```

| forget bias | forget gate at start | memory kept after 40 words |
|---|---|---|
| 0 (default) | 0.50 | $0.50^{40} \approx 10^{-12}$ |
| +1 | 0.73 | $0.73^{40} \approx 3\times10^{-6}$ |
| +3 | 0.95 | $0.95^{40} \approx 0.13$ |

With bias +3, the LSTM solves 40 and 80 words in **50 to 150 steps**, on every seed. At 160 words even that fails, because $0.95^{160} \approx 0.0003$ at the start. (Presumably a larger bias would push the limit further. I didn't test it.)

This is exactly Section 20's lesson, in time instead of depth: a residual path only helps if it **starts as the identity**. We started residual branches small (Section 20.3) so "do nothing" was the default. Here we start the forget gate near 1 so "remember everything" is the default. Jozefowicz, Zaremba & Sutskever (2015) found in a large search over recurrent architectures that simply adding a bias of 1 to the LSTM's forget gate closed most of the gap to the best architectures they found. It's the setting we use on IMDB.

> 📓 **Notebook rule:** *a highway only helps if it starts open.* An LSTM's forget gate at 0.5 forgets as fast as an RNN. Initialise it near 1, and gradients travel the whole sequence.

---

### 29.3 Measuring the highway

Same measurement as Section 28.5: how big is the gradient reaching each word?

![Gradient vs distance from the end: RNN vs LSTM, at initialisation and after training on IMDB](figures/fig135_lstm_gradients.png)

| gradient at a distant word ÷ gradient at the last word | plain RNN | **LSTM** |
|---|---|---|
| memory task at initialisation, 20 words back | 0.000001 | **0.01** (10,000× more) |
| memory task at initialisation, 100 words back | exactly 0 | $3\times10^{-9}$ |
| trained on IMDB, first of 200 words | 1 / 18,080 | **1 / 7** |

The trained IMDB LSTM (right panel) is the striking one. Its gradient is almost **flat across all 200 words**: the first word still receives a seventh of the learning signal of the last. The RNN's first word got one eighteen-thousandth. Inside the trained LSTM, the forget gates average 0.69, with 23% of them above 0.9. Some memory slots work as long-term storage and others are wiped quickly.

---

### 29.4 Reading the two sentences again

![The LSTM's P(positive) after each word of the two sentences](figures/fig136_lstm_reading.png)

| sentence | bag of words | RNN | **LSTM** |
|---|---|---|---|
| "i expected it to be great but it was terrible" | 36.6% | 9.9% | **11.4%** |
| "i expected it to be terrible but it was great" | 36.6% | 29.3% | **80.8%** |
| "this movie was good" | 48.0% | 42.0% | **76.1%** |
| "this movie was not good" | 41.2% | 27.3% | **28.4%** |

For the first time a model gets **both** "expected" sentences right. It's excited by "great" in the first sentence, then the final word flips it to 11%. In the second sentence it's gloomy after "terrible" until the final "great" lifts it to 81%. It also separates "good" (76%) from "not good" (28%) cleanly. ("Not bad" is still read as negative, 11%, which we already know is how IMDB reviewers use the phrase.)

---

### 29.5 IMDB: close, but still not past the bag

![Scoreboard: bag vs RNN vs LSTM vs GRU on the same words](figures/fig137_lstm_scoreboard.png)

| model | words read | best validation accuracy |
|---|---|---|
| bag of embeddings | last 200 | 89.4% |
| plain RNN (Section 28) | last 200 | 83.5% |
| **LSTM** (forget bias +1) | last 200 | 88.8% |
| **GRU** | last 200 | 89.5% |
| bag of embeddings | last 400 | **90.4%** |
| LSTM | last 400 | 89.9% |

The gates recover most of the 6-point gap the RNN lost. With 200 words, the **GRU** (Cho et al., 2014), a slimmer cousin with two gates instead of three and no separate cell state, ties the bag. Reading 400 words helps the LSTM (+1.1 points), and it helps the bag too. **Neither recurrent model beats a model that ignores word order**, and Section 26's word-pairs model (91.2%) still leads. So the test set stays closed again.

It's worth being clear about why:

1. **Most of the sentiment in a review is in its words, not their order.** The bag gets 90% with no order at all. Order-sensitive cases like "expected great… but terrible" are real but rare, so reading them correctly buys only a small amount of accuracy.
2. **The LSTM overfits fast.** Training accuracy reaches 99% by epoch 8 while validation stalls at 88.5%. It has far more ways to memorise 40,000 reviews than a bag does, and we gave it no dropout.
3. **It's slow.** An LSTM epoch took 26 seconds for 200 words and 48 for 400, against about 3 seconds for the bag, because each word has to wait for the one before it. That sequential bottleneck will matter a great deal soon.

Chollet reports the same pattern: on IMDB, a well-tuned bag of bigrams matches or beats recurrent models. Where RNNs and LSTMs really earned their place was in tasks where order **is** the meaning: translation, speech, generating text one word at a time. For those, a model must not only *read* a sequence but *produce* one.

---

### 📓 Notebook margin: the equation so far

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t
\qquad
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\qquad
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} \approx \text{diag}(\mathbf{f}_t)
$$

| idea | what we now know |
|---|---|
| cell state | a memory updated by gated **addition**: a residual stream through time |
| gates | sigmoid dimmer switches for forgetting, writing and revealing |
| default init | forget gate 0.5 halves the memory each step: 0 of 8 memory runs solved |
| forget bias +3 | solves 40 and 80 words in 50–150 steps, every seed |
| gradient flow | trained on IMDB: first word gets 1/7 of the last word's gradient (RNN: 1/18,080) |
| two sentences | 11% vs 81%: the first model to get both right |
| IMDB | LSTM 89.9% (400 words), GRU 89.5% (200): close to the bag, not past it |

---

### What comes next

Everything so far in Part V has read a sequence and produced **one number**. The problems that made recurrent networks famous produce a **sequence**: English in, French out. The standard design, the **encoder–decoder** (Sutskever et al., 2014), uses one LSTM to read the input and squeeze it into a single final state, and a second LSTM to write the output from that state.

**Section 30: The Bottleneck** builds one on a task we can train on a CPU and check exactly: reversing and transforming sequences of symbols. Then we'll measure what happens as the input gets longer, when a whole sentence has to fit through **one fixed-size vector**. That bottleneck is the problem **attention** was invented to solve (Bahdanau, Cho & Bengio, 2014), and with it begins Part VI.

---

*References: Sepp Hochreiter & Jürgen Schmidhuber (1997), "Long Short-Term Memory". Felix Gers, Jürgen Schmidhuber & Fred Cummins (2000), "Learning to Forget: Continual Prediction with LSTM" (the forget gate). Rafal Jozefowicz, Wojciech Zaremba & Ilya Sutskever (2015), "An Empirical Exploration of Recurrent Network Architectures" (forget-gate bias of 1). Kyunghyun Cho et al. (2014), "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation" (the GRU). Christopher Olah (2015), "Understanding LSTM Networks" (the cell-state-as-conveyor-belt picture). Rupesh Srivastava, Klaus Greff & Jürgen Schmidhuber (2015), "Highway Networks" (gated skip connections in depth, the bridge between LSTMs and ResNets). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 13–14 (LSTM and GRU layers; bag-of-bigrams vs sequence models on IMDB). All code in this series is PyTorch.*
