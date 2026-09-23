# Physics of AI — Part V · Remembering

## 28. Reading in Order

> *Sections 26 and 27 read a review as a bag: which words are there, never in what order. This section builds the first network that reads the way we do, one word after another, carrying a **memory** from each word to the next. It's the Section 2 neuron again, with one new wire: its own previous output fed back in. We'll build it from scratch, watch it change its mind mid-sentence, and then measure its fatal flaw, which turns out to be an old friend from Part III.*

---

### 28.1 One neuron, fed its own output

Take a layer of neurons. At step $t$ it receives two things: the embedding $\mathbf{x}_t$ of word $t$ (Section 27), and its **own output from the previous step**, $\mathbf{h}_{t-1}$. It combines them the only way we know, an affine map and a nonlinearity (Sections 5–6):

$$
\mathbf{h}_t = \tanh\big(W\,\mathbf{h}_{t-1} + U\,\mathbf{x}_t + \mathbf{b}\big), \qquad \mathbf{h}_0 = \mathbf{0}
$$

$\mathbf{h}_t$ is the **hidden state**: a running summary of everything read so far. After the last word, a linear layer turns $\mathbf{h}_T$ into P(positive). This is a **recurrent neural network (RNN)** (Elman, 1990).

![An RNN unrolled over four words: the same cell, reused at every step](figures/fig129_rnn_unrolled.png)

The crucial detail is that **$W$, $U$ and $\mathbf{b}$ are the same at every step**. That's weight sharing again, like the convolution in Section 22, but across **time** instead of space. A convolution reuses one kernel at every position of an image. An RNN reuses one cell at every position of a sentence, so it handles a 7-word review and a 2,000-word review with the same weights.

From scratch, it's a for loop:

```python
class MyRNN(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.U = nn.Linear(d_in, d_h)                 # reads the new word
        self.W = nn.Linear(d_h, d_h, bias=False)      # reads the memory
        self.d_h = d_h
    def forward(self, x):                             # x: (batch, time, d_in)
        h = torch.zeros(x.shape[0], self.d_h)         # h_0 = 0: an empty memory
        hs = []
        for t in range(x.shape[1]):                   # one word at a time, in order
            h = torch.tanh(self.W(h) + self.U(x[:, t]))
            hs.append(h)
        return torch.stack(hs, 1), h                  # every hidden state, and the last one
```

It matches PyTorch's `nn.RNN` to $7\times10^{-8}$. The classifier is: embedding (32 numbers per word) → RNN (64 hidden units) → linear layer on the last hidden state.

**Practical detail.** Batches need sequences of equal length, so we keep **200 words** per review (59% of reviews are shorter) and **pad** short ones with the `[PAD]` token on the *left*. That way the last step is always a real word, and the final hidden state has just read the end of the review.

---

### 28.2 Watching it read

Here's what a bag can't do. Feed the trained RNN our two sentences from Section 26 and read off P(positive) **after every word**, by applying the output layer to each $\mathbf{h}_t$:

![P(positive) after each word, for two sentences with the same words in a different order](figures/fig130_running_prediction.png)

For five words the lines are identical, because the sentences are. Then "great" pushes one reading up to 68%, and "terrible" pushes the other down to 9%. After "but", the two lines keep moving, and the final word pulls each one back the other way:

| sentence | bag of words (Section 26) | **RNN** |
|---|---|---|
| "i expected it to be great but it was terrible" | 36.6% | **9.9%** |
| "i expected it to be terrible but it was great" | 36.6% | **29.3%** |

For the first time, the **same words in a different order give different answers**, and the ordering is the right way round: the disappointment is judged far more negative than the pleasant surprise. (The pleasant surprise is still below 50%, a sign this memory isn't very good. We'll see why.)

---

### 28.3 On IMDB: memory loses to the bag

Now the real test: 39,582 reviews, the last 200 words of each, Adam, 12 epochs (about 20 seconds each on 2 CPUs), gradient clipping at norm 1 (see 28.6). To be fair, the bag baseline gets **exactly the same 200 words**:

![Validation accuracy per epoch: RNN vs bag of embeddings on the same words](figures/fig132_rnn_imdb.png)

| model, same 200 words per review | best validation accuracy |
|---|---|
| bag of embeddings (Section 27), last 200 words | **89.4%** |
| RNN, last 200 words | 83.5% |
| bag of embeddings, first 200 words | 88.5% |
| RNN, first 200 words | 80.1% |

**The model that reads in order is 6 points worse than the model that ignores order.** Its validation curve is also jumpy, moving between 79.7% and 83.5% in the last six epochs. Two more clues:

- **It prefers the end of the review.** Keeping the first 200 words costs the bag 0.9 points (reviews often end with a verdict), but it costs the RNN 3.4 points.
- **Most of its accuracy comes from the last few dozen words.** Give the trained RNN only the last $k$ words of each review:

| last $k$ words given | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| RNN validation accuracy | 69.8% | 74.6% | 79.4% | 82.1% | 83.5% |

Doubling from 100 to 200 words adds only 1.4 points. Something is stopping information from far back reaching the decision.

(The test set stays closed. Nothing here beats Section 26's 90.26%.)

---

### 28.4 A cleaner test: remember one word

IMDB is messy, so here's a task that measures memory and nothing else. Each sequence starts with a **clue** word (A or B), followed by $T$ random filler words. The label is simply: was the clue A? The answer is in the very first word, and the network has to carry it through $T$ steps of noise to the end.

```python
first = torch.randint(0, 2, (B,))                         # the clue: 0 or 1
rest  = torch.randint(3, 13, (B, T))                      # T random filler words
x = torch.cat([(first + 1)[:, None], rest], 1)            # clue first, then noise
label = first                                             # remember the first word
```

Same RNN recipe (hidden size 32, 1,500 steps, 3 random seeds for each $T$):

![Left: accuracy vs distance to the clue. Middle and right: gradient size vs distance from the end](figures/fig131_vanishing_time.png)

| words between clue and answer | 5 | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|---|
| RNN accuracy (3 seeds) | 100, 100, 100 | 100, 100, 100 | 100, 50, 50 | 51, 50, 50 | 50, 49, 49 | 50, 51, 50 |

Up to 10 words it's perfect. At 20 words, one seed in three manages it. From 40 on it's **coin-flipping**. The task is trivial for a person, one bit of memory, yet the RNN can't learn it across 40 words.

---

### 28.5 Why: 200 steps is 200 layers

Unroll the RNN over $T$ words and it's a $T$-layer network, one layer per word, **with the same weights in every layer**. Training it is ordinary backprop through that unrolled network, called **backpropagation through time**. The chain rule (Section 11) says the gradient reaching word $t$ from the end passes through every step in between:

$$
\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \;=\; \prod_{s=t+1}^{T} \frac{\partial \mathbf{h}_s}{\partial \mathbf{h}_{s-1}}
\;=\; \prod_{s=t+1}^{T} \text{diag}\big(1 - \mathbf{h}_s^2\big)\, W
$$

A product of $T - t$ factors. This is **Section 15's unstable product** and **Section 20's degradation problem** again, now running along the sentence instead of down the layers. And it's harder here, because every factor contains the *same* $W$, so an effect that's slightly too small at each step compounds with nothing to break it.

**Measured at initialisation** (middle panel, the memory task with 100 words): the gradient reaching a word **roughly halves with every step back**. Ten words back it's a thousandth of the gradient at the last word. Twenty words back it's a millionth. Past about 75 words it's **exactly zero**: the number is too small for 32-bit floats to hold. No learning signal about the clue arrives at all, which is why 28.4's accuracy fell to 50%.

**Measured on the trained IMDB model** (right panel), for the sensitivity of the output to each word's embedding across 1,000 full-length validation reviews:

| | gradient size |
|---|---|
| last word vs 50 words back | 10× larger |
| last word vs first word (200 back) | **18,080× larger** |

Why doesn't training fix it by making $W$ larger? It tried. The largest stretch factor (singular value) of the trained $W$ is **3.9**, well above 1. But look at the other factor, $1 - h^2$, the slope of tanh. **62%** of the trained hidden values sit above 0.95 in size, where tanh is flat and its slope is under 0.1. A bigger $W$ pushes the neurons into tanh's flat tails, which kills the gradient anyway. It's Section 6's saturation problem, compounded over 200 steps.

> 📓 **Notebook rule:** *a recurrent network is a very deep network with shared weights.* Every word it has to remember adds a layer between that word and the answer, and a plain recurrent cell has no residual path (Section 20) for the gradient to travel along.

---

### 28.6 The opposite failure: exploding gradients

Products of many factors can also **blow up** (Section 15). A standard safety measure for RNNs is **gradient clipping** (Pascanu et al., 2013): if the gradient vector's total length exceeds a threshold, scale it down to that length and keep its direction:

```python
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # shrink the step if it's too long
opt.step()
```

On our IMDB RNN it's a precaution more than a rescue. Over one epoch on 12,000 reviews without clipping, the median gradient norm was 0.35, the largest was 5.7, and only 2 of 188 steps were more than 10× the median. We clip in every run anyway, since one bad step can wreck a run. Clipping fixes steps that are **too big**. It can't do anything about gradients that are **too small**, and that is the problem 28.4 and 28.5 found.

---

### 📓 Notebook margin: the equation so far

$$
\mathbf{h}_t = \tanh\big(W\,\mathbf{h}_{t-1} + U\,\mathbf{e}_{w_t} + \mathbf{b}\big)
\qquad
P(\text{positive}) = \sigma(\mathbf{v}\cdot\mathbf{h}_T + c)
\qquad
\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{s=t+1}^{T}\text{diag}(1-\mathbf{h}_s^2)\,W
$$

| idea | what we now know |
|---|---|
| recurrence | one cell reused at every word; the hidden state is a running memory |
| order matters | "great … terrible" 9.9% vs "terrible … great" 29.3% (the bag gave 36.6% for both) |
| IMDB | RNN 83.5% vs bag 89.4% on the same 200 words |
| memory task | perfect at 10 words, chance by 40 |
| backprop through time | gradient roughly halves per word at init; exactly 0 beyond ~75 words |
| trained model | last word gets 18,080× the gradient of the first; 62% of hidden units saturated |
| clipping | fixes exploding steps; can't fix vanishing ones |

---

### What comes next

Section 20 solved vanishing gradients in deep networks with one idea: **add**, don't replace. $\mathbf{h} + F(\mathbf{h})$ gave the gradient a clean highway. An RNN replaces its whole memory at every step, $\mathbf{h}_t = \tanh(\dots)$, so there's no highway.

**Section 29: Gates and Highways** builds the **LSTM** (Hochreiter & Schmidhuber, 1997). It keeps a separate **cell state** that's updated by *adding* to it, with learned **gates** deciding what to forget, what to write and what to reveal. It's a residual connection through time, invented 18 years before ResNets. We'll build it from scratch, rerun the memory task, and see whether reading in order can finally beat the bag.

---

*References: Jeffrey Elman (1990), "Finding Structure in Time" (the simple recurrent network). Paul Werbos (1990), "Backpropagation Through Time: What It Does and How to Do It". Yoshua Bengio, Patrice Simard & Paolo Frasconi (1994), "Learning Long-Term Dependencies with Gradient Descent is Difficult". Sepp Hochreiter (1991), diploma thesis (the vanishing gradient in recurrent nets). Razvan Pascanu, Tomas Mikolov & Yoshua Bengio (2013), "On the difficulty of training recurrent neural networks" (gradient clipping). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 13 (recurrent layers, SimpleRNN's short memory) and ch. 14 (sequence models on IMDB). Michael Nielsen, *Neural Networks and Deep Learning*, ch. 5 (unstable gradients, which also apply to recurrent networks). All code in this series is PyTorch.*
