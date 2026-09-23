# Physics of AI — Part I · The Gears

## 12. Build It

> *Eleven sections ago we couldn't say what a neuron was. Now every piece of a working network has been derived, drawn, and checked against PyTorch. This last section of Part I puts the pieces together on the **full** MNIST dataset (60,000 training digits, 10,000 test digits) three times over: first with nothing but our own maths, then with autograd, then the way real PyTorch code is written. Then we compare our scores with Nielsen's and Chollet's, and look closely at what the network still gets wrong.*

---

### 12.1 The data (Sections 3 and 4)

```python
from torchvision import datasets

train = datasets.MNIST("data", train=True,  download=True)
test  = datasets.MNIST("data", train=False, download=True)

train.data.shape, train.data.dtype      # → (60000, 28, 28), torch.uint8
train.targets[:10]                      # → [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]

Xtr = train.data.reshape(-1, 784).float() / 255      # flatten (3.5), scale to [0, 1] (4.7)
Ytr = train.targets
Xte = test.data.reshape(-1, 784).float() / 255
Yte = test.targets
```

Every choice here has a reason we've already seen: `uint8 → float32` so the network can do arithmetic (3.3), `/ 255` so each $z$ stays on the slope of the activation (4.7), and flattening so a digit becomes one point in 784-D space (3.5).

The network is the same for all three builds: **784 → 128 → 64 → 10**, ReLU between the layers (Section 6), softmax + cross-entropy at the end (Section 8). Each build trains for **10 epochs** with **mini-batches of 64** (Section 10).

---

### 12.2 Way 1: from scratch

No autograd, no `nn`, no optimiser. Tensors and the maths from Section 11 only:

```python
def init(sizes, seed=0):
    g = torch.Generator().manual_seed(seed); params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        bound = 1 / fan_in**0.5                                   # same default PyTorch uses
        W = (torch.rand(fan_out, fan_in, generator=g) * 2 - 1) * bound
        b = (torch.rand(fan_out, generator=g) * 2 - 1) * bound
        params.append([W, b])
    return params

params = init([784, 128, 64, 10])
for epoch in range(10):
    for idx in torch.randperm(60000).split(64):                   # Section 10
        hs, zs = forward(params, Xtr[idx])                        # Sections 5–6
        grads  = backward(params, hs, zs, Ytr[idx])               # Section 11
        for (W, b), (gW, gb) in zip(params, grads):
            W -= 0.1 * gW;  b -= 0.1 * gb                          # Section 9
```

`forward` and `backward` are the exact functions from Section 11.5, unchanged.

| | test accuracy | training accuracy | time (CPU) |
|---|---|---|---|
| **Way 1, from scratch** | **96.69%** | 98.17% | 6.4 s |

**96.69% on digits it has never seen, from code where we can explain every line.**

---

### 12.3 Way 2: let autograd do Section 11's job

Same parameters, same loop, but we stop computing gradients ourselves and just call `backward()`:

```python
params = [[W.requires_grad_(), b.requires_grad_()] for W, b in init([784, 128, 64, 10])]

def model(X):
    h = X
    for l, (W, b) in enumerate(params):
        h = h @ W.T + b
        if l < len(params) - 1:
            h = torch.relu(h)
    return h

for epoch in range(10):
    for idx in torch.randperm(60000).split(64):
        loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
        for W, b in params: W.grad = b.grad = None
        loss.backward()                                          # ← replaces our backward()
        with torch.no_grad():
            for W, b in params: W -= 0.1 * W.grad;  b -= 0.1 * b.grad
```

| | test accuracy | time |
|---|---|---|
| Way 2, autograd | **96.42%** | 8.0 s |

Wait: Way 1 got 96.69%. Same starting weights, same shuffles, same algorithm. Why isn't it identical?

Tracking the two runs side by side shows the answer. After step 1 the weights differ by $7\times10^{-9}$, after 100 steps by $7\times10^{-8}$, and after the first epoch (938 steps) by $0.002$. The only difference is **floating-point rounding**: our softmax and PyTorch's `cross_entropy` do the same maths in a slightly different order. Those differences start at the eighth decimal place and compound over thousands of steps, like two identical balls released a hair apart on a bumpy hill.

> 📓 **Notebook rule:** *a difference of 0.3% between two runs is noise, not a result.* Before you believe that one model beats another, run each a few times. (This bites real papers more often than you'd think.)

---

### 12.4 Way 3: the PyTorch way

This is what you'd actually write. Nothing in it is new any more. It's the same pieces in PyTorch's standard packaging:

```python
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):                                   # a model is a class with a forward()
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                               # (N, 28, 28) → (N, 784)       Section 3.5
            nn.Linear(784, 128), nn.ReLU(),             # move, fold                   Sections 5–6
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 10))                          # 10 scores (logits)           Section 8.5
    def forward(self, x):
        return self.net(x)

train_dl = DataLoader(TensorDataset(train.data.float() / 255, train.targets),
                      batch_size=64, shuffle=True)      # shuffle + split every epoch   Section 10.3
test_dl  = DataLoader(TensorDataset(test.data.float() / 255, test.targets), batch_size=1000)

model = MLP()
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)   # SGD + momentum + per-weight steps  10.6

for epoch in range(10):
    for x, y in train_dl:
        opt.zero_grad()                                 # clear old gradients
        loss = F.cross_entropy(model(x), y)             # how wrong?                    Section 8
        loss.backward()                                 # backprop                      Section 11
        opt.step()                                      # step downhill                 Section 9

def evaluate(model, dl):
    model.eval(); correct = n = 0
    with torch.no_grad():                               # no graph needed for scoring
        for x, y in dl:
            correct += (model(x).argmax(1) == y).sum().item(); n += len(y)
    model.train(); return correct / n
```

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 95.04 | 96.33 | 97.80 | 98.23 | 98.61 | 98.81 | 99.07 | 99.22 | 99.49 | 99.41 |
| **test** | 94.77 | 96.01 | 97.09 | 97.12 | 97.41 | 97.63 | 97.43 | 97.68 | 97.70 | **97.87** |

**97.87% on the test set.** The one real change from Way 1 is the optimiser: Adam instead of plain SGD. Its momentum and per-weight step sizes (Section 10.6) are worth over a full percentage point here.

---

### 12.5 The scoreboard

Now put everything in one place. Solid bars are runs we did in this series, and hatched bars are numbers reported in the two books:

![Part I scoreboard](figures/fig50_scoreboard.png)

| model | test accuracy | notes |
|---|---|---|
| random guessing | 10.00% | the floor |
| "darkness" only | **22.30%** | Nielsen's baseline idea (guess the digit whose average ink is closest); he reports 22.25% |
| one layer, no hidden neurons | 92.11% | Section 2's neuron ×10 with softmax |
| **Nielsen-style network, rerun in PyTorch** | **95.01%** (best epoch) | 784→**30**→10, **sigmoid**, **squared error**, SGD η = 3, batch 10, 30 epochs, trained on 50,000 |
| Nielsen's reported result | 95.42% | his book, chapter 1 |
| Way 1: from scratch | 96.69% | ReLU, cross-entropy, plain SGD |
| Way 2: autograd | 96.42% | same algorithm, different rounding |
| **Way 3: PyTorch + Adam** | **97.87%** | |
| Chollet's reported result | 97.8% | his book, chapter 2 |
| best result Nielsen cites (Wan et al., 2013) | 99.79% | convolutional networks, Part IV territory |

Three things this table shows:

1. **We reproduced both books.** Our rerun of Nielsen's exact recipe lands at 95.0% against his 95.4% (most likely just a different random start; see 12.3 on run-to-run noise). Way 3 lands at 97.87% against Chollet's 97.8%.
2. **The 2.9-point gap between them is Part I's lessons.** Going from Nielsen's 2015 recipe to Way 3 means ReLU instead of sigmoid (Section 6.5), cross-entropy instead of squared error (Section 8.3), wider hidden layers, and Adam (Section 10.6). Every one of those choices was explained by a picture somewhere in this series.
3. **The rest of the way to 99.79% needs new ideas.** Flattening a digit into 784 numbers throws away which pixels are neighbours (Section 3.5). Getting that back is what convolutional networks do in Part IV.

---

### 12.6 What it still gets wrong

Way 3 misclassifies **213** of the 10,000 test digits. Here are the 24 it was *most* sure about:

![The 24 most confident mistakes](figures/fig52_mistakes.png)

Be honest looking at these. Some are digits most people would also struggle with: a 6 lying on its side, a 5 that's mostly a loop, a 2 that could be a sloppy 8. A few are *clearly* readable to us but not to the network, like the tall, clean 3 that the network called a 2. And some look like near-copies of each other: MNIST's test set contains a few almost-duplicate writing samples, and the network makes the same mistake on both.

Where do the mistakes go?

![Confusion between true digit and predicted digit (errors only)](figures/fig53_confusion.png)

| most confused pairs (both directions) | mistakes |
|---|---|
| **4 ↔ 9** | 19 |
| **7 ↔ 9** | 15 |
| 3 ↔ 9 · 5 ↔ 6 · 3 ↔ 5 · 2 ↔ 8 | 11 each |

At the top is our old friend from Section 7.4, **4 vs 9**. Uncrumpling helped a lot, but the two sheets still touch in places.

One more thing to notice: the network was **100% sure** of most of these wrong answers. It has no sense of "I don't know". Being confidently wrong is a real problem in practice (think of a medical scan), and we'll come back to it.

---

### 12.7 The gap

One more picture from Way 3's training log:

![Training vs test accuracy per epoch](figures/fig51_train_vs_test.png)

By epoch 10 the network gets **99.41%** of the training digits right, but only **97.87%** of the test digits. On digits it has seen, it's nearly perfect. On new digits it's noticeably worse, and the gap **widens** as training goes on: training accuracy keeps climbing while test accuracy flattens out.

Part of what the network learned is *"what digits look like"*, and part of it is *"what **these particular** 60,000 digits look like"*. The second part is **memorising**, and it doesn't carry over to new handwriting. That gap is the starting point of Part II.

---

### 12.8 Part I in one loop

Here's the whole of Part I folded into the ten lines that train a network:

```python
X = images.reshape(-1, 784).float() / 255                # 3 · the world is numbers, 4 · fair scale
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(),    # 2 · neurons, 5 · layers move space,
                      nn.Linear(128, 64),  nn.ReLU(),    # 6 · ReLU folds it, 7 · depth uncrumples
                      nn.Linear(64, 10))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)      # 9 · downhill, 10 · momentum
for epoch in range(10):
    for idx in torch.randperm(len(X)).split(64):         # 10 · random spoonfuls
        loss = F.cross_entropy(model(X[idx]), Y[idx])    # 8 · measure wrongness (softmax + −log p)
        opt.zero_grad(); loss.backward()                 # 11 · the chain rule, backwards
        opt.step()                                       # 9 · one step down the valley
```

When we started, those lines would have looked like magic. Now every one of them has a picture behind it.

---

### 📓 Notebook margin: Part I, the whole equation

$$
\underbrace{\mathbf{x} = \tfrac{\text{pixels}}{255}}_{3,\,4}
\;\longrightarrow\;
\underbrace{\mathbf{h}_\ell = \text{ReLU}(W_\ell\mathbf{h}_{\ell-1} + \mathbf{b}_\ell)}_{2,\,5,\,6,\,7}
\;\longrightarrow\;
\underbrace{C = -\log\,\text{softmax}(\mathbf{s})_{\text{true}}}_{8}
\;\longrightarrow\;
\underbrace{\mathbf{w} \leftarrow \mathbf{w} - \eta\,\nabla C}_{9,\,10,\,11}
$$

| build | test accuracy | what it proves |
|---|---|---|
| from scratch | 96.69% | we understand every line |
| autograd | 96.42% | `backward()` is Section 11; run-to-run noise is ~0.3% |
| PyTorch + Adam | **97.87%** | matches Chollet's 97.8% |
| Nielsen's recipe | 95.01% | matches Nielsen's 95.4%; the 2.9-point gap is Part I's lessons |

---

## End of Part I

We now have a machine that learns. Part I's question was *how*. **Part II: Making It Learn** asks something harder: *how do we make it learn the **right** thing?*

It starts from the gap in 12.7:

- **Section 13: Memorising vs Learning.** What overfitting really is, why a validation set exists, and a network that memorises random labels perfectly.
- Then the toolkit for closing the gap: more data and augmentation, weight decay, dropout, early stopping, better initialisation, better optimisers and learning-rate schedules, and batch normalisation.

---

*References: Michael Nielsen, *Neural Networks and Deep Learning*, ch. 1 (the 784-30-10 network, its 95.42% result, the darkness and SVM baselines, the 99.79% record by Wan et al.). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 2 (the first MNIST network and its 97.8% test accuracy; "Looking back at our first example"; reimplementing it from scratch). All code in this series is PyTorch; all "solid bar" numbers were produced by the code accompanying this section.*
