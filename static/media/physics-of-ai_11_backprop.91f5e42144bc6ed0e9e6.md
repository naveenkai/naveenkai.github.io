# Physics of AI — Part I · The Gears

## 11. The Chain Rule Is Backprop

> *One black box is left. Since Section 6, every training loop has called `loss.backward()` and received a slope for every one of the network's 218,618 weights, in under a millisecond. Section 9 showed we could get any single slope by nudging a weight and watching the loss, but doing that for every weight would be hopelessly slow. This section opens the last box. The method is called **backpropagation**, and underneath it's just the chain rule from school calculus, applied carefully and in the right order.*

---

### 11.1 Why "nudge and see" isn't enough

Section 9.2 measured a slope like this: nudge one weight, rerun the network, see how much the loss moved. It works, and it gave the same answer as autograd to 4 decimal places.

Now count the cost. The Section 7 network has **218,618** weights, so one full gradient by nudging means 218,618 extra forward passes, for **every** training step:

![One backward sweep vs 218,618 nudges](figures/fig49_cost.png)

| method, batch of 64 digits | time per step |
|---|---|
| one forward pass | 0.19 ms |
| forward + backprop: **all** 218,618 slopes | **0.65 ms** |
| nudge-and-see: one forward pass per weight | **≈ 41 seconds** |

Backprop gets every slope for about the price of **three and a half forward passes**. That's roughly 60,000× faster than nudging, and the gap grows with the size of the network. Without this trick, training anything bigger than a toy would be impossible.

---

### 11.2 The chain rule: exchange rates

Suppose 1 US dollar buys ₹84, and 1 euro buys $1.10. How many rupees per euro? You don't need a new rate. You **multiply the rates along the chain**:

$$
\frac{₹}{€} \;=\; \frac{₹}{\$}\times\frac{\$}{€} \;=\; 84 \times 1.10 \;=\; 92.4
$$

A derivative is also a rate: *how much does this change per unit change of that?* So for a chain of computations $w \rightarrow z \rightarrow a \rightarrow C$, the rate at which the loss changes with the weight is the product of the rates along the way:

$$
\frac{\partial C}{\partial w} \;=\; \frac{\partial C}{\partial a}\cdot\frac{\partial a}{\partial z}\cdot\frac{\partial z}{\partial w}
$$

That's the **chain rule**. Each factor is **local**: it only involves one small step of the computation, which is easy to work out on its own. Backprop is nothing more than computing those local rates and multiplying them together, starting from the loss and working back toward the weights.

---

### 11.3 One neuron, by hand

Take the movie neuron with one night of standardised inputs, $\mathbf{x} = [1.2,\, 0.5]$, weights $\mathbf{w} = [2,\, -3]$, bias $b = 0.5$, and the truth $y = 1$ (you did watch).

**Forward** (left to right, computing values):

$$
z = \mathbf{w}\cdot\mathbf{x} + b = 2(1.2) - 3(0.5) + 0.5 = 1.40
\qquad a = \sigma(1.40) = 0.802
\qquad C = -\log a = 0.220
$$

**Backward** (right to left, computing rates):

| step | local rate | running product $= \partial C / \partial(\cdot)$ |
|---|---|---|
| start at the loss | $\partial C/\partial C = 1$ | 1 |
| through $C = -\log a$ | $-1/a = -1.247$ | $\partial C/\partial a = -1.247$ |
| through $a = \sigma(z)$ | $\sigma'(z) = a(1-a) = 0.159$ | $\partial C/\partial z = -0.198$ |
| through $z = \mathbf{w}\cdot\mathbf{x} + b$ | $\partial z/\partial \mathbf{w} = \mathbf{x}$, $\;\partial z/\partial b = 1$ | $\partial C/\partial \mathbf{w} = [-0.237,\, -0.099]$, $\;\partial C/\partial b = -0.198$ |

![The one-neuron computation, forward and backward](figures/fig46_neuron_chain.png)

Notice $\partial C/\partial z = -0.198 = a - y$, exactly the shortcut from Section 8.4. Now let PyTorch check our arithmetic:

```python
w = torch.tensor([2., -3.], requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
x = torch.tensor([1.2, 0.5])

F.binary_cross_entropy_with_logits(w @ x + b, torch.tensor(1.)).backward()
w.grad, b.grad        # → [-0.2374, -0.0989], -0.1978      ✓ same as by hand
```

The gradient reads naturally, too: both slopes are negative, so both weights should *increase*. That pushes $z$ up, $a$ toward 1, and the prediction toward "you'll watch", which was the truth.

---

### 11.4 A tiny network, by hand

Now two inputs, two ReLU neurons, one output. Small enough to write every number down, big enough to show the two rules that make backprop work.

$$
\mathbf{x} = [1,\, 2], \quad
W_1 = \begin{bmatrix} 0.5 & -0.5 \\ 0.3 & 0.8 \end{bmatrix}, \quad
\mathbf{b}_1 = [0,\, -1], \quad
\mathbf{w}_2 = [1.5,\, -2], \quad
b_2 = 0.2, \quad y = 1
$$

![A 2-2-1 network, forward values and backward gradients](figures/fig47_tiny_network_graph.png)

**Forward:**

| quantity | value |
|---|---|
| $\mathbf{z}_1 = W_1\mathbf{x} + \mathbf{b}_1$ | $[-0.50,\; 0.90]$ |
| $\mathbf{h} = \text{ReLU}(\mathbf{z}_1)$ | $[0.00,\; 0.90]$, where the first neuron is **off** |
| $z_2 = \mathbf{w}_2\cdot\mathbf{h} + b_2$ | $-1.60$ |
| $a = \sigma(z_2)$, $\;C = -\log a$ | $0.168$, $\;1.784$ (confidently wrong) |

**Backward**, starting from the output:

1. **Output:** $\delta_2 = \partial C/\partial z_2 = a - y = -0.832$.
2. **Output weights:** $\partial C/\partial \mathbf{w}_2 = \delta_2\,\mathbf{h} = [0,\; -0.749]$, and $\partial C/\partial b_2 = -0.832$.
3. **Back into the hidden layer:** $\partial C/\partial \mathbf{h} = \delta_2\,\mathbf{w}_2 = [-1.248,\; 1.664]$.
4. **Through ReLU:** its slope is 1 where the neuron was on and **0 where it was off**, so $\boldsymbol{\delta}_1 = [0,\; 1.664]$. The first neuron's gradient is **blocked**. It had no effect on the output, so it gets no blame.
5. **Hidden weights:** $\partial C/\partial W_1 = \boldsymbol{\delta}_1\,\mathbf{x}^\top = \begin{bmatrix} 0 & 0 \\ 1.664 & 3.328 \end{bmatrix}$, and $\partial C/\partial \mathbf{b}_1 = [0,\; 1.664]$.
6. **All the way to the input:** $\partial C/\partial \mathbf{x} = W_1^\top\boldsymbol{\delta}_1 = [0.499,\; 1.331]$.

PyTorch agrees with every one of those numbers:

```python
# autograd:  g_W1 = [[0, 0], [1.664, 3.328]]   g_b1 = [0, 1.664]
#            g_w2 = [0, -0.749]                g_b2 = -0.832
#            dC/dx = [0.499, 1.331]
```

The two rules that did all the work:

> 📓 **Rule 1: multiply along a path.** Each step back multiplies by that step's local rate (a weight, or an activation's slope).
>
> 📓 **Rule 2: add across paths.** When a value feeds into several places (every input here feeds *both* hidden neurons), its gradient is the **sum** of the gradients coming back from each place.

Rule 2 is exactly what $W_1^\top\boldsymbol{\delta}_1$ does in step 6. Each input collects blame from every neuron it fed, weighted by how strongly it fed them. Transposing $W$ turns "who did I send signal to?" into "who do I collect blame from?".

---

### 11.5 The whole algorithm in four lines

Put the steps from 11.4 into matrix form and they work for any number of layers and any batch size. With $\mathbf{h}_0 = \mathbf{x}$, hidden layers $\mathbf{h}_\ell = \text{ReLU}(W_\ell\mathbf{h}_{\ell-1} + \mathbf{b}_\ell)$, and softmax + cross-entropy at the end:

$$
\begin{aligned}
&\text{(1) output error:} && \boldsymbol{\delta}_L = \mathbf{p} - \mathbf{y}_{\text{one-hot}} \\
&\text{(2) weight gradients:} && \frac{\partial C}{\partial W_\ell} = \boldsymbol{\delta}_\ell\,\mathbf{h}_{\ell-1}^{\top}, \qquad \frac{\partial C}{\partial \mathbf{b}_\ell} = \boldsymbol{\delta}_\ell \\
&\text{(3) send error back through the weights:} && \mathbf{g} = W_\ell^{\top}\boldsymbol{\delta}_\ell \\
&\text{(4) and back through the activation:} && \boldsymbol{\delta}_{\ell-1} = \mathbf{g} \odot \text{ReLU}'(\mathbf{z}_{\ell-1})
\end{aligned}
$$

Repeat (2)–(4) from the last layer to the first. ($\odot$ means element-wise multiply, and $\text{ReLU}'$ is just "1 if the neuron was on, else 0".) Nielsen's chapter 2 calls these "the four fundamental equations of backpropagation", written there for sigmoid neurons.

Here it is in PyTorch **without autograd**, just tensors and the four lines:

```python
def forward(params, X):
    hs, zs = [X], []
    for l, (W, b) in enumerate(params):
        z = hs[-1] @ W.T + b;  zs.append(z)
        hs.append(torch.relu(z) if l < len(params) - 1 else z)   # no ReLU on the output
    return hs, zs

def backward(params, hs, zs, Y):
    p = torch.softmax(hs[-1], dim=1)
    delta = (p - F.one_hot(Y, 10).float()) / len(Y)                # (1)
    grads = [None] * len(params)
    for l in reversed(range(len(params))):
        W, b = params[l]
        grads[l] = [delta.T @ hs[l], delta.sum(0)]                   # (2)
        if l > 0:
            delta = (delta @ W) * (zs[l-1] > 0).float()              # (3) and (4)
    return grads
```

**Check it against autograd** on the same 218,618-weight MNIST network and a batch of 64 digits:

```
largest difference, over all 218,618 gradients:  7.5e-09
```

That's floating-point rounding. It's the same algorithm.

**Now train with it.** Here's a plain SGD loop where every gradient comes from our own `backward`, with no `loss.backward()` anywhere:

```python
for epoch in range(20):
    for idx in torch.randperm(4000).split(64):
        hs, zs = forward(params, Xtr[idx])
        grads  = backward(params, hs, zs, Ytr[idx])
        for (W, b), (gW, gb) in zip(params, grads):
            W -= 0.1 * gW;  b -= 0.1 * gb
```

| after epoch | 1 | 5 | 20 |
|---|---|---|---|
| test accuracy | 26.7% | 79.3% | **93.4%** |

Total time: **1.3 seconds**. And 93.4% is exactly what Section 10's batch-64 run got using PyTorch's optimizer, with the same starting weights and the same shuffles. It's the same computation, now written out in full.

---

### 11.6 Why one backward sweep is enough

Why is backprop so much cheaper than nudging each weight? Look at step (3). The error $\boldsymbol{\delta}_\ell$ is computed **once** per layer and then **shared** by every weight in that layer and by every layer below it.

Nudging asks 218,618 separate questions of the form "what if I change *this* one weight?", and each question reruns the whole network. Backprop asks one question, "how much does the loss care about each **intermediate value**?", and answers it layer by layer, reusing each answer for the layer before. The work is a handful of matrix multiplies, the same kind and size as the forward pass. That's why the measured cost is about 2.5 forward passes, not 218,618.

> 📓 **Notebook rule:** *forward computes values, backward computes blame, and both cost about the same.* That efficiency is the whole reason large networks can be trained at all.

---

### 11.7 What `loss.backward()` actually does

PyTorch does all this automatically, and now you know what "automatically" means. During the forward pass, every operation on a tensor with `requires_grad=True` is **recorded**, along with how to compute its local rate. You can walk the record yourself:

```python
w = torch.tensor([2., -3.], requires_grad=True)
L = F.binary_cross_entropy_with_logits(w @ x + 0.5, torch.tensor(1.))

# follow L.grad_fn back to the weight:
# BinaryCrossEntropyWithLogitsBackward0 <- AddBackward0 <- DotBackward0 <- AccumulateGrad
```

That's the computation graph from 11.3, stored backwards. `loss.backward()` walks it from the loss to the weights, multiplies local rates along each path, adds across branches, and leaves the result in each weight's `.grad`. It's **exactly** what we did by hand in 11.4, and what our 15-line `backward` did in 11.5.

---

### 11.8 A warning hiding in rule 1

Rule 1 says gradients get **multiplied** at every step back. So what happens in a deep network if every step multiplies by something small?

Remember from Section 6.5 that the sigmoid's slope is **at most 0.25**. Here's the size of the weight gradient at each layer of a 10-layer network, sigmoid vs ReLU, same data and one backward pass:

![Gradient size per layer, sigmoid vs ReLU](figures/fig48_vanishing.png)

| | layer 10 (near output) | layer 1 (near input) | shrink factor |
|---|---|---|---|
| sigmoid | $5.7\times10^{-2}$ | $3.4\times10^{-9}$ | **≈ 17 million×** |
| ReLU | $9.0\times10^{-3}$ | $1.6\times10^{-4}$ | ≈ 55× |

With sigmoid, the first layer gets a learning signal **17 million times weaker** than the last. It effectively doesn't learn at all. This is the **vanishing gradient problem**, the main reason deep networks were so hard to train for decades, and one of the biggest reasons ReLU took over.

ReLU is far better but still not perfect: the signal still shrinks about 55× over 10 layers with default settings. The fixes (careful weight initialisation, normalisation layers, and the "skip connections" that made 100-layer networks possible) are the story of **Part III**. Nielsen's chapter 5, "Why are deep neural networks hard to train?", is entirely about this.

---

### 📓 Notebook margin: the equation so far

$$
\underbrace{\boldsymbol{\delta}_L = \mathbf{p} - \mathbf{y}}_{\text{blame at the output}}
\qquad
\underbrace{\boldsymbol{\delta}_{\ell-1} = \big(W_\ell^{\top}\boldsymbol{\delta}_\ell\big) \odot \text{ReLU}'(\mathbf{z}_{\ell-1})}_{\text{pass blame backward}}
\qquad
\underbrace{\frac{\partial C}{\partial W_\ell} = \boldsymbol{\delta}_\ell\,\mathbf{h}_{\ell-1}^{\top}}_{\text{blame × what came in}}
$$

| idea | what we now know |
|---|---|
| chain rule | rates multiply along a chain, like exchange rates |
| backprop, rule 1 | **multiply** local rates along each path back from the loss |
| backprop, rule 2 | **add** across paths wherever a value was used more than once |
| ReLU in backprop | a gate: passes blame where the neuron was on, blocks it where off |
| cost | one backward sweep ≈ 2.5 forward passes vs 218,618 for nudging |
| from scratch | our 15-line backprop matches autograd to $10^{-8}$ and trains MNIST to 93.4% |
| vanishing gradients | many small factors multiplied: sigmoid loses ~17 million× over 10 layers |

**Every black box from Part I is now open.** Data becomes tensors (3), gets scaled (4), gets moved and folded by layers (5–7), is scored by a loss (8), and the weights walk downhill (9) on mini-batch estimates (10) of gradients computed by backprop (11).

---

### What comes next

**Section 12: Build It** closes Part I by putting everything together on MNIST, three ways:

1. **From scratch:** our own layers, our own backprop, our own SGD. Every line is something we derived.
2. **With autograd:** we keep our model but let `loss.backward()` do section 11's job.
3. **The PyTorch way:** `nn.Module`, `DataLoader`, `optim`: what real code looks like, with every piece now familiar.

Then we'll compare all three against Nielsen's 95.4% (NumPy, sigmoid, 2015) and Chollet's 97.8% (Keras), train on the full 60,000 digits, and look at the digits our network still gets wrong.

---

*References: Michael Nielsen, *Neural Networks and Deep Learning*, ch. 2 ("How the backpropagation algorithm works": the four fundamental equations; why backprop is fast compared with computing each partial derivative separately) and ch. 5 (the vanishing gradient problem). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 2 ("The chain rule", "Automatic differentiation with computation graphs", the backward pass as reverse traversal of the graph). All code in this series is PyTorch.*
