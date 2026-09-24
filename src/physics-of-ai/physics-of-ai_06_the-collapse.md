# Physics of AI — Part I · The Gears

## 6. The Collapse

> *Section 5 ended with an uncomfortable fact: stack two layers and you get one layer. Stack a hundred and you still get one. If that were the whole story, "deep" learning would be a single neuron with extra paperwork. This section shows the collapse happening on real data, then fixes it with one tiny function, and for the first time in this series we watch space **bend**.*

---

### 6.1 A pattern no line can split

Here's a new dataset: two interlocking crescents, a classic toy problem called "two moons". Think of it as any situation where the answer **curves**. Blue is one class and orange the other, and they hook into each other.

```python
import torch, torch.nn as nn
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=600, noise=0.15, random_state=0)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
X = (X - X.mean(0)) / X.std(0)                 # Section 4 habit: standardise first
Xtr, Xte, ytr, yte = X[:400], X[400:], y[:400], y[400:]
```

From Section 2 we know a single neuron draws a single straight line. No straight line can separate two hooks, so the best it can do is cut through the middle and accept some mistakes.

We need to **train** these models, so we'll use a short training loop as a black box for now. Sections 8–11 open it up piece by piece: the loss, the gradient, the step and backpropagation.

```python
def train(model, steps=3000, lr=0.01):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)   # "adjust the weights" (Section 10)
    lossf = nn.BCEWithLogitsLoss()                        # "how wrong are we"   (Section 8)
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(model(Xtr).squeeze(1), ytr)
        loss.backward()                                   # "which way is downhill" (Section 11)
        opt.step()
    with torch.no_grad():
        return ((model(Xte).squeeze(1) > 0).float() == yte).float().mean().item()
```

---

### 6.2 The collapse, on real data

Three contenders:

```python
one_neuron = nn.Linear(2, 1)

five_linear = nn.Sequential(                 # 5 layers, 3,297 weights, no activations
    nn.Linear(2, 32), nn.Linear(32, 32), nn.Linear(32, 32),
    nn.Linear(32, 32), nn.Linear(32, 1))

with_relu = nn.Sequential(                   # 2 hidden layers WITH ReLU in between
    nn.Linear(2, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 1))
```

| model | weights | test accuracy |
|---|---|---|
| 1 neuron | 3 | **89.5%** |
| 5 linear layers | 3,297 | **89.5%** |
| 2 layers + ReLU | 1,185 | **98.5%** |

![The collapse: 1 neuron vs 5 linear layers vs ReLU network](figures/fig21_collapse.png)

Look at the first two panels. **Over a thousand times more weights, exactly the same straight line, and exactly the same score.** The five-layer network *is* one neuron. It's Section 5.8's algebra ($W_2W_1 = W$) happening in front of you.

The third panel has *fewer* weights than the second, and its boundary curves around the hooks. The only difference is `nn.ReLU()` between the layers.

---

### 6.3 What ReLU does: it folds

ReLU, the **Re**ctified **L**inear **U**nit, is almost embarrassingly simple:

$$
\text{ReLU}(z) = \max(0, z)
$$

Positive numbers pass through unchanged and negative numbers become 0. It's applied **element-wise** (Section 5.3), separately to every neuron's output.

In one dimension that's just a bent line. The interesting part is what it does to **space**. Take a cloud of points (coloured so you can track where each one goes), apply an affine map, then ReLU:

![ReLU folds space onto the axes](figures/fig22_relu_fold.png)

- **Middle panel:** the affine step from Section 5. Rotated, sheared and slid, but every line is still straight.
- **Right panel:** ReLU. Everything that was left of the vertical axis gets pressed flat onto it, and everything below the horizontal axis gets pressed flat onto that. Space has been **folded**.

A fold is something no matrix can do. Matrices keep lines straight and parallel. A fold creates a **crease**, where the same line runs one way on one side and gets flattened on the other.

> 📓 **Notebook rule:** *a matrix can stretch, rotate, shear and slide. It can never fold. The activation function is the fold.*

---

### 6.4 Why the stack no longer collapses

The collapse in Section 5.8 depended on one property. Linear maps satisfy

$$
f(\mathbf{a} + \mathbf{b}) = f(\mathbf{a}) + f(\mathbf{b})
$$

and that's exactly why $W_2(W_1\mathbf{x})$ could be merged into $(W_2W_1)\mathbf{x}$.

Put a ReLU between them and the property breaks, because $\max(0, \cdot)$ doesn't distribute over addition. For example, $\text{ReLU}(-3 + 5) = 2$ but $\text{ReLU}(-3) + \text{ReLU}(5) = 5$. We can check this on the two trained networks (removing the bias part first, so we're testing the pure transformation):

```python
g = lambda f, v: f(v) - f(torch.zeros(1, 2))     # strip the constant offset
a, b = torch.randn(1, 2), torch.randn(1, 2)

# five_linear:  g(a+b) = 0.6743    g(a) + g(b) = 0.6743    ← linear. Collapses.
# with_relu:    g(a+b) = 30.63     g(a) + g(b) = 27.62     ← not linear. Can't be merged.
```

Now $W_2\,\text{ReLU}(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$ **cannot** be written as one matrix. Each layer adds something the previous one couldn't do, which means **depth finally means something**.

The full recipe for a layer is now complete:

$$
\boxed{\;\mathbf{h} = \text{ReLU}(W\mathbf{x} + \mathbf{b})\;}
\qquad
\text{affine (move space)} \;\rightarrow\; \text{nonlinear (fold it)}
$$

---

### 6.5 Why ReLU and not sigmoid?

We already had a nonlinearity: the sigmoid from Section 2 bends too, and Nielsen's book uses it throughout. So why does almost every modern network (and Chollet's book) use ReLU in the hidden layers?

**Look at the slopes**, because slopes are what learning runs on (Sections 2.6 and 4.6):

```python
z = torch.tensor([-6., -2., 0.5, 2., 6.], requires_grad=True)
# sigmoid slope: [0.002, 0.105, 0.235, 0.105, 0.002]
# ReLU slope:    [0.,    0.,    1.,    1.,    1.   ]
```

![Sigmoid vs ReLU and their slopes](figures/fig23_activation_slopes.png)

- **Sigmoid's slope is at most 0.25**, and it fades to almost nothing on both sides. That's the saturation we fought in Section 4. Stack many sigmoid layers and these small slopes **multiply** (we'll see exactly why in the backprop section), so the signal gets weaker at every layer.
- **ReLU's slope is exactly 1** for any positive input. It never saturates on that side, so the learning signal passes through at full strength.

ReLU has its own flaw: its slope is exactly **0** for negative inputs, so a neuron stuck in the negative region can stop learning (a "dead" neuron). We'll deal with that in Part II. For now, the rule of thumb is **ReLU inside, sigmoid (or softmax) only at the very end** when we need a probability.

---

### 6.6 Curves from straight pieces

ReLU is made of two straight pieces joined at a corner. So how does a network built from it draw a **curve**?

Each ReLU neuron adds **one crease** to space. More neurons means more creases, and enough straight segments joined at creases start to look like a curve. Here's one hidden layer with more and more neurons:

```python
for width in [1, 2, 4, 8, 32]:
    model = nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, 1))
    train(model)
```

![Decision boundaries for 1, 2, 4, 8, 32 ReLU neurons](figures/fig24_widths.png)

| hidden ReLU neurons | 1 | 2 | 4 | 8 | 32 |
|---|---|---|---|---|---|
| test accuracy | 89.5% | 89.5% | 91.5% | 91.5% | **99.0%** |

Read the boundaries left to right. **One** neuron gives a straight line, **two** give a bent line with one corner, **four** and **eight** give a polyline, and **thirty-two** trace the hook closely. Look carefully and the "curve" is still made of short straight pieces.

> 📓 **Notebook rule:** *a ReLU network draws with a ruler. Give it enough straight pieces and it can trace any shape.* (That "any" is a real theorem, the **universal approximation theorem**, and Part III is devoted to it.)

---

### 6.7 Watching space bend

Everything so far showed the **decision boundary** bending in the input space. But from Section 5 we know the network's real job is to **move space**. So what does the data look like *from inside the network*?

We give the network a 2-neuron layer just before the final output, so we can draw what it "sees" there:

```python
body = nn.Sequential(nn.Linear(2, 16), nn.ReLU(),
                     nn.Linear(16, 16), nn.ReLU(),
                     nn.Linear(16, 2))          # ← a 2-D "view" we can plot
head = nn.Linear(2, 1)                          # final decision: one straight line
model = nn.Sequential(body, head)               # trained: 98.0% test accuracy
```

Now we move every point, and the grid lines, from where they start (input space) to where `body` sends them:

![Space bending inside a ReLU network](figures/fig25_bending.gif)

Watch the grid. It's no longer rotated as a rigid whole like in Section 5; it's **creased and pulled** at many places. By the end the two hooks are pulled apart so far that the final layer, `head`, just a single neuron drawing a single straight line, can separate them.

**The network didn't learn a curved boundary. It learned to bend space until a straight boundary was enough.** The last layer is still the humble neuron from Section 2, and every layer before it exists to make that neuron's job easy.

(To keep the animation honest: the final picture is standardised along its two main directions so it fits on screen. It's a change of ruler, like Section 4, and not a change of shape.)

---

### 📓 Notebook margin: the equation so far

$$
\mathbf{h}_1 = \text{ReLU}(W_1\mathbf{x} + \mathbf{b}_1)
\qquad
\mathbf{h}_2 = \text{ReLU}(W_2\mathbf{h}_1 + \mathbf{b}_2)
\qquad
\hat{y} = \sigma(\mathbf{w}\cdot\mathbf{h}_2 + b)
$$

| idea | what we now know |
|---|---|
| stacking linear layers | collapses to one layer: same line, same accuracy (89.5% = 89.5%) |
| activation function | the **fold**, the one thing a matrix can't do |
| ReLU vs sigmoid | ReLU keeps a slope of 1 for positive inputs; sigmoid saturates |
| ReLU networks | draw with straight pieces; more neurons, more pieces |
| a deep network | bends space until the **last neuron's straight line** is enough |

---

### What comes next

That last animation is the central picture of deep learning, and it deserves more than one section. Two moons is flat: two dimensions, gently hooked. Real data is not. Handwritten sevens and ones live in **784 dimensions**, crumpled into each other like two sheets of paper balled up together.

**Section 7: Uncrumpling Paper** takes the bending idea into high dimensions. We'll see why *depth* (many folds in sequence) beats *width* (many folds side by side), and what each layer of a real MNIST network does to the digits.

---

*References: François Chollet, *Deep Learning with Python*, 3rd ed., ch. 2 (why activation functions are needed: chaining affine transforms yields an affine transform; ReLU; the geometric interpretation of deep learning). Michael Nielsen, *Neural Networks and Deep Learning*, ch. 1 (sigmoid neurons) and ch. 4 (a visual view of how networks build functions). All code in this series is PyTorch.*
