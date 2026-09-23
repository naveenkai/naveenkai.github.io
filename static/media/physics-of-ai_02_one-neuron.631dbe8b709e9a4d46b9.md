# Physics of AI — Part I · The Gears

## 2. One Neuron

> *In the prologue we got stuck on a simple question: you can read a scribbled "7" instantly, but you can't write down the rules for how you do it. So we stop trying to write the rules. We build something small that can **learn** them, and we start with the smallest piece there is.*

---

### 2.1 A decision you make every night

It's 8:30 pm at the PG and you're deciding whether to order dinner. You don't think about it much, but your brain is weighing a few facts:

- **Am I hungry?** Yes or no.
- **Is the PG dinner bad today?** Yes or no.
- **Is it month-end?** If so, the wallet is thin.

Some facts count more than others. Hunger pushes hard toward ordering and bad PG food pushes a little. Month-end pushes **against** ordering. Once the total push crosses some line, you open the app.

We can write that down as numbers:

| fact | input $x_j$ | importance $w_j$ |
|---|---|---|
| hungry? | $x_1 \in \{0,1\}$ | $w_1 = +6$ |
| PG dinner bad? | $x_2 \in \{0,1\}$ | $w_2 = +4$ |
| month-end? | $x_3 \in \{0,1\}$ | $w_3 = -5$ |

Then pick a **threshold**, say 5: you order only if the weighted evidence beats it.

$$
\text{order} =
\begin{cases}
1 & \text{if } \; w_1x_1 + w_2x_2 + w_3x_3 > \text{threshold} \\
0 & \text{otherwise}
\end{cases}
$$

That's the whole machine. Frank Rosenblatt called it a **perceptron** in the late 1950s. It's the first gear in everything we build in this series.

![A single neuron deciding whether to order dinner](figures/fig1_neuron.png)

Two things to notice before we go on:

1. **Weights are opinions.** A big positive weight means "this matters a lot, and it pushes toward yes". A negative weight means "this pushes toward no". The model's entire personality lives in these numbers.
2. **Change the weights and you get a different person.** Someone with $w_3 = -1$ doesn't care much about month-end. Same machine, different numbers, different behaviour.

Keep that second point in mind, because learning will turn out to be nothing more than finding the right numbers.

---

### 2.2 Moving the threshold inside: the bias

A threshold on one side and a sum on the other is untidy, so we move the threshold over:

$$
w_1x_1 + w_2x_2 + w_3x_3 - \text{threshold} > 0
$$

Then we give $-\text{threshold}$ its own name, the **bias** $b$:

$$
\boxed{\;\text{order} = 1 \;\text{ if }\; \mathbf{w}\cdot\mathbf{x} + b > 0\;}
\qquad \mathbf{w}\cdot\mathbf{x} = \sum_j w_j x_j
$$

The notebook way to read $b$ is as **eagerness**, meaning how easily this neuron says yes:

- **Big positive $b$:** an eager neuron that fires on almost no evidence (the friend who's *always* up for ordering).
- **Big negative $b$:** a stubborn neuron that needs overwhelming evidence (the friend who's saving for a bike).

In our example $b = -5$. Here is the neuron in PyTorch, checked against every possible night:

```python
import torch

w = torch.tensor([6., 4., -5.])   # hungry, PG-bad, month-end
b = torch.tensor(-5.)

# all 8 possible nights, one per row
nights = torch.tensor([[h, p, m] for h in (0., 1.)
                                  for p in (0., 1.)
                                  for m in (0., 1.)])

z = nights @ w + b          # weighted evidence for every night at once
order = (z > 0).int()
```

```
[hungry, pg_bad, month_end]   z      decision
[0, 0, 0]                    -5      no
[0, 0, 1]                   -10      no
[0, 1, 0]                    -1      no
[0, 1, 1]                    -6      no
[1, 0, 0]                    +1      ORDER
[1, 0, 1]                    -4      no
[1, 1, 0]                    +5      ORDER
[1, 1, 1]                    +0      no      ← on the fence
```

Look at the last row. You're hungry and the food is bad, but it's month-end, and the evidence comes out to **exactly zero**. The neuron is sitting right on the fence. We'll come back to this night, because it breaks the perceptron.

Also look at `nights @ w`. We didn't loop over the nights. We stacked them into a matrix and ran all eight decisions in **one matrix multiplication**. That habit will matter a lot from here on.

---

### 2.3 What a neuron looks like: a line

So far this is arithmetic. Now let's draw it.

Swap the yes/no inputs for continuous ones: **hunger level** $x_1$ from 0 to 10 and **how bad the PG dinner is** $x_2$ from 0 to 10. Every past night becomes a point on a 2D plane, blue if you ordered and orange if you didn't.

The neuron's decision flips exactly where

$$
w_1x_1 + w_2x_2 + b = 0
$$

and in 2D that equation is **a straight line**. The neuron splits the plane in two: "order" on one side and "don't" on the other.

![The decision boundary rotates with w and slides with b](figures/fig2_decision_boundary.png)

This picture explains what the two parameters actually do:

- **$\mathbf{w}$ sets the direction.** The green arrow is the weight vector. It always points perpendicular to the line, toward the "yes" side. Change $\mathbf{w}$ and the line **rotates**.
- **$b$ sets the position.** Change $b$ and the line **slides** without turning. A more negative $b$ pushes it away, so you need more hunger or worse food before you order.

With three inputs the line becomes a plane, and with 784 inputs (the pixels of one MNIST digit) it becomes a *hyperplane*. We can't draw that, but the idea is identical: **one neuron is one flat cut through space.** Hold on to this, because Part I is really a story about what happens when you stack and bend these cuts.

---

### 2.4 The problem: the perceptron is a cliff

We want the machine to **learn** its weights and not rely on us hand-picking 6, 4 and −5. Learning, at its simplest, goes like this:

> Nudge a weight a little → watch how the output changes → keep the nudges that help.

Now go back to the on-the-fence night $[1, 1, 1]$, where $z = 0$ and the neuron says no. Nudge the hunger weight from **6.00 to 6.01**:

```python
def perceptron(x, w, b):
    return int(torch.dot(w, x) + b > 0)

x = torch.tensor([1., 1., 1.])
perceptron(x, torch.tensor([6.00, 4., -5.]), b)   # → 0
perceptron(x, torch.tensor([6.01, 4., -5.]), b)   # → 1
```

A change of **0.01** flipped the answer completely. On every other night the same nudge changed **nothing at all**.

That's the problem. The step function is flat everywhere and then jumps off a cliff at zero:

- On the flat parts, a small nudge does nothing, so there's no signal about which direction helps.
- At the cliff edge, a small nudge flips everything, so the signal is wildly out of proportion.

You can't steer something whose response to a small push is either *nothing* or *everything*. With many perceptrons wired together, one flip cascades through the rest and the whole network lurches. **To learn, we need small changes in weights to cause small changes in output.**

---

### 2.5 The fix: bend the cliff into a slope

So we keep the neuron exactly as it is, $z = \mathbf{w}\cdot\mathbf{x} + b$, and swap the hard step for a smooth S-curve, the **sigmoid**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
\qquad\Rightarrow\qquad
\text{output} = \sigma(\mathbf{w}\cdot\mathbf{x} + b)
$$

![Step is a cliff, sigmoid is a slope](figures/fig3_step_vs_sigmoid.png)

Read it in three zones:

- **$z$ very positive:** $e^{-z} \to 0$, so $\sigma \to 1$. The neuron is confidently yes, just like the perceptron.
- **$z$ very negative:** $e^{-z} \to \infty$, so $\sigma \to 0$. It's confidently no, again like the perceptron.
- **$z$ near zero:** here it's different. The output slides smoothly through 0.5, so the neuron is saying "*I'm not sure*".

Try the fence night again:

```python
def neuron(x, w, b):
    return torch.sigmoid(torch.dot(w, x) + b)

neuron(x, torch.tensor([6.00, 4., -5.]), b)   # → 0.5000
neuron(x, torch.tensor([6.01, 4., -5.]), b)   # → 0.5025
```

A small push now gives a small, **proportional** response, which is exactly what learning needs.

The sigmoid isn't a new idea either. It's the perceptron with the edges sanded off. Scale the weights up and the S-curve sharpens until it *becomes* the step:

![Scaling z sharpens the sigmoid into the step](figures/fig4_sharpness.png)

Everything the perceptron could do, a sigmoid neuron can do too, with a slope as a bonus.

In the 2D picture, the hard line becomes a **soft band**. Far from the line the neuron is sure, and near it the neuron hedges:

![Hard vs soft decision boundary](figures/fig5_hard_vs_soft.png)

---

### 2.6 Why the slope is everything (a preview)

"Small change in → small change out" has a precise form. Calculus says that near the current weights:

$$
\Delta\,\text{output} \;\approx\; \sum_j \frac{\partial\,\text{output}}{\partial w_j}\,\Delta w_j \;+\; \frac{\partial\,\text{output}}{\partial b}\,\Delta b
$$

Each $\partial / \partial w_j$ is a **sensitivity**: how much the output moves per unit nudge of that weight. Once you know the sensitivities, you know which way to turn every knob.

- For the **step**, every sensitivity is 0 (flat) or undefined (the cliff), so the formula tells you nothing.
- For the **sigmoid**, every sensitivity is a real, finite number.

PyTorch can compute these sensitivities for us. That's what `autograd` does:

```python
w = torch.tensor([6., 4., -5.], requires_grad=True)
b = torch.tensor(-5.,          requires_grad=True)

y = torch.sigmoid(torch.dot(w, x) + b)
y.backward()                 # "how sensitive is y to each knob?"

w.grad   # → tensor([0.25, 0.25, 0.25])
b.grad   # → tensor(0.25)
```

Try the same thing with a hard step (`torch.heaviside`) and PyTorch refuses outright:

```
RuntimeError: derivative for aten::heaviside is not implemented
```

That error sums up this whole section: **no slope, no learning.**

(The 0.25 isn't random. At $z = 0$ the sigmoid is at its steepest, and its slope there is exactly $\sigma(0)\,(1-\sigma(0)) = 0.5 \times 0.5$. We'll derive that properly when we reach gradient descent.)

---

### 2.7 The same neuron, the way PyTorch writes it

Everything above, $\mathbf{w}\cdot\mathbf{x} + b$, is packaged in PyTorch as a single layer, `nn.Linear`. It's the same thing Keras calls a `Dense` layer.

```python
import torch.nn as nn

layer = nn.Linear(in_features=3, out_features=1)   # holds w (1×3) and b (1)
with torch.no_grad():                               # load our hand-picked opinions
    layer.weight.copy_(torch.tensor([[6., 4., -5.]]))
    layer.bias.fill_(-5.)

torch.sigmoid(layer(nights)).squeeze()
```

```
[0.007, 0.000, 0.269, 0.002, 0.731, 0.018, 0.993, 0.500]
```

Compare this with the perceptron table in 2.2. The confident nights stay near 0 or 1, the borderline ones (0.269, 0.731) show some doubt, and the fence night comes out at **0.500**, "I really can't tell". The perceptron threw that information away.

For the rest of the series we won't hand-pick weights. `nn.Linear` starts with random ones, and our job becomes getting from random to right. That's learning.

---

### 📓 Notebook margin: the equation so far

$$
\hat{y} \;=\; \sigma(\,\mathbf{w}\cdot\mathbf{x} + b\,)
$$

| symbol | notebook meaning | geometry |
|---|---|---|
| $\mathbf{x}$ | the facts about the world | a point in space |
| $\mathbf{w}$ | how much each fact matters | the direction of the cut |
| $b$ | how eager the neuron is | where the cut sits |
| $\sigma$ | turns a cliff into a slope | the soft edge of the cut |

This equation will keep growing throughout the series. By the end it will look like $\text{softmax}(QK^\top/\sqrt{d})\,V$, and you'll see it's the same idea underneath.

---

### What we skipped, and what comes next

We quietly assumed something big: that *hunger* and *how bad the food is* **are numbers** at all. For a neuron everything must be a number, including the pixels of a handwritten "7", the words in a sentence and the frames of a video. How do we turn the world into numbers, and why does it matter so much *how* we do it?

That's **Section 3: The World Is Numbers**.

---

*References: Michael Nielsen, *Neural Networks and Deep Learning*, ch. 1 (perceptrons, sigmoid neurons). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 2 (tensor operations, geometric view). All code in this series is PyTorch.*
