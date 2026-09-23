# Physics of AI — Part I · The Gears

## 5. Tensor Operations as Physics

> *For three sections we worked on the input: numbers, encoded honestly, on a fair scale. Now we look at the machine itself. In Section 2 one neuron drew one line. Here we put many neurons side by side and find that what they do together is physical: they **move space**, stretching, rotating, shearing and sliding it. By the end you'll be able to look at a weight matrix and picture the motion it performs.*

---

### 5.1 From one neuron to a layer

One neuron has a weight **vector** $\mathbf{w}$ and a bias $b$, and gives one number:

$$
z = \mathbf{w}\cdot\mathbf{x} + b
$$

Four neurons looking at the same input have four weight vectors. Stack them as the **rows** of a matrix and put their biases in a vector:

$$
W = \begin{bmatrix} \text{—}\; \mathbf{w}_1 \;\text{—} \\ \text{—}\; \mathbf{w}_2 \;\text{—} \\ \text{—}\; \mathbf{w}_3 \;\text{—} \\ \text{—}\; \mathbf{w}_4 \;\text{—} \end{bmatrix}
\qquad
\mathbf{z} = W\mathbf{x} + \mathbf{b}
$$

That's a **layer**: several neurons, same input, all computed at once. In PyTorch it's the `nn.Linear` we already used, now with more than one output:

```python
import torch, torch.nn as nn

layer = nn.Linear(3, 4)          # 3 inputs → 4 neurons
layer.weight.shape               # → (4, 3)   one row per neuron
layer.bias.shape                 # → (4,)     one bias per neuron

out = layer(nights)              # our 8 nights from Section 2
out.shape                        # → (8, 4)   8 nights × 4 neuron outputs
```

Under the hood it's one line of maths:

```python
manual = nights @ layer.weight.T + layer.bias
torch.allclose(out, manual)                                  # → True

# column 2 of the output is exactly neuron 2's dot product
torch.allclose(out[:, 2], nights @ layer.weight[2] + layer.bias[2])   # → True
```

(PyTorch stores `weight` as *(outputs, inputs)* and computes `x @ W.T`, so each row stays "one neuron". Chollet's Keras stores it the other way round. The maths is the same.)

---

### 5.2 The one shape rule you need

Matrix multiplication has exactly one rule: **the inner sizes must match, and the outer sizes survive.**

$$
(a,\; \mathbf{b}) \;@\; (\mathbf{b},\; c) \;\longrightarrow\; (a,\; c)
$$

![The shape rule for matrix multiplication](figures/fig20_shape_rule.png)

Each of the 8 rows (nights) is dotted with each of the 4 columns (neurons), so we get 8 × 4 numbers. Get the order wrong and PyTorch tells you straight away:

```python
A = torch.randn(8, 3)
B = torch.randn(3, 4)
(A @ B).shape        # → (8, 4)
B @ A                # RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x4 and 8x3)
```

This is also why **order matters** in matrix multiplication: `A @ B` and `B @ A` are different things, and often only one of them is even allowed.

> 📓 **Notebook rule:** *when a shape error appears, write the shapes side by side. The inner pair is almost always the culprit.*

---

### 5.3 The other two gears: element-wise ops and broadcasting

A layer needs two more operations besides `@`.

**Element-wise operations** act on every number independently. The activation functions are all element-wise:

```python
x = torch.tensor([-2., -0.5, 0., 1.5, 3.])
torch.sigmoid(x)     # → [0.119, 0.378, 0.500, 0.818, 0.953]
torch.relu(x)        # → [0.0,   0.0,   0.0,   1.5,   3.0  ]   ← max(x, 0): meet it properly in Section 6
```

**Broadcasting** is what makes `+ b` work on a batch. The output is `(128, 4)`, one row per example, but `b` is only `(4,)`. PyTorch conceptually copies `b` down all 128 rows without ever using the memory:

```python
batch = torch.randn(128, 4)
b     = torch.randn(4)
(batch + b).shape                    # → (128, 4)

# same as explicitly repeating b 128 times
torch.allclose(batch + b, batch + b.unsqueeze(0).expand(128, 4))   # → True
```

The rule: line the shapes up **from the right**. Sizes must be equal, or one of them must be 1 (or missing). `(128, 4)` and `(4,)` line up on the 4, so `b` is stretched along the missing axis.

**Why we never write loops.** Here is 1,000 MNIST-sized inputs times a `(784, 10)` weight matrix, done with Python loops and done with one `@`:

| method | time for 1,000 rows (CPU, measured) |
|---|---|
| Python triple loop | **≈ 30 seconds** |
| `X @ W` | **≈ 0.2 milliseconds** |

That's a difference of **over 100,000×** for identical numbers. `@` hands the work to highly tuned routines (BLAS on the CPU, CUDA on a GPU) that process many numbers per instruction. The practical lesson: **think in whole tensors, not in elements.** Every time we stacked nights into a matrix instead of looping, we were already doing this.

---

### 5.4 A matrix moves space

Now for the physics. Forget neural networks for a moment and take a plain 2D world: a grid of lines and a little house (asymmetric on purpose, so you can tell when it's flipped). Multiply every point by a 2 × 2 matrix $W$ and watch what happens:

![Gallery of transformations](figures/fig15_transform_gallery.png)

Every panel is the same operation, $\mathbf{y} = W\mathbf{x}$, with different numbers in $W$:

| motion | $W$ | what happens |
|---|---|---|
| **scale** | `[[1.5, 0], [0, 0.6]]` | stretch sideways, squash vertically |
| **rotate 30°** | `[[cos, −sin], [sin, cos]]` | spin around the origin |
| **shear** | `[[1, 0.8], [0, 1]]` | slide the top sideways, like pushing a deck of cards |
| **reflect** | `[[−1, 0], [0, 1]]` | mirror; notice the chimney switches sides |

```python
import math
θ = math.pi / 2                                   # 90°
R = torch.tensor([[math.cos(θ), -math.sin(θ)],
                  [math.sin(θ),  math.cos(θ)]])
R @ torch.tensor([1., 0.])      # → [0, 1]    east turns to north
R @ torch.tensor([0., 1.])      # → [-1, 0]   north turns to west
```

Two things every one of these motions has in common:

1. **Straight lines stay straight** and parallel lines stay parallel. The grid bends as a whole, never into curves.
2. **The origin never moves.** $W \cdot \mathbf{0} = \mathbf{0}$, always.

These are *linear* transformations. Keep both properties in mind: the second is why we need $b$, and the first is why we'll need something more in Section 6.

---

### 5.5 Reading a matrix without doing any maths

How do you look at a matrix and *see* its motion? Watch what happens to the two unit arrows $\mathbf{e}_1 = [1, 0]$ and $\mathbf{e}_2 = [0, 1]$:

```python
W = torch.tensor([[2., 1.],
                  [0., 1.]])
W @ torch.tensor([1., 0.])      # → [2, 0]   = column 1 of W
W @ torch.tensor([0., 1.])      # → [1, 1]   = column 2 of W
```

![Columns of W are where the basis vectors land](figures/fig16_basis_columns.png)

> 📓 **Notebook rule:** *column $j$ of $W$ is where the $j$-th axis lands.* Every other point follows along, because any point is just "some of $\mathbf{e}_1$ plus some of $\mathbf{e}_2$".

So `[[2, 1], [0, 1]]` says: *the x-axis doubles in length, and the y-axis tips over to point diagonally.* That's a stretch plus a shear, which you can read straight off the numbers.

---

### 5.6 Adding $\mathbf{b}$: the slide

A pure matrix can't move the origin. But a neuron's decision line from Section 2 had to be able to sit **anywhere**, not just pass through zero. That's the bias's job. Adding $\mathbf{b}$ **slides** all of space without bending it:

$$
\mathbf{y} = \underbrace{W\mathbf{x}}_{\text{stretch / rotate / shear}} + \underbrace{\mathbf{b}}_{\text{slide}}
$$

Linear transform plus translation is called an **affine transformation**, and it's exactly what `nn.Linear` computes. Watch it happen in two stages:

![Affine transformation: W·x then + b](figures/fig17_affine.gif)

This connects back to Section 2.3: $\mathbf{w}$ rotated the decision line and $b$ slid it. Same roles, now for all of space at once.

---

### 5.7 One neuron, seen again: measuring along a direction

Now we can see what a single neuron *actually computes*. The dot product $\mathbf{w}\cdot\mathbf{x}$ measures **how far $\mathbf{x}$ reaches in the direction of $\mathbf{w}$**, scaled by the length of $\mathbf{w}$:

```python
w   = torch.tensor([2., 1.])
pts = torch.tensor([[1., 1.], [2., -4.], [3., 0.]])
pts @ w          # → [3, 0, 6]
```

The point `[2, −4]` scores 0. It's perpendicular to $\mathbf{w}$, so it reaches **nowhere** along $\mathbf{w}$.

![One neuron as a measurement along w](figures/fig18_projection.png)

Colour every point by its $z$ and the picture is clear: **all points on a line perpendicular to $\mathbf{w}$ get the same score.** The $z = 0$ line is exactly Section 2's decision boundary, and it was always perpendicular to $\mathbf{w}$. Now we know why.

Scale that up to a layer:

- **Each row of $W$ is one direction** the layer measures.
- A layer with 4 neurons measures every input along **4 directions** and returns those 4 measurements as the new description of the point.

**A layer re-describes each point in a new coordinate system of its own choosing.** Training is the process of choosing *which directions are worth measuring*. The layer can also change the number of dimensions:

| layer | what it does to space |
|---|---|
| `nn.Linear(3, 4)` | lifts 3-D points into a 4-D space |
| `nn.Linear(784, 10)` | squashes a 784-D digit down to 10 numbers, one per class |
| `nn.Linear(2, 2)` | same-size motion, like the gallery above |

---

### 5.8 Stacking motions

If one layer moves space, two layers should move it twice. Here's a shear followed by a scale-and-rotate:

![Composition of two transformations equals one](figures/fig19_composition.png)

Look at the third panel. **The combination of two motions is itself a single motion**, one matrix, $W = W_2 W_1$. The bias works out the same way:

$$
W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 \;=\; \underbrace{(W_2W_1)}_{W}\,\mathbf{x} \;+\; \underbrace{(W_2\mathbf{b}_1 + \mathbf{b}_2)}_{\mathbf{b}}
$$

PyTorch agrees:

```python
l1, l2 = nn.Linear(2, 5), nn.Linear(5, 3)        # two layers, with a 5-D detour in between

W = l2.weight @ l1.weight                        # (3, 2)
b = l2.weight @ l1.bias + l2.bias                # (3,)

x = torch.randn(10, 2)
torch.allclose(l2(l1(x)), x @ W.T + b, atol=1e-6)   # → True
```

Two layers, a detour through five dimensions, and the result is *exactly* one `nn.Linear(2, 3)`.

This should bother you. If stacking layers just produces another single layer, then a 100-layer network is secretly **one** layer, and "deep" learning is no deeper than the single neuron in Section 2.

---

### 📓 Notebook margin: the equation so far

$$
\mathbf{z} = W\mathbf{x} + \mathbf{b}
\qquad\text{(one layer)}
\qquad\qquad
\hat{y} = \sigma(\mathbf{z})
$$

| piece | notebook meaning | geometry |
|---|---|---|
| a **row** of $W$ | one neuron | one direction being measured |
| a **column** of $W$ | how one input feeds every neuron | where one input axis lands |
| $\mathbf{b}$ | eagerness of each neuron | slides all of space |
| `@` | many dot products at once | stretch · rotate · shear · reflect |
| broadcasting | `+ b` for a whole batch | the same slide for every point |
| stacking layers (so far) | `W₂W₁` | **still just one motion** |

---

### What comes next

We just found a problem at the heart of the whole idea. Affine motions keep lines straight, and chaining them only produces another affine motion. However many layers we stack, all we can ever do is stretch, rotate, shear and slide. We can never **bend**.

Real data, like the sevens that look a bit like ones or the nights that don't fit a single line, needs bending.

**Section 6: The Collapse** is about the small ingredient that stops the stack collapsing into one layer. You've already met it, twice: σ in Section 2, and `torch.relu` in 5.3.

---

*References: François Chollet, *Deep Learning with Python*, 3rd ed., ch. 2 ("The gears of neural networks: tensor operations": element-wise operations, broadcasting, tensor product, geometric interpretation, affine transforms). Michael Nielsen, *Neural Networks and Deep Learning*, ch. 1 (vectorised form of the layer equation). All code in this series is PyTorch.*
