# Physics of AI — Part I · The Gears

## 7. Uncrumpling Paper

> *Section 6 ended with an animation: two hooked moons, a network that bent space, and one straight line that could finally split them. That was two dimensions and two classes. Real data is 784 dimensions and ten classes, tangled far worse. This section takes the bending picture to real handwritten digits, looks inside each layer, and answers a question we've been dodging since Section 5: why **deep**, and not just **wide**?*

---

### 7.1 The picture to keep in your head

Take two sheets of paper, one blue and one orange. Lay them on top of each other and crumple them into a ball.

Now someone hands you the ball and asks: *is this point blue or orange?* In the crumpled ball, blue and orange are pressed against each other everywhere, and no single flat cut through the ball separates them.

But you know what to do. You **uncrumple** it, one careful movement at a time, until the two sheets lie flat and apart. Then a single straight cut between them is enough.

That's what a deep network does:

- **The crumpled ball is your raw data.** Sevens and ones, fours and nines, all living in 784-D pixel space and pressed against each other in complicated ways.
- **Each layer is one uncrumpling movement.** An affine step moves the paper (Section 5), and a ReLU folds it (Section 6).
- **The last layer is the straight cut**, the humble neuron from Section 2.

Every movement is simple. What makes it powerful is doing many of them in sequence. Let's check whether a real network actually behaves like this.

---

### 7.2 A real network, looked at from the inside

We train a small network on handwritten digits. To keep things fast we use **5,000 MNIST digits** (4,000 to train, 1,000 to test). The full 60,000 behaves the same, just with higher accuracy.

```python
import torch, torch.nn as nn

X = images[:5000].reshape(5000, 784).float() / 255    # Section 3 + Section 4
y = labels[:5000]
Xtr, Xte, ytr, yte = X[:4000], X[4000:], y[:4000], y[4000:]

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),     # layer 1
    nn.Linear(256, 64),  nn.ReLU(),     # layer 2
    nn.Linear(64, 16),   nn.ReLU(),     # layer 3
    nn.Linear(16, 10))                  # output: one score per digit

# trained with the same black-box loop as Section 6 (30 passes, batches of 64)
# test accuracy: 93.8%
```

Now the interesting part. We push the 1,000 test digits through the network and **stop at every layer** to look at what that layer sees:

```python
def views(x):
    out, h = {"pixels": x}, x
    for i, m in enumerate(model):
        h = m(h)
        if isinstance(m, nn.ReLU) or m is model[-1]:
            out[f"after module {i}"] = h
    return out
```

Each layer's view is a cloud of points in 256-D, 64-D, 16-D or 10-D, which we can't draw. So we draw its **shadow**: the flat 2-D view that keeps the most spread (a technique called PCA). A shadow always loses information, but it never *invents* structure, so if clusters appear in the shadow they're really there.

![1,000 test digits seen from each layer](figures/fig27_layer_views.png)

Read it left to right:

- **Pixels:** one blob. A few digits (0 on the left, 1 on the right) sit at the edges, but the middle is a crumpled mix.
- **Layer 1:** zeros pull away at the bottom and threes and fives start to gather, but it's still mostly tangled.
- **Layers 2 and 3:** distinct patches appear: nines at one edge, twos at another, threes gathering at the bottom.
- **Output:** ten colours in ten regions.

---

### 7.3 Measuring the uncrumpling

Shadows are suggestive, but let's put a number on it. For each layer we ask:

> *How far apart are the class centres, compared to how spread out each class is?*

$$
\text{separation} = \frac{\text{average distance}^2 \text{ between class centres and the overall centre}}{\text{average distance}^2 \text{ of each digit from its own class centre}}
$$

Big means tight, well-separated clusters, and small means crumpled together.

| view | dims | separation | a single straight cut (linear probe) reads the digit |
|---|---|---|---|
| pixels | 784 | **0.27** | 87.3% |
| layer 1 | 256 | 0.94 | 94.2% |
| layer 2 | 64 | 1.82 | 94.0% |
| layer 3 | 16 | 2.60 | 93.2% |
| output | 10 | **4.28** | 93.9% |

Two things to notice.

**Separation climbs at every single layer**, 16× from pixels to output. Every layer really does uncrumple a bit more, which is the central claim of this section, and it holds up.

**The linear probe** (train one fresh `nn.Linear` on each layer's view and see how well it reads the digit) **jumps after layer 1 and then levels off.** Honest reading: on this easy-ish dataset, one fold already makes the classes *mostly* separable by a flat cut. The later layers don't add much *separability*; they make the clusters **tighter and further apart**, which is what makes the final decision robust. On harder data (faces, speech, language) the later layers matter much more, and we'll see that in Parts IV–VI.

(Why is the pixel probe already 87%? 784 dimensions give a straight cut a lot of room to wiggle through. High dimensions are roomy, and some of the "crumpling" is easier to undo than our paper picture suggests. We'll come back to this when we talk about overfitting in Part II.)

---

### 7.4 Zooming in on a hard pair: 4 and 9

Some digits are genuinely crumpled into each other. A **4** with a closed top looks a lot like a **9**. Here's the same shadow trick using only fours and nines:

![4 vs 9: pixels vs layer 3](figures/fig28_four_vs_nine.png)

In pixel space the two are fully mixed, with blue dots and orange crosses interleaved everywhere. By layer 3, the fours sit on the left and the nines on the right, with a much smaller zone of overlap. Nobody told the network "look at whether the top is closed". It found a way to move space so that this difference became a **direction**.

---

### 7.5 What did the first layer learn to look at?

Section 5.7 said each neuron **measures one direction**. In pixel space a direction has 784 numbers, which is exactly an image. So we can *look* at what each first-layer neuron measures:

```python
W = model[0].weight.detach().reshape(256, 28, 28)    # one 28×28 image per neuron
```

![24 first-layer neurons as images](figures/fig29_first_layer_weights.png)

Blue pixels push that neuron's output up and red pixels push it down. You can make out **strokes and blobs**: a diagonal bar here, a curve there, a ring in one of them. They're noisy, since we trained on only 4,000 digits, but they're already stroke detectors of a sort. Nobody designed them. They came out of training.

This is the first rung of a ladder we'll climb in Part IV: early layers measure simple local patterns, later layers combine them into parts, and the last layers combine parts into "a 4" or "a 9".

---

### 7.6 Why deep beats wide: counting folds

Now the question from the start of this section. A network with 20 ReLU neurons could put them all in **one wide layer** or stack them as **ten layers of two**. Same number of neurons. Does the arrangement matter?

Go back to paper. Fold a sheet in half and you have **2** layers of paper. Fold it again, *on top of the first fold*, and you have **4**. Again, **8**. Each fold works on the result of the previous one, so the count **doubles**.

We can build exactly this with ReLU. Two neurons can fold the interval $[0, 1]$ in half, a "tent":

$$
\text{tent}(x) = 2\,\text{ReLU}(x) - 4\,\text{ReLU}(x - 0.5)
$$

```python
relu = torch.relu
def tent(x):                                  # one layer, 2 ReLU neurons
    return 2*relu(x) - 4*relu(x - 0.5)

def deep(x, layers):
    for _ in range(layers):
        x = tent(x)                           # fold the already-folded result
    return x
```

![Composing the fold: 2, 4, 8, 16 pieces](figures/fig26_folding.png)

Count the straight pieces:

| arrangement | total neurons | straight pieces |
|---|---|---|
| 1 layer × 2 | 2 | 2 |
| 2 layers × 2 | 4 | 4 |
| 4 layers × 2 | 8 | 16 |
| **10 layers × 2** | **20** | **1,024** |
| 1 wide layer × 20 | 20 | **21** (at most) |

Put 20 neurons **side by side** and each one adds one crease, so at most 21 pieces. **Stack** them and each layer folds the previous folds, so you get $2^{10} = 1{,}024$ pieces from the same 20 neurons. (I counted these by scanning the actual functions in PyTorch, not by formula.)

![Depth vs width: pieces per neuron](figures/fig30_depth_vs_width.png)

> 📓 **Notebook rule:** *width adds folds. Depth multiplies them.* That's the geometric reason "deep" learning is deep.

Two honest caveats:

- This is the **best case**, a construction where every fold is perfectly placed. A trained network doesn't use its folds this efficiently. But the ceiling on what depth *can* express grows exponentially, while width's grows only linearly.
- Depth has a price: stacking many layers makes **training** harder. The learning signal has to pass back through every fold (remember sigmoid's 0.25 slopes multiplying, in Section 6.5). Much of Part III is about paying that price.

---

### 📓 Notebook margin: the equation so far

$$
\mathbf{h}_\ell = \text{ReLU}(W_\ell\,\mathbf{h}_{\ell-1} + \mathbf{b}_\ell),
\qquad \mathbf{h}_0 = \mathbf{x}
\qquad\longrightarrow\qquad
\text{scores} = W_{\text{out}}\,\mathbf{h}_L + \mathbf{b}_{\text{out}}
$$

| idea | what we now know |
|---|---|
| raw data | a crumpled ball: classes pressed together in 784-D |
| each layer | one uncrumpling move: separation 0.27 → 0.94 → 1.82 → 2.60 → 4.28 |
| last layer | the Section 2 neuron (×10), a flat cut in an uncrumpled space |
| first-layer weights | directions in pixel space, which look like strokes |
| depth vs width | width **adds** folds (21), depth **multiplies** them (1,024) |

That completes the **forward** story of Part I. We know what a network **is**: numbers in, a stack of moves and folds, a straight cut at the end.

---

### What comes next

Every model in the last two sections was trained with a black box: `loss.backward()`, `opt.step()`. We took it on faith that the weights somehow find their way from random to right.

Now we open the box. First question: to improve, the network has to know **how wrong it is**, and "wrong" needs to be a single number we can push down.

That's **Section 8: Measuring Wrongness**, on loss functions: why squared error seemed natural, why it's the wrong choice for classification, and where cross-entropy comes from.

---

*References: François Chollet, *Deep Learning with Python*, 3rd ed., ch. 2 (the crumpled-paper view of deep learning: "uncrumpling a complicated manifold of data"). Michael Nielsen, *Neural Networks and Deep Learning*, ch. 1 ("Toward deep learning": networks as hierarchies of simpler questions). The exponential-folds argument follows Montúfar et al. (2014), "On the Number of Linear Regions of Deep Neural Networks", and Telgarsky (2016), "Benefits of depth in neural networks". All code in this series is PyTorch.*
