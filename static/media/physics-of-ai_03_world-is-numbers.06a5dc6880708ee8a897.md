# Physics of AI — Part I · The Gears

## 3. The World Is Numbers

> *Section 2 ended with one neuron, $\sigma(\mathbf{w}\cdot\mathbf{x} + b)$, and a question we skipped. The neuron multiplies $\mathbf{x}$ by weights, but hunger isn't a number, a bad PG dinner isn't a number, and a handwritten "7" certainly isn't. So before a neuron can think about anything, the world has to become numbers. This section is about how that happens.*

---

### 3.1 What does a neuron actually see?

You look at the image below and see a **7**. The neuron sees the grid on the right.

![A 7 as the neuron sees it](figures/fig7_digit_as_numbers.png)

That's the whole trick. A handwritten digit is a 28 × 28 grid of brightness values: **0** where the paper is blank, **255** where the ink is darkest, and everything in between on the soft edges of the stroke.

No shapes, no strokes, no "sevenness", just 784 numbers. Whatever the network ends up knowing about sevens, it has to work it out from those numbers.

This holds far beyond digits. Sound is a list of air-pressure samples, a sentence is a list of token IDs, and a video is a stack of pixel grids. **Every kind of data a neural network touches is first turned into numbers arranged in a regular shape.** That arranged block of numbers has a name: a **tensor**.

---

### 3.2 The rank ladder

A tensor is a container of numbers, and the main thing to know about it is **how many directions it extends in**. That count is its **rank** (also called `ndim`, the number of **axes**).

We'll climb the ladder using things we've already built.

![The rank ladder: scalar to video](figures/fig6_rank_ladder.png)

**Rank 0, scalar.** One number, with no direction to extend in.

```python
import torch

hunger = torch.tensor(7.)
hunger.ndim, hunger.shape        # → 0, torch.Size([])
```

**Rank 1, vector.** A row of numbers along one axis. One night from Section 2 is a vector: `[hungry, pg_bad, month_end]`.

```python
night = torch.tensor([1., 1., 0.])
night.ndim, night.shape          # → 1, torch.Size([3])
```

**Rank 2, matrix.** A stack of vectors, so two axes: rows × columns. We already built one. The `nights` tensor from Section 2 was all eight possible nights stacked together:

```python
nights.ndim, nights.shape        # → 2, torch.Size([8, 3])
#                                        8 nights × 3 facts each
```

**Rank 3.** A stack of matrices, for example a pile of images. This is exactly MNIST:

```python
from torchvision import datasets

mnist  = datasets.MNIST("data", train=True, download=True)
images = mnist.data              # the raw pixel tensor
labels = mnist.targets

images.ndim, images.shape        # → 3, torch.Size([60000, 28, 28])
```

Read the shape out loud: *sixty thousand images, each 28 rows by 28 columns.*

**Rank 4 and 5.** A colour image needs three grids (red, green, blue), so a batch of colour images is rank 4. A video is a sequence of colour images, so a batch of videos is rank 5. The idea doesn't change as you go up; you just add another axis each time.

> 📓 **Notebook rule:** *every new axis answers a new question.* "Which image?" "Which row?" "Which column?" "Which colour?" "Which frame?" If you can name the question each axis answers, you understand the tensor.

---

### 3.3 Three facts about any tensor

Every tensor you'll ever meet is described completely by three things:

| attribute | what it tells you | MNIST |
|---|---|---|
| `ndim` | how many axes | `3` |
| `shape` | how long each axis is | `(60000, 28, 28)` |
| `dtype` | what kind of number is stored | `torch.uint8` |

The `dtype` deserves a closer look:

```python
images.dtype                     # → torch.uint8
images.min(), images.max()       # → 0, 255
```

`uint8` means *unsigned 8-bit integer*: whole numbers from 0 to 255, one byte each. It's compact and perfect for storage, but it's **not** what we feed a neuron. Neurons need to multiply, add, and later take gradients, which means real-valued numbers. So the first thing we do with MNIST is:

```python
x = images.float() / 255
x.dtype, x.min(), x.max()        # → torch.float32, 0.0, 1.0
```

Two changes happen in that one line:

1. **`uint8 → float32`:** the neuron can now work with fractions and gradients.
2. **`÷ 255`:** every pixel now sits between 0 and 1.

The first change is about *format*. The second is about *scale*, and that turns out to matter much more than it looks. Keep it in mind for Section 4.

---

### 3.4 Slicing, and the axis that means "which example"

Because a tensor is regular, we can cut pieces out of it with plain indexing, one slice per axis, separated by commas.

```python
images[10:100].shape             # → (90, 28, 28)   images 10 … 99
images[:, 7:-7, 7:-7].shape      # → (60000, 14, 14) centre of every image
images[:, 14:, 14:].shape        # → (60000, 14, 14) bottom-right of every image
```

`:` means "take everything along this axis", and negative indices count from the end, so `7:-7` drops 7 pixels from each edge.

![Slicing a digit](figures/fig8_slicing.png)

Notice what we did **not** slice in the second and third lines: the first axis. We took every image and cropped each one the same way. That first axis is special.

**Axis 0 is the samples axis.** In almost every tensor in deep learning, the first axis answers *"which example?"*. For `nights` it was *which night*, and for `images` it's *which digit*. The other axes describe the insides of one example.

This matters because a network never looks at all 60,000 digits at once. It looks at a small **batch**, a slice along axis 0:

```python
batch_size = 128
n = 3
batch = images[batch_size * n : batch_size * (n + 1)]
batch.shape                      # → (128, 28, 28)
```

![A batch of digits](figures/fig9_batch.png)

Remember `nights @ w` in Section 2, where all eight decisions came out of one multiplication? A batch works the same way: 128 digits go through the network in one matrix operation. The "why" of batches (and why they're *random*) comes when we reach gradient descent. For now, just notice that the batch lives on axis 0.

---

### 3.5 Turning the grid into a vector the neuron can read

Our neuron from Section 2 takes a **vector** $\mathbf{x}$ and dots it with a weight vector $\mathbf{w}$. A digit is a **28 × 28 matrix**. To make them fit, we lay the 28 rows end to end:

```python
flat = images.reshape(len(images), 784)     # or .reshape(len(images), -1)
flat.shape                                   # → (60000, 784)
```

Reshaping never changes the numbers, only how they're arranged. 28 × 28 = 784, so nothing is lost and nothing is added.

A neuron that looks at a digit now needs **784 weights**, one opinion per pixel:

```python
import torch.nn as nn
pixel_neuron = nn.Linear(784, 1)
pixel_neuron.weight.shape        # → (1, 784)
```

Same equation, $\sigma(\mathbf{w}\cdot\mathbf{x} + b)$, but $\mathbf{x}$ now has 784 entries. Geometrically, each digit is **one point in 784-dimensional space**, and the neuron's decision boundary from Section 2.3 is a flat cut through that space. We can't picture 784 dimensions, but the maths doesn't care.

(Flattening throws away the fact that neighbouring pixels are *neighbours*. That will bother us later, and fixing it is exactly what ConvNets do in Part IV.)

---

### 3.6 Everyday data as shapes

Once you get used to thinking in shapes, most datasets fall into a few families:

| data | shape | axes answer… | example |
|---|---|---|---|
| **Table** | `(samples, features)` | which row, which column | our `nights`: `(8, 3)` |
| **Time series** | `(samples, timesteps, features)` | which series, which moment, which reading | 30 days of Outer Ring Road traffic, every minute, 3 readings (speed, vehicle count, rain): `(30, 1440, 3)` |
| **Images** | `(samples, channels, height, width)` | which image, which colour, which row, which column | 128 photos: `(128, 3, 256, 256)` |
| **Video** | `(samples, frames, channels, height, width)` | … + which frame | a batch of reels |
| **Text** | `(samples, tokens)` | which sentence, which word position | the full story in Part V |

**A PyTorch detail about images.** PyTorch puts the colour axis *before* height and width: `(N, C, H, W)`, "channels-first". Keras and many image libraries use `(N, H, W, C)`, "channels-last". Images loaded from disk often arrive channels-last, so you'll write this line a lot:

```python
batch_hwc = torch.zeros(128, 256, 256, 3)      # as loaded: N, H, W, C
batch_chw = batch_hwc.permute(0, 3, 1, 2)      # for PyTorch: N, C, H, W
batch_chw.shape                                # → (128, 3, 256, 256)
```

**Why shapes decide what's affordable.** One minute of 1080p Instagram reel at 30 fps, stored as raw `float32`, would be:

$$
\underbrace{1800}_{\text{frames}} \times \underbrace{3}_{\text{colours}} \times 1920 \times 1080 \times \underbrace{4 \text{ bytes}}_{\texttt{float32}} \approx \mathbf{44.8\ GB}
$$

That's for *one* minute of *one* video. This is why we work in batches, why real video is stored compressed, and why the `dtype` from 3.3 matters for more than tidiness.

---

### 3.7 How you encode matters: a quiet trap

So far our numbers were **measurements**: brightness, speed, hunger level. Their size means something, since 200 is brighter than 100.

Now suppose we add a new fact to our dinner neuron: *which area of Bengaluru you're in tonight.* Koramangala, Indiranagar or Yelahanka. The quick option is to number them:

```python
area = {"Koramangala": 1, "Indiranagar": 2, "Yelahanka": 3}
```

It looks harmless, but look at what we just told the model:

![Label encoding invents distances; one-hot doesn't](figures/fig10_encoding.png)

- **An order:** Yelahanka > Indiranagar > Koramangala. Greater in *what*? We invented that.
- **Distances:** Koramangala is "1 away" from Indiranagar but "2 away" from Yelahanka.

```python
codes = torch.tensor([[1.], [2.], [3.]])
torch.cdist(codes, codes)
# [[0, 1, 2],
#  [1, 0, 1],
#  [2, 1, 0]]      ← distances that mean nothing
```

A neuron makes this worse. It has **one weight** for the area input, so whatever effect it assigns to Koramangala, it's forced to give Yelahanka **three times** that effect. The encoding has decided the model's opinion before any learning happens.

**The fix is one-hot encoding.** Give every area its own axis:

```python
import torch.nn.functional as F
one_hot = F.one_hot(torch.tensor([0, 1, 2]), num_classes=3).float()
# Koramangala → [1, 0, 0]
# Indiranagar → [0, 1, 0]
# Yelahanka   → [0, 0, 1]

torch.cdist(one_hot, one_hot)
# [[0.00, 1.41, 1.41],
#  [1.41, 0.00, 1.41],
#  [1.41, 1.41, 0.00]]   ← every area equally different (√2)
```

Now each area gets **its own weight**, and the neuron is free to learn that Koramangala pushes you to order while Yelahanka doesn't, with no fake ordering baked in.

> 📓 **Notebook rule:** *the numbers you choose are a claim about the world.* If two things aren't "more" or "less" than each other, don't encode them as bigger and smaller numbers.

This is the first time we've justified a piece of **feature engineering** from the maths, not from a checklist. We'll keep doing that.

---

### 📓 Notebook margin: the equation so far

$$
\hat{y} \;=\; \sigma(\,\mathbf{w}\cdot\mathbf{x} + b\,),
\qquad
\mathbf{x} \in \mathbb{R}^{784}
$$

| idea | what we now know |
|---|---|
| $\mathbf{x}$ | not "the facts" loosely but a **tensor**, with a rank, a shape and a dtype |
| a digit | a point in 784-D space, pixels scaled to $[0, 1]$ |
| a batch | a slice along **axis 0**, pushed through in one matrix multiply |
| encoding | a *claim* about the world: measurements stay numbers, categories go one-hot |

---

### What we skipped, and what comes next

In 3.3 we divided every pixel by 255 and moved on quickly. Here's why that deserves a whole section.

Go back to the dinner neuron and add one more honest input, **monthly income in ₹**, next to **hunger level**:

| input | range |
|---|---|
| hunger level | 0 – 10 |
| monthly income | 0 – 1,00,000 |

Both are real measurements, and both are encoded "correctly". Yet one of them is ten thousand times bigger than the other. What does that do to a neuron's line, or to any model that measures distance between points?

That's **Section 4: Why Scale Matters**, where we'll watch a simple KNN classifier fall apart on raw numbers and recover after one line of normalisation.

---

*References: François Chollet, *Deep Learning with Python*, 3rd ed., ch. 2 ("Data representations for neural networks": tensors, rank, shape, dtype, slicing, batches, real-world data tensors). Michael Nielsen, *Neural Networks and Deep Learning*, ch. 1 (the 784-input network for MNIST). All code in this series is PyTorch.*
