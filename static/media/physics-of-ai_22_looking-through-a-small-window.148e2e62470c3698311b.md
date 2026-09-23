# Physics of AI — Part IV · Seeing

## 22. Looking Through a Small Window

> *Every network so far has started the same way: take a 28 × 28 image and flatten it into 784 numbers (Section 3.5). We noted back then that flattening throws something away: which pixels are **next to** which. This section proves that our networks really are blind to it, then fixes it with the most important idea in computer vision, the **convolution**: one small set of weights, slid across the whole image. It's the Section 2 neuron again, looking through a small window and reused everywhere.*

---

### 22.1 The blind spot, proved

Here's a strange experiment. Pick one random shuffle of the 784 pixel positions and apply that **same** shuffle to every image, in training and in validation. The digits become unrecognisable noise to a human:

![A digit, its shuffled version, and how each network copes](figures/fig100_shuffle.png)

Now train our usual dense network (an MLP, 784 → 256 → 10) on the shuffled images:

| dense network (MLP), 3 epochs | validation accuracy |
|---|---|
| original pixels | 96.29% |
| **shuffled pixels** | **96.33%** |

The same score, within noise. The network can't tell the difference, and there's a simple reason. A dense layer gives every pixel its own weight, and it never uses the fact that pixel 300 sits next to pixel 301. Shuffling the pixels just shuffles which weight goes with which pixel, and training finds the matching weights just as easily.

That's a huge waste. The most important fact about an image is that **nearby pixels belong together**: strokes, edges, corners and loops are all local patterns. A network that ignores this has to learn them from scratch, separately, at every position.

---

### 22.2 The idea: a small window, slid everywhere

Instead of one neuron looking at all 784 pixels at once, take a neuron that looks at only a **3 × 3 patch**, just 9 pixels. Slide that same neuron across the image, one step at a time, and record its output at every position. Those outputs form a new image, called a **feature map**.

![Sliding a 3×3 window across a digit](figures/fig97_sliding.gif)

At each position the neuron does exactly what it did in Section 2: multiply each of the 9 pixels by its weight, add them up, add a bias. The 9 weights are called the **kernel** (or **filter**). Written out in plain loops:

```python
def conv2d_loops(img, k):                          # img: (H, W), k: (3, 3)
    kh, kw = k.shape
    H, W = img.shape
    out = torch.zeros(H - kh + 1, W - kw + 1)
    for i in range(H - kh + 1):
        for j in range(W - kw + 1):
            out[i, j] = (img[i:i+kh, j:j+kw] * k).sum()   # the window · the kernel
    return out
```

On a 28 × 28 digit this gives a 26 × 26 map (a 3-wide window fits in 26 positions across). It matches PyTorch's `F.conv2d` to $5\times10^{-7}$.

(Small print: strictly, this is a *cross-correlation*. The mathematical "convolution" flips the kernel first. Deep learning libraries don't flip, and since the kernel is learned, it makes no difference.)

---

### 22.3 What a kernel can see

The 9 numbers in a kernel decide what pattern it responds to. Some hand-made examples on a real digit:

![Four hand-made kernels and their output maps](figures/fig98_hand_filters.png)

- **Vertical edges** `[[-1,0,1],[-2,0,2],[-1,0,1]]` compare the right column of the window with the left. The output is large where brightness changes left-to-right, which lights up both sides of each vertical stroke, with opposite signs.
- **Horizontal edges** do the same top-to-bottom, so the crossbar of the 4 lights up and the vertical strokes mostly don't.
- **A diagonal kernel** picks out slanted strokes.
- **A blur** (all weights 1/9) averages the window, which smooths the image.

Each kernel is a tiny **pattern detector**, and its output map says *where* in the image that pattern appears.

---

### 22.4 Two superpowers: sharing and sliding

**Superpower 1: weight sharing.** The same 9 weights are used at every position. Compare the cost of producing a 26 × 26 output map:

| layer producing a 26 × 26 map | weights |
|---|---|
| dense (784 inputs → 676 outputs) | **530,660** |
| one 3 × 3 convolution | **10** (9 + 1 bias) |

That's 53,000 times fewer, and it's not a cheap approximation. It encodes a true fact about images: **a stroke detector that works in the top-left should also work in the bottom-right.** A dense layer must learn that separately for every location, and a convolution learns it once.

**Superpower 2: shift in, shift out.** Move the digit 4 pixels to the right and the convolution's output map moves exactly 4 pixels to the right, with the same pattern:

![Shifting the digit shifts the feature map](figures/fig99_shift.png)

```python
# feature map of the shifted digit  vs  shifted feature map of the original digit
conv:  max difference = 0.0                        # exactly the same, just moved
dense: features change by 132% of their size       # a completely different pattern
```

This is called **translation equivariance**: move the input and the features move with it, unchanged. A dense layer has no such property. To a dense layer, a digit moved 4 pixels over is a whole new set of numbers. (Remember Section 14.5, where we had to *augment* with shifted digits to teach the dense network this. A convolution has it built in.)

> 📓 **Notebook rule:** *a convolution is one neuron with a small window, reused at every position.* Look locally, share the weights, and let the output move with the input.

---

### 22.5 A first convolutional network

Stack two convolution layers and put a dense layer at the end to make the decision:

```python
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1  = nn.Conv2d(1, 16, 3)           # 16 different 3×3 kernels on the 1-channel image
        self.c2  = nn.Conv2d(16, 32, 3)          # 32 kernels, each looking at all 16 maps below
        self.out = nn.Linear(32 * 24 * 24, 10)   # 28 → 26 → 24 after two 3×3 convs
    def forward(self, x):
        h = F.relu(self.c1(x[:, None]))          # (N, 16, 26, 26)
        h = F.relu(self.c2(h))                   # (N, 32, 24, 24)
        return self.out(h.flatten(1))
```

Two new ideas appear here:

- **Many kernels per layer.** `Conv2d(1, 16, 3)` learns **16** different 3 × 3 kernels, so it produces 16 feature maps, one per pattern. These are the **channels**, the image version of "several neurons in a layer" (Section 5.1).
- **Kernels that look through all channels.** In the second layer, each kernel is 16 × 3 × 3: it looks at a 3 × 3 patch of *all 16 maps* at once, so it can combine patterns: "a vertical edge here *and* a horizontal edge there", which starts to look like a corner.

Same 3 epochs, same optimiser as the dense network:

| 3 epochs | weights | validation accuracy |
|---|---|---|
| dense network (MLP) | 203,530 | 96.29% |
| **small CNN** | 189,130 | **98.42%** |

**Two points better with fewer weights**, and most of the CNN's weights (184,330 of them) are in that final dense layer, not in the convolutions. The two conv layers together have just 4,800.

Now the shuffle test from 22.1, on the CNN:

| small CNN | validation accuracy |
|---|---|
| original pixels | **98.42%** |
| shuffled pixels | 96.44% |

The CNN **loses its advantage** when neighbours are scrambled. Its 3 × 3 windows now look at pixels that have nothing to do with each other. (It doesn't collapse completely, because its large final dense layer can still learn to read the scrambled maps like an MLP would, and 96.44% is almost exactly the MLP's score.) This is the proof: the CNN's extra two points come **entirely** from using neighbourhoods.

---

### 22.6 What did it learn to look for?

Here are the 16 kernels the first layer learned by itself, and what each lights up on a digit:

![16 learned 3×3 kernels and their feature maps](figures/fig101_learned_filters.png)

No one told it about edges, yet many of the learned kernels are edge detectors at different angles (bright on one side, dark on the other), and a few are close to blurs that respond to "ink here". Their maps split the digit into pieces: one lights up the left edges of strokes, another the right edges, another the horizontal crossbar. One kernel's map is completely dark on this digit, a detector for something this 4 simply doesn't contain.

This is the first rung of the ladder promised in Section 7.5: **the first layer finds edges.** In the next section, the layers above start combining edges into bigger shapes.

---

### 📓 Notebook margin: the equation so far

$$
\text{feature}[c_{\text{out}}, i, j] \;=\; b_{c_{\text{out}}} + \sum_{c_{\text{in}}}\;\sum_{u=0}^{2}\sum_{v=0}^{2} K[c_{\text{out}}, c_{\text{in}}, u, v]\cdot \text{input}[c_{\text{in}},\, i+u,\, j+v]
$$

| idea | what we now know |
|---|---|
| the blind spot | a dense network scores the same on shuffled pixels (96.29% vs 96.33%) |
| convolution | one small kernel slid everywhere; 26 × 26 map from a 28 × 28 image |
| weight sharing | 10 weights instead of 530,660 for a 26 × 26 map |
| equivariance | shift the digit 4 px → the map shifts 4 px, exactly |
| channels | many kernels per layer; each kernel looks across all input channels |
| small CNN | 98.42% vs 96.29% (dense), with fewer weights; shuffling removes the gain |
| learned kernels | mostly edge detectors, discovered on their own |

---

### What comes next

Our CNN still has two problems. It ends with a huge dense layer (97% of its weights), and each unit in the second layer sees only a **5 × 5** patch of the original image. That's not enough to see a whole digit, let alone a face.

**Section 23: Zooming Out** introduces **pooling**, which shrinks the feature maps so that each deeper layer sees a wider area of the image. We'll track how the "receptive field" grows layer by layer, replace the giant dense layer with a tiny one, build a proper CNN in the classic LeNet style, and see how close it gets to the 99.79% record Nielsen quoted back in Part I.

---

*References: Michael Nielsen, *Neural Networks and Deep Learning*, ch. 6 ("Deep learning": local receptive fields, shared weights and biases, feature maps). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 8 ("Image classification": the convolution operation, translation invariance, learning spatial hierarchies of patterns). Yann LeCun et al. (1998), "Gradient-based learning applied to document recognition" (LeNet). All code in this series is PyTorch.*
