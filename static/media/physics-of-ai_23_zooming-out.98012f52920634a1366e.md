# Physics of AI — Part IV · Seeing

## 23. Zooming Out

> *Section 22's convolutions see the world through a 3 × 3 keyhole. Stacking two of them only widened the view to 5 × 5 pixels, far too small to take in a whole digit, let alone a face. And our first CNN still ended with a giant dense layer holding 97% of its weights. This section fixes both with one simple operation, **pooling**, and builds a proper convolutional network in the classic LeNet style. Then, with Part II's recipe, we open the test set and see how close plain convolutions get to the record.*

---

### 23.1 Pooling: keep the strongest, shrink the map

**Max pooling** chops a feature map into non-overlapping 2 × 2 squares and keeps only the **largest** value in each. The map shrinks to half its width and half its height:

![Max pooling: 4×4 → 2×2, and a real edge map before and after](figures/fig102_pooling.png)

```python
def maxpool2x2(fm):                                        # fm: (H, W), even sizes
    H, W = fm.shape
    return fm.reshape(H // 2, 2, W // 2, 2).amax(dim=(1, 3))   # max of each 2×2 block
```

This matches PyTorch's `F.max_pool2d` exactly. There are no weights at all, and nothing is learned. Why is throwing away three quarters of the numbers a *good* idea? Three reasons.

**1. It keeps the "whether" and blurs the "exactly where".** A feature map says "there's a vertical edge *here*". After pooling it says "there's a vertical edge *somewhere in this 2 × 2 area*". For recognising a digit, knowing a stroke exists in roughly the right region matters much more than its exact pixel. We can measure it: shift the digits a pixel or two and compare their features before and after.

![Feature similarity after small shifts, with 0, 1 and 2 pooling layers](figures/fig104_shift_similarity.png)

| features after… | similarity after a 1-px shift | after a 2-px shift |
|---|---|---|
| conv only | 0.894 | 0.749 |
| conv + 1 pool | 0.933 | 0.820 |
| conv + 2 pools | **0.964** | **0.899** |

Every pooling layer makes the features less sensitive to exactly where the digit sits. Section 22's convolution was **equivariant** (move the input, the map moves too). Pooling adds a little **invariance** (move the input, the features barely change), which is what a classifier ultimately wants.

**2. It widens the view of every layer above it.** A 3 × 3 kernel applied to a pooled map covers a 6 × 6 region of the map *before* pooling. Pooling makes each later convolution see twice as far.

**3. It saves a lot of compute.** Each pool divides the number of positions by 4, so later layers can afford more channels.

---

### 23.2 How far can one unit see?

The **receptive field** of a unit is the patch of the original image that can affect it. Here's how it grows through the network we're about to build (measured by backprop: take the gradient of one unit with respect to the input and see which pixels get a non-zero gradient):

![Receptive field growing layer by layer](figures/fig103_receptive_field.png)

| after… | receptive field |
|---|---|
| one 3 × 3 conv | 3 × 3 |
| two convs | 5 × 5 |
| + pool | 6 × 6 |
| + conv | 10 × 10 |
| + conv | 14 × 14 |
| + pool | **16 × 16** |
| *six convs with no pooling at all* | *13 × 13* |

Before pooling, each conv adds 2 pixels to the view. After one pool, each conv adds **4**. With two pools, the last units see a 16 × 16 patch, more than half a digit, with far fewer layers than convolutions alone would need. This is the **zoom out** in the title: early layers see strokes, middle layers see parts, and the last layers see most of the digit.

(Measurement note: we used average pooling for this measurement, because with max pooling the gradient only flows to the winning pixel of each 2 × 2 and makes the field look smaller than it is.)

---

### 23.3 The ConvNet

This is the classic shape from Yann LeCun's LeNet (1998), updated with Part II's tools: **conv, conv, pool; conv, conv, pool; decide**.

![The ConvNet architecture](figures/fig105_architecture.png)

```python
def block(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False),    # padding=1 keeps 28×28 as 28×28
            nn.BatchNorm2d(cout),                              # Section 17, per channel
            nn.ReLU()]

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            *block(1, 32),  *block(32, 32), nn.MaxPool2d(2),   # 28×28 → 14×14, 32 channels
            *block(32, 64), *block(64, 64), nn.MaxPool2d(2))   # 14×14 → 7×7,   64 channels
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 10))
    def forward(self, x):
        return self.head(self.features(x[:, None]))
```

A few details worth naming:

- **`padding=1`** adds a one-pixel border of zeros so a 3 × 3 conv keeps the map the same size. The shrinking is then done deliberately, by pooling.
- **Channels grow as maps shrink**: 1 → 32 → 64 channels while 28 → 14 → 7 pixels across. It's a common pattern: less *where*, more *what*.
- **`BatchNorm2d`** is batch norm for images. It normalises each channel over the batch **and** over all positions.

Quick comparison, 3 epochs on 50,000 digits, no augmentation:

| network | weights | validation accuracy |
|---|---|---|
| Section 22's small CNN (no pooling, giant dense layer) | 189,130 | 98.42% |
| **ConvNet** (pooling, small dense layer) | **96,554** | **99.22%** |
| ConvNet with a *global average* head | 65,834 | 98.99% |

Half the weights and a higher score. (The comparison isn't perfectly controlled, since Section 22's run used plain Adam without batch norm or a schedule, but the direction is clear.)

The last row replaces the dense head with **global average pooling**: average each of the 64 final 7 × 7 maps down to one number, then a tiny 64 → 10 layer. It gets close with even fewer weights. Most modern image networks end this way, since it means the head doesn't care about image size at all. We'll keep the flatten head here because it scored a little higher on validation.

---

### 23.4 The full recipe, and opening the envelope

Now the ConvNet with Part II's recipe (Section 18): AdamW, warm-up + cosine, augmentation (±12° rotations, ±2.5-pixel shifts), 15 epochs, validation checked, then the test set opened **once**. (To keep CPU time reasonable this one trains on the 50,000-digit training split instead of all 60,000.)

| | test accuracy | test mistakes | weights |
|---|---|---|---|
| Part I: dense network (Section 12) | 97.87% | 213 | 109,386 |
| Part II: dense network + recipe (Section 18) | 99.25% | 75 | 536,586 |
| **Part IV: ConvNet + recipe** | **99.59%** | **41** | **96,554** |

(Validation accuracy before opening the test set: 99.50%.)

![Scoreboard: Part I → Part II → Part IV](figures/fig106_scoreboard3.png)

**The mistakes nearly halve again, with 5.6 times fewer weights than Part II's dense network.** That's the payoff for building in what we know about images: local patterns, reused everywhere, with position mattering less as you zoom out.

We're now **20 mistakes** from the 99.79% record Nielsen quoted in Part I, which was set with convolutional networks too, plus more tricks such as ensembles of several networks.

Here are all 41 digits it still gets wrong:

![All 41 mistakes of the ConvNet](figures/fig107_cnn_mistakes.png)

Look at them honestly. Many are hard for people too: a 6 that's almost a straight slash, a 4 whose top is closed into a 9, a 7 drawn with a hooked foot like a 2, some faint broken strokes. A few are genuinely readable (the thick, blocky 5 it called a 3), so there's still room, but most of what's left is the long tail of truly ambiguous handwriting. Its average confidence on these mistakes is 74%, and only 3 of the 41 were made with more than 99% confidence.

> 📓 **Notebook rule:** *build in what you know.* A dense layer assumes nothing about its input. Convolutions assume that nearby pixels matter together and that a pattern means the same thing anywhere. When that assumption is true, as it is for images, you get more accuracy from fewer weights.

---

### 📓 Notebook margin: the equation so far

$$
\text{image} \;\xrightarrow{\;\text{[conv} \to \text{BN} \to \text{ReLU]} \times 2 \;\to\; \text{pool}\;}
\;\xrightarrow{\;\text{[conv} \to \text{BN} \to \text{ReLU]} \times 2 \;\to\; \text{pool}\;}
\;\xrightarrow{\;\text{flatten} \;\to\; \text{dense}\;} \;\text{10 scores}
$$

| idea | what we now know |
|---|---|
| max pooling | keep the largest of each 2 × 2; no weights |
| invariance | 2 pools: features 96% similar after a 1-px shift (conv alone: 89%) |
| receptive field | 3 → 5 → 6 → 10 → 14 → 16 pixels; pooling doubles how fast it grows |
| channels grow as maps shrink | 1 → 32 → 64 channels, 28 → 14 → 7 pixels |
| global average pooling | a tiny head: 65,834 weights, 98.99% in 3 epochs |
| ConvNet + recipe | **99.59% test (41 mistakes)** with 96,554 weights |

---

### What comes next

MNIST is almost used up. 41 mistakes out of 10,000, many of them honestly ambiguous, leaves little to learn from. Real images are much harder: colour, cluttered backgrounds, objects at any size, angle and position.

**Section 24: Real Pictures** moves to **CIFAR-10**: 60,000 small colour photos of aeroplanes, cars, birds, cats, deer, dogs, frogs, horses, ships and trucks. We'll see our MNIST-sized ConvNet struggle, then bring in the modern patterns from Chollet's chapter 9: deeper stacks, **residual connections** (Section 20, now with convolutions), batch norm and augmentation. It's the first problem in this series where depth clearly and measurably *wins*.

---

*References: Yann LeCun, Léon Bottou, Yoshua Bengio & Patrick Haffner (1998), "Gradient-based learning applied to document recognition" (LeNet: convolution and subsampling layers). Michael Nielsen, *Neural Networks and Deep Learning*, ch. 6 (pooling layers; convolutional networks on MNIST; the 99.79% record of Wan et al., 2013). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 8 (max pooling and why it's used; building a convnet from scratch). Min Lin, Qiang Chen & Shuicheng Yan (2013), "Network In Network" (global average pooling). All code in this series is PyTorch.*
