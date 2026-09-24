# Physics of AI — Part IV · Seeing

## 24. Real Pictures

> *MNIST is nearly solved: 41 mistakes out of 10,000. It's time for a harder problem. **CIFAR-10** has 60,000 tiny colour photos of planes, cars, birds, cats and six other things, taken in the real world with messy backgrounds, odd angles and bad lighting. This is where the gap between "a network" and "a network built for images" gets big, and it's the first place in this series where **depth, with residual connections, measurably wins**.*

---

### 24.1 The new data

![CIFAR-10 samples: 8 photos from each of the 10 classes](figures/fig108_cifar_samples.png)

| | MNIST | CIFAR-10 |
|---|---|---|
| image | 28 × 28, 1 channel (grey) | 32 × 32, **3 channels** (red, green, blue) |
| numbers per image | 784 | **3,072** |
| what's in it | one centred digit on a black background | an object anywhere, any size, any angle, with background clutter |
| classes | 10 digits | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| split used here | 50k train / 10k val / 10k test | **45k train / 5k val / 10k test** |

Look at the cat row. One cat is black on a red blanket, another is a grey blur against a window, another is mostly background. A bird can be a close-up of a head or a speck on the water. **The same class looks completely different from photo to photo**, and that's the whole difficulty.

The input is now a tensor of shape `(N, 3, 32, 32)`, with channels first as PyTorch expects. We standardise each colour channel separately using the training set's mean and spread (Section 4 again):

```python
Xf   = X_train.float() / 255                               # (45000, 3, 32, 32), values 0..1
mean = Xf.mean(dim=(0, 2, 3), keepdim=True)                # one mean per colour channel
std  = Xf.std(dim=(0, 2, 3), keepdim=True)
norm = lambda Z: (Z.float() / 255 - mean) / std            # apply the same numbers to val and test
```

The only change for convolutions is the first layer: `nn.Conv2d(3, 32, 3)` instead of `nn.Conv2d(1, 32, 3)`. Each first-layer kernel is now 3 × 3 × **3**, so it looks at a small patch of all three colours at once.

(Two honest notes on the data. The official CIFAR-10 site was blocked in my environment, so I used a public mirror that stores the same 60,000 images as JPEG files. JPEG compression changes pixels very slightly, so numbers may differ a little from the original files. Also, everything here trains on **2 CPU cores**, so each network gets only **10 epochs**. The original paper trained for about 180 epochs on GPUs.)

---

### 24.2 Augmentation for photos: crop and flip

Section 14 augmented digits with small rotations and shifts. For photos, the standard pair is:

- **Random crop.** Pad the 32 × 32 photo to 40 × 40 (by mirroring its edges), then cut a random 32 × 32 window out of it. That shifts the image by up to 4 pixels in any direction.
- **Horizontal flip.** Mirror the photo left-to-right half the time. A horse facing left is still a horse. (We **don't** flip digits, because a mirrored 2 isn't a 2. Augmentation must respect what the label means.)

![Random crops and flips of four training photos](figures/fig109_augment.png)

```python
def augment(x, g):                                           # x: (B, 3, 32, 32)
    B  = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")              # (B, 3, 40, 40)
    i, j = torch.randint(0, 9, (B,), generator=g), torch.randint(0, 9, (B,), generator=g)
    rows = (i[:, None] + torch.arange(32))[:, :, None].expand(B, 32, 32)
    cols = (j[:, None] + torch.arange(32))[:, None, :].expand(B, 32, 32)
    out  = xp[torch.arange(B)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)   # a random 32×32 window each
    flip = torch.rand(B, generator=g) < 0.5
    out[flip] = out[flip].flip(3)                            # mirror left-right
    return out
```

A new crop and flip is drawn every time a photo is used, so across 10 epochs the network never sees exactly the same picture twice.

---

### 24.3 Four networks, one recipe

Every network gets exactly the same Part II recipe (Section 18): AdamW (learning rate 0.002, weight decay 5 × 10⁻⁴), batch 128, 1 epoch of warm-up then cosine decay, crop + flip augmentation, **10 epochs**, same seed.

**1. The dense network (MLP).** Flatten 3,072 numbers → 512 → 256 → 10, with batch norm. It has **1.7 million** weights, 1.5 million of them in the first layer alone.

**2. Section 23's ConvNet**, unchanged except for 3 input channels: conv, conv, pool, conv, conv, pool, dense. **107k weights.**

**3 & 4. A 20-layer network, plain and residual.** This is the **ResNet-20** layout from He et al.'s 2015 paper, built from Section 20's residual block, now with convolutions:

```python
class Block(nn.Module):
    def __init__(self, cin, cout, stride, residual):
        super().__init__()
        self.residual = residual
        self.c1 = conv_bn(cin, cout, stride)        # 3×3 conv + BatchNorm (stride 2 halves the map)
        self.c2 = conv_bn(cout, cout)               # 3×3 conv + BatchNorm
        self.short = (nn.Identity() if stride == 1 and cin == cout
                      else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))
    def forward(self, x):
        h = self.c2(F.relu(self.c1(x)))
        return F.relu(h + self.short(x)) if self.residual else F.relu(h)     # the only difference
```

- **Three stages** of 3 blocks each, with 16 → 32 → 64 channels on 32 → 16 → 8 pixel maps. Channels grow as maps shrink, just like Section 23.
- It shrinks maps with **stride-2 convolutions** instead of max pooling: the kernel jumps 2 pixels at a time, so the output is half the size. It does the same job as pooling, but it's learned.
- When a block changes the shape, the skip can't be a plain copy, so a cheap **1 × 1 convolution** reshapes it (the "projection shortcut").
- The head is **global average pooling** + one 64 → 10 layer (Section 23.3). That's 64 × 10 + 10 = 650 weights for the whole decision.
- Layer count: 1 stem conv + 9 blocks × 2 convs + 1 dense = **20 layers, 272k weights**.

The plain version is **identical** except that it drops the `+ self.short(x)`. Same layers, same weights, same everything else.

---

### 24.4 The results

![Validation curves for all five runs, and training vs validation accuracy](figures/fig110_cifar_curves.png)

| network | weights | training acc | **validation acc** |
|---|---|---|---|
| dense network (MLP) | 1,708,042 | 59.5% | 57.5% |
| Section 23 ConvNet | 106,730 | 82.3% | 80.1% |
| plain 20-layer CNN | 272,474 | 79.6% | 77.4% |
| **ResNet-20** | 272,474 | 84.2% | **81.7%** |

(Training accuracy is measured on 5,000 un-augmented training photos, in eval mode.)

Four findings.

**1. Dense networks fall apart on real photos.** On MNIST, with the full recipe, the dense network was less than half a point behind the ConvNet (99.25% vs 99.59%). On CIFAR-10 it's **23 points** behind, with **16 times more weights**. It isn't overfitting either: it can't even fit its own training data (59.5%). Photos are too varied for "one weight per pixel". A cat in the top-left and a cat in the bottom-right share almost no pixel values, so the network has to learn every position separately. Section 22's shared, sliding kernels are no longer a nice extra. **For real images they are the whole game.**

**2. Plain depth hurts, even at 20 layers.** The plain 20-layer network has 2.5× the weights of Section 23's 5-layer ConvNet (4 convs + 1 dense) and scores **worse**, on training data as well as validation. It's Section 20's degradation problem, now with convolutions, and batch norm doesn't prevent it.

**3. Residual connections turn depth into a win.** Add `+ self.short(x)` and the same 20 layers go from 77.4% to **81.7%**, 4.3 points better, and ahead of the shallow ConvNet.

**4. But be honest about the size of that win.** ResNet-20 beats Section 23's ConvNet by **1.6 points**. That's real (bigger than the ~0.2-point seed noise from Section 18), but not dramatic. Ten CPU epochs is a short budget, and deeper networks usually need longer to pay off. He et al. trained this exact layout for about 180 epochs and reported **91.25%**. The claim here is modest: at the same depth, residual beats plain clearly. Depth then beats shallow, but only once residual connections make depth trainable.

---

### 24.5 What happens at 56 layers?

He et al.'s famous figure compared 20 and 56 plain layers on CIFAR-10. Let's push our block to **56 layers** (9 blocks per stage), plain vs residual, for 3 epochs:

![Training loss at 56 layers: plain vs residual](figures/fig111_depth56.png)

| 56 layers, 3 epochs | weights | training loss | training acc | validation acc |
|---|---|---|---|---|
| plain-56 | 855,770 | 1.64 | 38.6% | 38.3% |
| **ResNet-56** | 855,770 | **0.75** | **73.6%** | **71.4%** |

At 56 layers the difference stops being a few points and becomes a gap you can't miss. With the same weights, the same budget and the same seed, the plain network is still stuck below 40% after 3 epochs, **on its own training data**. The residual one has passed 70%.

For scale: after 3 epochs, plain-20 had already reached 59.4% validation accuracy, and it was still mid-schedule, with a higher learning rate than plain-56 had by its third epoch. So the comparison, if anything, favours plain-56. Going from 20 to 56 plain layers made it **worse**, which is exactly the result that puzzled He and colleagues in 2015 (Section 20.1). The residual version doesn't have that problem.

(These are only 3-epoch runs, each with its own short cosine schedule, because a 56-layer epoch takes over 4 minutes on 2 CPUs. It's a check of trainability, not a final score.)

---

### 24.6 Augmentation, measured

Same ResNet-20, same 10 epochs, with no crops or flips:

| ResNet-20 | training acc | validation acc | gap |
|---|---|---|---|
| no augmentation | **93.9%** | 79.6% | 14.3 points |
| **crop + flip** | 84.2% | **81.7%** | 2.5 points |

Without augmentation the network **memorises**: its training accuracy shoots up to 93.9% while validation lags 14 points behind. That's Section 13's overfitting, back in full force. With augmentation the training score is *lower*, because every batch holds photos it hasn't seen in that exact form, but validation is **2 points higher** and the gap shrinks to 2.5 points. On MNIST augmentation was a finishing touch. On small photo datasets it's one of the most important regularisers you have.

---

### 24.7 Opening the envelope

Only ResNet-20 goes to the test set, **once**:

> **ResNet-20: 81.50% test accuracy** (validation was 81.66%)

Validation and test agree to within 0.2 points, so the validation set did its job (Section 13).

![Confusion matrix on the test set](figures/fig112_confusion.png)

Per class, the network is far from uniform:

| easiest | | hardest | |
|---|---|---|---|
| automobile | 93.5% | **cat** | **63.4%** |
| ship | 90.8% | bird | 68.3% |
| truck | 89.1% | dog | 72.9% |
| frog | 89.0% | deer | 77.7% |

The single biggest confusion is **cat ↔ dog**: 283 of the 1,850 test mistakes, about 1 in 6. Next come cat ↔ frog (110), automobile ↔ truck (99), bird ↔ frog (96) and bird ↔ cat (93). These are sensible confusions. Small furry animals on a sofa look alike at 32 × 32 pixels, and so do cars and pickup trucks. Machines are the easy classes (distinctive shapes, often on plain backgrounds of road, sky or sea). Animals are the hard ones.

![The 16 most confident test mistakes](figures/fig113_cifar_mistakes.png)

The most confident mistakes show **what the network actually leans on**. Two cars on grey water-like backgrounds become "ship". An aeroplane over the sea becomes "ship", and a boat against the sky becomes "airplane". A cat photographed on green becomes "frog". The pattern is clear: **background colour is a strong clue**, and when it points the wrong way, the network follows it with full confidence. Sky suggests plane, water suggests ship, green suggests frog or deer.

That's a shortcut, not understanding, and it's what Section 25 will look into directly. Overall, the average confidence on the 1,850 mistakes is 61%, and 17 of them were made with more than 99% confidence.

> 📓 **Notebook rule:** *on real images, structure beats size, and residuals make depth pay.* The dense network had 16× more weights and lost by 23 points. The plain 20-layer net lost to a 5-layer one until one `+ x` turned it into the winner.

---

### 📓 Notebook margin: the equation so far

$$
\text{photo} \xrightarrow{\text{conv-BN-ReLU}}
\underbrace{\Big[\;\mathbf{h} \leftarrow \text{ReLU}\big(\mathbf{h} + \text{BN}(\text{conv}(\text{ReLU}(\text{BN}(\text{conv}(\mathbf{h})))))\big)\Big]}_{\times 9,\;\;16 \to 32 \to 64 \text{ channels}}
\xrightarrow{\text{global avg pool}} \xrightarrow{\text{dense}} \text{10 scores}
$$

| idea | what we now know |
|---|---|
| CIFAR-10 | 3 × 32 × 32 colour photos; the same class looks very different from photo to photo |
| dense on photos | 57.5% with 1.7M weights; can't even fit the training set |
| ConvNet on photos | 80.1% with 107k weights |
| plain 20 layers | 77.4%: worse than 5 layers (degradation) |
| ResNet-20 | 81.7% val, **81.50% test**, 272k weights, 10 CPU epochs |
| crop + flip | train–val gap 14.3 → 2.5 points; +2 points validation |
| 56 layers, 3 epochs | plain 38.6% train vs residual 73.6%: degradation, fixed |
| test analysis | cat ↔ dog is 1 in 6 mistakes; backgrounds drive the confident errors |
| stride-2 conv, 1×1 shortcut, GAP head | the ResNet toolkit |

---

### What comes next

We've built networks that see, and measured how well they do. But *what* are they looking for? Section 22 showed first-layer kernels acting as edge detectors. What about layer 10, or 19?

**Section 25: What the Network Sees** opens up the trained ResNet-20. We'll look at its feature maps layer by layer, generate the image that most excites a chosen channel (by running gradient descent on the **input** instead of the weights), and draw **heatmaps** of which parts of a photo drove each decision, to see whether the network looks at the horse or at the grass around it. That closes Part IV.

---

*References: Alex Krizhevsky (2009), "Learning Multiple Layers of Features from Tiny Images" (CIFAR-10). Kaiming He, Xiangyu Zhang, Shaoqing Ren & Jian Sun (2015), "Deep Residual Learning for Image Recognition" (ResNet-20/56 on CIFAR-10: 3 stages of 16/32/64 channels, stride-2 convs, global average pooling, crop + flip augmentation, the plain-vs-residual comparison). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 8–9 (data augmentation; modern convnet architecture patterns: residual connections, batch normalisation). Michael Nielsen, *Neural Networks and Deep Learning*, ch. 6 (expanding the training data; convolutional networks). All code in this series is PyTorch.*
