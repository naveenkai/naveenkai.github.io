# Physics of AI — Part IV · Seeing

## 25. What the Network Sees

> *Section 24 trained a ResNet-20 that gets 81.5% on photos it has never seen. That's a number. It doesn't tell us **what** the network looks at, or **why** it said "ship" when the photo was clearly a car. This section opens up the trained network, with no new training at all, and asks it six questions. Some of the answers are reassuring, and some are not. The last one is the most important result in this part of the series.*

---

### 25.1 Layer by layer

Feed one photo through and look at the feature maps (Section 22) at four depths:

![First-layer kernels, and feature maps at three depths](figures/fig114_layers.png)

- **The first layer's 16 kernels** are tiny 3 × 3 colour patterns: bright/dark edges, colour contrasts (green vs pink, blue vs orange). They're the colour versions of the MNIST edge detectors in Section 22.6.
- **After the first conv (32 × 32)** the maps are still clearly the horse: outlines, legs, the ground line, the bright back. Each map is a *filtered copy* of the photo.
- **End of stage 2 (16 × 16)**: you can still guess the layout (vertical leg-like bars, a horizontal body), but the maps no longer look like a picture.
- **End of stage 3 (8 × 8)**: just blobs. Each map says *roughly where* something is, but what that "something" is has become abstract.

This is Chollet's picture of a deep network as an **information distillation pipeline**: as you go deeper, the maps carry less about *what the pixels look like* and more about *what the object is*.

---

### 25.2 Measuring "more about what the object is"

That last sentence is a claim, so let's measure it. Take the features at each stage, average each channel down to a 2 × 2 grid, and train the simplest possible classifier on them: **one linear layer** (Section 5, a straight-line decision). The network itself stays frozen. We only ask how *easy to read* the class is at each depth.

![Linear classifier on each stage's features; channel purity by stage](figures/fig115_probes.png)

| features taken from… | numbers per photo | linear classifier accuracy |
|---|---|---|
| raw pixels (2 × 2 average colour) | 12 | 28.5% |
| after the first conv | 64 | 46.9% |
| end of stage 1 | 64 | 55.7% |
| end of stage 2 | 128 | 71.5% |
| end of stage 3 | 256 | **81.0%** |

Accuracy climbs steadily with depth. Compare "after first conv" and "end of stage 1": the **same** 64 numbers per photo, but the deeper ones are 9 points easier to read. By the last stage a straight line gets 81.0%, essentially the whole network's score. This is Section 7's **uncrumpling paper**, now measured on real photos: each stage flattens the data a little more, until the classes can be separated by a plane.

(This kind of test is called a **linear probe**. It's one of the most widely used tools for understanding what a network, including a language model, has learned at each layer.)

---

### 25.3 What does one channel respond to?

Pick one channel, run all 5,000 validation photos through the network, and collect the **9 photos** that make that channel fire hardest:

![Top-9 validation photos for four stage-3 channels; top patches for four stage-1 channels](figures/fig116_channels.png)

**Top row, stage 3.** Channel 47 fires for **trucks** (big boxy vehicles, whatever the colour). Channel 51 fires for **boats**, channel 56 for **horses on grass**, and channel 23 for **close-up fluffy dog faces**. Nobody named these channels. The network invented a "truck detector" and a "dog face detector" because they helped reduce the loss.

**Bottom row, stage 1.** Early channels see only a 15 × 15 patch of the photo (their receptive field, Section 23.2), so here we show that patch. These respond to **local patterns**: a bright diagonal edge, a dark vertical bar, a light-over-dark boundary. They fire on cars, dogs and planes alike, because edges are everywhere.

The right-hand plot in the figure above measures this over **every** channel. Of each channel's top 9 photos, how many share a class?

| stage | top-9 photos sharing a class (average) |
|---|---|
| random photos (for comparison) | 2.6 |
| stage 1 | 3.4 |
| stage 2 | 4.2 |
| stage 3 | **7.0** |

Early channels are **general** (edges and textures, useful for every class). Late channels are **specific** (one object type each). This is the "edges → parts → objects" ladder from Section 7.5, which we could only promise back then.

---

### 25.4 Asking the network to draw

Flip the question around. Instead of searching real photos for what excites a class, **create** the photo that excites it most. Start from faint random noise and run **gradient descent on the input pixels**, with the weights frozen, to push up one class score:

```python
x = (torch.randn(1, 3, 32, 32) * 0.1).requires_grad_(True)      # the image is now the parameter
opt = torch.optim.Adam([x], lr=0.05)
for step in range(400):
    shifted = torch.roll(x, shifts=random_shift(), dims=(2, 3))   # small random jitter each step
    score = model(shifted)[0, c]                                  # class c's score
    loss = -score + 0.02 * (x ** 2).sum()                         # climb the score, keep pixels small
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 4 == 0: x.data = gaussian_blur(x.data)              # smooth away pixel-level noise
```

It's the same backprop as Section 11, but the gradient is taken with respect to the *image* instead of the weights. The jitter, blur and size penalty stop it from producing pure noise (more on that danger in 25.7).

![Images generated to maximise each class, two random starts each](figures/fig117_dreams.png)

(Colours are stretched per image so the patterns are visible.)

They're not photos, but the network's idea of each class is recognisable: **antlers and thin legs** for deer, **rows of legs** for horse, a **hull on water** for ship, **wheels under a box** for truck, a **round body with a curved rim** for automobile, an **eye and beak** for bird, and **bumpy, mottled texture** for frog. Notice what's repeated: the network doesn't care about *one* horse. It draws legs over and over, because "more leg evidence" means a higher horse score. What it has learned is a collection of **parts and textures**, not a single whole-object template.

---

### 25.5 Where did it look? Heatmaps

Our ResNet ends with **global average pooling** and one dense layer (Section 24.3). That design gives us a heatmap for free. The class score is

$$
\text{score}_c = b_c + \sum_k W_{c,k}\;\underbrace{\tfrac{1}{64}\sum_{i,j} A_k[i,j]}_{\text{average of channel }k}
\;=\; \frac{1}{64}\sum_{i,j}\;\underbrace{\Big(b_c + \sum_k W_{c,k} A_k[i,j]\Big)}_{\text{score at position }(i,j)}
$$

The averaging and the dense layer can be done **in either order**. So the class score is literally the **average of 64 position scores**, one per cell of the 8 × 8 grid. Map those 64 numbers back onto the photo and you see which regions voted for the class. This is the **class activation map (CAM)** of Zhou et al. (2016):

```python
A = features_stage3(x)                                  # (64 channels, 8, 8)
cam = torch.einsum("k,kij->ij", W[c], A) + b[c]         # (8, 8): one score per position
assert torch.isclose(cam.mean(), model(x)[0, c])        # the class score is exactly the mean
```

(For networks without this clean ending, **Grad-CAM** gets the same kind of map using gradients. For ours it gives the same map up to a constant scale.)

As a second, completely independent check, use **occlusion** (Zeiler & Fergus, 2014). Slide a grey 8 × 8 square over the photo and record how much the predicted class's probability *drops* at each position. Where hiding the patch hurts most, the network was relying on it.

![CAM and occlusion heatmaps: six correct predictions and four confident mistakes](figures/fig118_heatmaps.png)

On the six correct predictions, the CAM sits on the **object**: the plane's body, the horse, the ship's hull, the bird's eye, the truck's cab. Across all correctly classified validation photos, **48%** of the positive CAM weight falls in the centre quarter of the grid, where objects usually are (a uniform map would put 25% there).

Occlusion is more selective. The network often depends on **one small key region**: a single spot on the horse's leg, the middle of the ship's hull, the deer's head. Hide that one patch and the prediction collapses. The two methods answer slightly different questions ("what voted?" vs "what couldn't it do without?"), so they don't have to agree exactly.

The four **mistakes** (from Section 24's most confident errors) are revealing. For the car it called "ship", the CAM covers the car, but occlusion points at the **road surface below it**: grey, flat and water-like. The aeroplane called "ship" was photographed over the sea, and occlusion points at the **water**. The network was looking at the right object, but the evidence that tipped the decision came from **context**.

---

### 25.6 The background test

Heatmaps are suggestive. Let's get a hard number. Hide the **centre 16 × 16 of every validation photo** (a quarter of the image, where the object usually is) with a flat block of the dataset's average colour, and see what the network still gets right:

![Accuracy per class with the centre hidden](figures/fig119_hide_centre.png)

| | accuracy |
|---|---|
| full photos | 81.7% |
| **centre quarter hidden** | **43.2%** |
| only the centre quarter kept (the rest hidden) | 41.0% |
| chance | 10% |

With the middle of the photo **gone**, the network still scores **four times chance**. It reads a lot from the edges of the frame: the sky, the water, the grass, the road. By class:

- **Ship: 67%** with the boat hidden. Water above and below is enough.
- **Airplane 46%, bird 49%**: sky and branches.
- **Dog 10%, frog 22%, deer 27%**: these depend on the animal itself. (Their backgrounds are probably shared with other classes: dogs and cats both sit indoors, and frogs and deer both sit on mottled ground.)
- **Truck 89%**, which is suspicious. Hiding the centre makes the network say "truck" for **26%** of *all* photos. A big flat grey rectangle looks like the side of a truck's box to it. That isn't background knowledge, it's a **shortcut**.

(Honest caveat: a grey square in the middle is something the network never saw in training, so part of the drop is just "weird input". But the point stands: most of the time it can guess the class **without seeing the object**.)

Is this bad? Partly it's sensible. Ships really are on water, and a human uses context too. But it means the network's 81.5% is built partly on **correlations in how the photos were taken**, not only on the objects. Put a ship on a road and it will struggle. This is called **shortcut learning**, and it's one of the main reasons a model that looks great on its test set can fail in the real world.

> 📓 **Notebook rule:** *a network learns whatever predicts the label, not what you meant it to learn.* If sky predicts "airplane", it will learn sky. Test accuracy can't tell the difference. Only experiments like these can.

(A related check: turn every photo **greyscale** and accuracy only drops from 81.7% to 77.9%. Colour helps a little, but shape and texture carry most of the decision.)

---

### 25.7 The unsettling one: invisible changes

Section 25.4 used gradient descent on the input to make a class score go **up**. Now start from a real photo and use the same trick to make the **correct** class score go **down**, but only allow each pixel to change by a tiny amount $\varepsilon$, a few steps out of 255:

```python
def attack(x, y, eps, steps=10):
    d = torch.zeros_like(x, requires_grad=True)
    for _ in range(steps):
        loss = F.cross_entropy(model(x + d), y)          # how wrong is it on the true label?
        g, = torch.autograd.grad(loss, d)
        d.data += (eps / 4) * g.sign()                   # step every pixel to make it MORE wrong
        d.data.clamp_(-eps, eps)                         # but never more than eps per pixel
    return x + d
```

(This is **PGD**, projected gradient descent, from Madry et al., 2018. Its one-step version is Goodfellow's **FGSM**, 2015.)

![Invisible perturbations flip confident predictions; accuracy vs the size of the change](figures/fig120_adversarial.png)

| largest change per pixel (out of 255) | validation accuracy |
|---|---|
| 0 (clean photos) | 81.3% |
| 0.5 | 66.2% |
| 1 | 46.9% |
| 2 | 16.7% |
| 4 | **0.9%** |
| 8 | 0.0% |

(Measured on 2,000 validation photos.)

Look at the photos on the left. A horse the network was **99.9%** sure about becomes "airplane" at **97.6%**. A car at 99.2% becomes "truck" at 100%, and a ship becomes "airplane". The change (shown in the middle, amplified 32×) is structured noise, and **you can't see it** in the altered photo. At a maximum of 4 steps out of 255 per pixel, accuracy falls from 81% to **under 1%**.

These are **adversarial examples** (Szegedy et al., 2013). They exist for essentially every standard network, not just ours. Here's why, in this series' own terms. The class score is built from sums like $\sum w_i x_i$ over **3,072** inputs (Section 2). Nudge every pixel by a tiny $\varepsilon$ in the direction of its weight's sign, and the score moves by $\varepsilon \sum |w_i|$: thousands of tiny pushes, **all in the same direction**. No single pixel changes visibly, but together they move the input across a decision boundary. High-dimensional spaces are strange that way: every photo sits surprisingly close to a boundary, in some direction a human would never look.

(Small honest detail: with only 2 steps out of 255, the most confident photos sometimes survived, which is why the examples above use 4.)

> 📓 **Notebook rule:** *a network can be right for reasons that aren't ours.* 81% accuracy, sensible heatmaps, readable "dreams", and yet a change you can't see flips its answer. A network that sees well isn't the same as one that sees the way we do.

---

### 📓 Notebook margin: Part IV in one table

$$
\text{score}_c \;=\; \frac{1}{64}\sum_{i,j}\Big(b_c + \sum_k W_{c,k}\,A_k[i,j]\Big)
\qquad\text{(the class score is an average of position scores)}
$$

| section | idea | biggest measured effect |
|---|---|---|
| 22 | convolution: one small kernel, slid everywhere | dense net ignores pixel order (96.3% shuffled = unshuffled); CNN +2 pts |
| 23 | pooling, receptive fields, the ConvNet | MNIST: **99.59% test**, 41 mistakes, 96k weights |
| 24 | real photos: ResNet-20 + crop/flip | CIFAR-10: dense 57.5% → **ResNet-20 81.5% test**; plain-56 38.6% train vs residual 73.6% |
| 25 | linear probes | 28.5% (pixels) → 81.0% (stage 3): depth uncrumples |
| 25 | channel specialisation | top-9 class purity 3.4 → 7.0 out of 9 |
| 25 | CAM + occlusion | right object usually; context tips the mistakes |
| 25 | background test | 43.2% with the centre hidden (4× chance) |
| 25 | adversarial examples | 4/255 per pixel: 81.3% → **0.9%** |

---

## End of Part IV

Part IV gave networks eyes. **Convolutions** build in what we know about images: look locally, reuse the same detector everywhere (22). **Pooling** and depth widen the view from strokes to whole objects (23). **Residual connections** make that depth trainable on real photos (24). And looking inside (25) shows a network that really does build up from edges to parts to objects, while also taking **shortcuts** and hiding **blind spots** no accuracy number reveals.

But everything so far has one thing in common: the input arrives **all at once**, as a fixed-size grid. Language doesn't. A sentence is a **sequence** of any length, where the meaning of word 12 can depend on word 2. There's no fixed grid to slide a kernel across, and "which words are next to each other" isn't the whole story.

**Part V: Remembering** starts with **Section 26: Text as Numbers**. How do you turn words into tensors? We'll build a vocabulary, one-hot vectors, and the bag-of-words model (and see what it throws away), then set up the problem that recurrent networks, and eventually attention, were invented to solve.

---

*References: François Chollet, *Deep Learning with Python*, 3rd ed., ch. 10 ("Interpreting what convnets learn": visualising intermediate activations, visualising filters by gradient ascent in input space, class activation heatmaps). Bolei Zhou et al. (2016), "Learning Deep Features for Discriminative Localization" (CAM). Ramprasaath Selvaraju et al. (2017), "Grad-CAM". Matthew Zeiler & Rob Fergus (2014), "Visualizing and Understanding Convolutional Networks" (occlusion sensitivity; layer-by-layer features). Guillaume Alain & Yoshua Bengio (2016), "Understanding intermediate layers using linear classifier probes". Alexander Mordvintsev, Christopher Olah & Mike Tyka (2015), "Inceptionism" (jitter and blur regularised activation maximisation). Robert Geirhos et al. (2020), "Shortcut Learning in Deep Neural Networks". Christian Szegedy et al. (2013), "Intriguing properties of neural networks"; Ian Goodfellow, Jonathon Shlens & Christian Szegedy (2015), "Explaining and Harnessing Adversarial Examples" (FGSM, the linearity explanation); Aleksander Madry et al. (2018), "Towards Deep Learning Models Resistant to Adversarial Attacks" (PGD). All code in this series is PyTorch.*
