# Physics of AI — Part II · Making It Learn

## 14. Holding the Network Back

> *Section 13 showed the disease: given room, a network memorises its training data, noise and all. It gets more confident on examples it has already seen and less reliable on new ones. This section is the medicine cabinet: four standard ways to push a network toward learning the pattern instead of the particulars. We test every one on the same overfitting setup, so they can be compared fairly.*

---

### 14.1 The test bench

We use the setup from Section 13.4, chosen because it overfits badly:

- **1,000** training digits (a small set makes memorising easy)
- the big network **784 → 512 → 512 → 10** (669,706 weights, more than enough to memorise)
- Adam, learning rate 0.001, batch 128, **150 epochs** (plenty of time to overfit)
- **10,000 validation digits** held out from the training set, used for every comparison
- the official **test set stays sealed** until the very end (Section 13.5)

Without any help, the results are:

| | training | validation accuracy | validation loss |
|---|---|---|---|
| **no regularisation** (epoch 150) | 100% | 89.5% | 0.648 |

That's the number to beat, in both accuracy and loss. Remember from 13.4 that the loss also measures **overconfidence**.

---

### 14.2 Early stopping: quit while you're ahead

The simplest fix costs nothing. In 13.4 the validation loss bottomed out at epoch 7 and then climbed for 143 more epochs. So **keep the weights from the best validation epoch**, not the last:

```python
best_loss, best_state = float("inf"), None
for epoch in range(150):
    train_one_epoch(model)
    val_loss = evaluate(model, X_val, Y_val)
    if val_loss < best_loss:
        best_loss  = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}   # snapshot
model.load_state_dict(best_state)                                            # roll back to the best
```

| | stops at | validation accuracy | validation loss |
|---|---|---|---|
| no regularisation, last epoch | 150 | 89.5% | 0.648 |
| **early stopping** | **7** | 88.5% | **0.395** |

The loss improves a lot (the network hasn't had time to become overconfident), but accuracy is a point *lower*, because at epoch 7 it simply hadn't finished learning yet. The lowest-loss epoch and the highest-accuracy epoch aren't always the same. Pick the one you care about and stop on that.

Early stopping is almost always worth having as a safety net. In practice you usually wait some number of epochs without improvement (the "patience") before giving up, instead of stopping at the very first bump.

---

### 14.3 Weight decay: keep the weights small

**The idea.** Add a penalty to the loss for having large weights:

$$
C_{\text{total}} \;=\; C_{\text{cross-entropy}} \;+\; \lambda \sum_{\text{weights}} w^2
$$

$\lambda$ (lambda) sets how much we care about small weights compared with fitting the data. Now look at what this does to the gradient step from Section 9. The penalty's slope for each weight is $2\lambda w$, so every update becomes:

$$
w \;\leftarrow\; w - \eta\,\frac{\partial C}{\partial w} \;-\; \underbrace{2\eta\lambda\, w}_{\text{shrink toward 0}}
$$

Every step, each weight **decays** a little toward zero, unless the data keeps pushing it back up. Only weights the data really needs stay large. That's where the name comes from.

```python
loss = F.cross_entropy(model(xb), yb)
loss = loss + lam * sum((p**2).sum() for name, p in model.named_parameters() if name.endswith("weight"))
```

(In practice you'd pass `weight_decay=...` to the optimiser, or use `torch.optim.AdamW`, which applies the decay in the cleanest way for Adam. We write it out explicitly here so you can see the penalty.)

**Why small weights mean a smoother boundary.** Go back to Section 6: each ReLU neuron adds a crease to space, and the weights control **how sharp** each crease is and how steeply the output climbs across it. Huge weights allow sudden cliffs, the kind that carve a tiny island around one noisy point. Small weights force every change to be gentle. Here it is on the noisy moons from Section 13.3, same big network, three penalty strengths:

![Weight decay on noisy moons: λ = 0, 0.001, 0.01](figures/fig61_wd_moons.png)

| $\lambda$ | size of all weights $\lVert w \rVert$ | training | new points |
|---|---|---|---|
| 0 | 125.5 | 99.0% | 87.8% |
| **0.001** | **6.1** | 91.8% | **90.7%** |
| 0.01 | 2.5 | 83.7% | 84.5% |

With $\lambda = 0.001$ the weights shrink **20-fold**, the islands vanish, and the boundary follows the moons. With $\lambda = 0.01$ the network is held back so hard it can barely bend, and the boundary is almost a straight line. That's **underfitting**: too simple to capture even the real pattern. Regularisation is a dial, and you tune it on the **validation** set.

On MNIST (1,000 digits):

| | validation accuracy | validation loss | $\lVert w \rVert$ |
|---|---|---|---|
| no regularisation | 89.5% | 0.648 | 26.8 |
| λ = 0.0001 | 89.2% | 0.484 | 13.2 |
| **λ = 0.001** | **89.8%** | **0.364** | 9.6 |
| λ = 0.01 | 87.6% | 0.427 | 6.9 (underfits) |

The accuracy barely moves here, but the validation loss drops by almost half. Weight decay's main effect on MNIST is to **stop the overconfidence**: small weights mean small logits, and small logits can't produce "100% sure".

---

### 14.4 Dropout: train a team where anyone might be absent

**The idea**, from Hinton's group (Srivastava et al., 2014): during training, at every step, **switch off a random fraction $p$ of the hidden neurons**, setting their outputs to zero for that step. The next step, switch off a different random set.

![Dropout: a different random half of the network each step](figures/fig62_dropout.png)

```python
model = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.5),     # each neuron: 50% chance of being off this step
    nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, 10))
```

**Why it helps.** Think of a cricket team where any player might miss any given match. Nobody can become the one specialist everyone else depends on, and every player has to be useful on their own. Neurons are the same. Memorising a particular training digit often relies on a fragile *combination* of neurons, and dropout breaks those combinations at random, so only features that work **independently** survive. Another way to see it: each step trains a different thinned-out sub-network, and the final network behaves like an average of thousands of them.

**At test time, everyone plays.** `model.eval()` switches dropout off. (During training, PyTorch scales the surviving outputs up by $1/(1-p)$, so the average signal is the same in both modes.) That's why Section 12's `evaluate` called `model.eval()` first. Forget it, and your test predictions will be noisy, changing every time you run them.

| | validation accuracy | validation loss |
|---|---|---|
| no regularisation | 89.5% | 0.648 |
| dropout p = 0.2 | 89.9% | 0.639 |
| **dropout p = 0.5** | **90.6%** | 0.578 |

A modest gain of about one point here. Dropout matters more in very large networks with lots of neurons to co-adapt. On a 1,000-digit problem, the next tool is in a different league.

---

### 14.5 Data augmentation: more data for free

Section 13.6 said **more data** is the best cure: going from 1,000 to 5,000 real digits lifted test accuracy from 89.8% to 94.5%. Real data costs money, though. But we *know* something about handwritten digits: a "4" shifted a couple of pixels, or tilted a few degrees, **is still a 4**.

So every time a training digit is used, show a **slightly moved** copy instead: a random rotation of up to ±12° and a random shift of up to ±2.5 pixels, new every time:

![One digit and nine augmented versions](figures/fig63_augmentation.png)

```python
def augment(x, gen):                              # x: a batch of (28, 28) images
    B = x.shape[0]
    angle = (torch.rand(B, generator=gen) * 2 - 1) * math.radians(12)
    shift = (torch.rand(B, 2, generator=gen) * 2 - 1) * (2.5 / 14)     # ±2.5 pixels
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1),
                         torch.stack([sin,  cos, shift[:, 1]], 1)], 1)  # an affine map! (Section 5.6)
    grid = F.affine_grid(theta, (B, 1, 28, 28), align_corners=False)
    return F.grid_sample(x[:, None], grid, align_corners=False)[:, 0]
```

(Look at `theta`: rotation matrix plus translation vector, the **affine transformation** from Section 5.6, now moving images instead of houses.)

| | validation accuracy | validation loss |
|---|---|---|
| no regularisation | 89.5% | 0.648 |
| **augmentation** | **95.0%** | **0.231** |

**A 5.5-point jump from the same 1,000 digits.** That's about what we got in Section 13.6 from **five times more real data** (5,000 digits gave 94.5%). The network never sees exactly the same image twice, so it *can't* memorise pixel patterns. It's forced to learn what stays the same when the digit moves, which is the actual shape.

> 📓 **Notebook rule:** *the best regulariser is knowledge about your problem.* Augmentation works because it encodes a true fact, "moving a digit doesn't change its label", straight into the data. Choose augmentations that preserve the label. Flipping a 6 upside down makes a 9, so don't.

---

### 14.6 All together, and opening the envelope

Each tool attacks memorising from a different side, so they combine:

![Validation curves for each method](figures/fig59_reg_curves.png)

![Summary: validation accuracy and loss for every method](figures/fig60_reg_bars.png)

| method | validation accuracy | validation loss |
|---|---|---|
| none | 89.5% | 0.648 |
| early stopping (epoch 7) | 88.5% | 0.395 |
| weight decay λ = 0.001 | 89.8% | 0.364 |
| dropout p = 0.5 | 90.6% | 0.578 |
| augmentation | 95.0% | 0.231 |
| **augmentation + dropout 0.5 + weight decay 0.001** | **95.3%** | **0.166** |

Look at the loss curves on the left. The unregularised run (grey) climbs steadily from epoch 7 onwards. The combined run (green) is still slowly *improving* at epoch 150. It stopped overfitting.

Every choice so far was made on the validation set. **Now, once, we open the test set**:

| model | test accuracy | test loss |
|---|---|---|
| no regularisation | 89.79% | 0.638 |
| **all three combined** | **95.59%** | **0.152** |

That's +5.8 points of accuracy from the same 1,000 training digits, and a test loss four times lower, so the network is far less often confidently wrong.

---

### 14.7 What to take away

| tool | what it does | worked here? |
|---|---|---|
| **early stopping** | stop before memorising starts | loss yes, accuracy no; a free safety net |
| **weight decay** | small weights mean gentle creases, no islands | fixed overconfidence; too much makes it underfit |
| **dropout** | no neuron can rely on another | +1 point; matters more in bigger networks |
| **augmentation** | new, label-preserving data for free | **+5.5 points**, the biggest win by far |

Two lessons to keep:

1. **Regularisation trades training fit for generalisation.** Every tool made the *training* score worse or no better, and the validation score better. If a change raises validation and lowers training, that's usually a sign it's working.
2. **Too much is as bad as too little.** $\lambda = 0.01$ underfit on both datasets. Every one of these is a dial, tuned on the validation set.

---

### 📓 Notebook margin: the equation so far

$$
C_{\text{total}} \;=\; \underbrace{-\log p_{\text{true}}\big(\text{augment}(\mathbf{x})\big)}_{\text{fit (slightly moved) data}}
\;+\; \underbrace{\lambda\,\lVert W \rVert^2}_{\text{keep weights small}}
\qquad\text{with dropout inside the network, and stop at the best validation epoch}
$$

| idea | what we now know |
|---|---|
| early stopping | best validation loss at epoch 7: 0.395 vs 0.648 at the end |
| weight decay | shrinks weights 20× on moons; removes islands; λ too big underfits |
| dropout | a random half of neurons off each step; `model.eval()` turns it off |
| augmentation | 1,000 augmented digits ≈ 5,000 real ones (95.0% vs 94.5%) |
| combined | test 89.8% → **95.6%** on the same 1,000 digits |

---

### What comes next

Every network so far has started from **random** weights, and we've never asked *how* random. It turns out to matter a great deal. In Section 11.8, a 10-layer network's gradient shrank 17 million times on its way back to the first layer. Part of that was the sigmoid, but part was the **size of the starting weights**.

**Section 15: Starting Right** is about weight initialisation: why all zeros fails completely, why "just random" isn't enough for deep networks, and the simple scaling rule (He initialisation) that keeps signals the same size from the first layer to the last.

---

*References: Michael Nielsen, *Neural Networks and Deep Learning*, ch. 3 ("Regularization": L2 regularisation / weight decay and why smaller weights reduce overfitting; dropout; artificially expanding the training data with small rotations). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 5 (early stopping, weight regularisation, dropout, and "underfitting vs overfitting") and ch. 8 (data augmentation). Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov (2014), "Dropout: A Simple Way to Prevent Neural Networks from Overfitting". All code in this series is PyTorch.*
