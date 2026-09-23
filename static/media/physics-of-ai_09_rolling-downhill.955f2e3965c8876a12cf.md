# Physics of AI — Part I · The Gears

## 9. Rolling Downhill

> *Section 8 turned "how wrong is the network?" into a landscape: a smooth valley over the weights, with the best weights at the bottom. We can draw that valley for two weights. The MNIST network has about 220,000, and its valley can't be drawn or searched point by point. This section is about how to find the bottom of a landscape you can't see, using only the slope of the ground under your feet.*

---

### 9.1 Blindfolded on a hillside

Imagine you're standing somewhere on Nandi Hills, blindfolded, and you want to get to the bottom of the valley. You can't see the map, but you **can** feel the ground under your feet: which way it tilts, and how steeply.

So you do the obvious thing:

1. Feel which way is downhill.
2. Take a small step that way.
3. Repeat.

That's the entire algorithm. It's called **gradient descent**, and every network in this series, from the dinner neuron to the Transformer at the end, is trained by some version of it. The rest of this section just makes "feel the slope" and "small step" precise.

---

### 9.2 The slope in one direction: the derivative

Start with one weight. Hold the PG-food weight fixed and look at how the loss changes as we slide the hunger weight $w_1$. That's the 1-D slice of the valley we drew in Section 8:

![The slope at a point and the step it suggests](figures/fig36_slope_step.png)

The **derivative** is the slope of that curve at the point where you're standing. It answers: *if I nudge $w_1$ by a tiny amount $\varepsilon$, how much does the loss change?*

$$
C(w_1 + \varepsilon) \;\approx\; C(w_1) + \underbrace{\frac{dC}{dw_1}}_{\text{slope}}\cdot\,\varepsilon
$$

At $w_1 = 2$ the slope is negative (−0.158), meaning *increasing $w_1$ lowers the loss*. So we step **right**, the opposite way to the slope:

$$
w_1 \;\leftarrow\; w_1 - \eta\,\frac{dC}{dw_1}
$$

$\eta$ (eta) is the **learning rate**, meaning how big a step we take. A steep slope gives a big step and a gentle slope gives a small one, and the minus sign always points us downhill.

You can even *measure* a slope without calculus: nudge the weight a little each way and see how the loss changes. PyTorch's `autograd` gives the same answer exactly:

```python
w = torch.tensor([2., 3.], requires_grad=True)
loss_fn(w).backward()
w.grad                                    # → [-0.14951, 0.12074]   (autograd)

eps = 1e-3                                # nudge each weight ±0.001 and compare
# numeric slope for w1: -0.14950
# numeric slope for w2:  0.12071            ← same, to 4 decimal places
```

(For these pictures the dinner neuron has just two weights, hunger and PG food, and no bias. Our inputs are standardised, so the best bias is ≈ −0.01 and dropping it changes nothing visible.)

---

### 9.3 Many directions at once: the gradient

With two weights there are two slopes, one per direction. Stack them into a vector and you have the **gradient**:

$$
\nabla C = \left[\,\frac{\partial C}{\partial w_1},\; \frac{\partial C}{\partial w_2}\,\right]
$$

With 220,000 weights it's a vector of 220,000 slopes, one per knob, but the idea is identical. The gradient points in the direction of **steepest uphill**. So the update for *all* weights at once is:

$$
\boxed{\;\mathbf{w} \;\leftarrow\; \mathbf{w} - \eta\,\nabla C\;}
$$

**Why is this guaranteed to go down?** Nielsen has a lovely argument for it. For a small step $\Delta\mathbf{w}$, the change in loss is approximately the gradient dotted with the step (the many-weight version of the formula in 9.2):

$$
\Delta C \;\approx\; \nabla C \cdot \Delta\mathbf{w}
$$

Now *choose* the step to be $\Delta\mathbf{w} = -\eta\,\nabla C$:

$$
\Delta C \;\approx\; \nabla C \cdot (-\eta\,\nabla C) \;=\; -\eta\,\lVert \nabla C \rVert^2 \;\le\; 0
$$

A squared length is never negative, so the loss **always goes down**, as long as the step is small enough for the approximation to hold. Here's a real check at $\mathbf{w} = [2, 3]$ with $\eta = 0.1$:

```python
predicted drop:  -0.003693      # −η ‖∇C‖²
actual drop:     -0.003671      # measured by recomputing the loss
```

The prediction is off by less than 1%. The "small enough" condition matters, and 9.5 shows what happens when you ignore it.

---

### 9.4 Gradient descent, by hand

Here's the whole algorithm in PyTorch, with nothing hidden:

```python
w  = torch.tensor([-2., -2.], requires_grad=True)   # a bad starting guess
lr = 1.0

for step in range(201):
    loss = loss_fn(w)          # 1. how wrong are we?          (Section 8)
    w.grad = None
    loss.backward()            # 2. which way is uphill?       (autograd; Section 11 opens this)
    with torch.no_grad():
        w -= lr * w.grad       # 3. step the other way
```

| step | loss | $(w_1, w_2)$ |
|---|---|---|
| 0 | 2.465 | (−2.00, −2.00) |
| 1 | 1.913 | (−1.41, −1.54) |
| 5 | 0.581 | (0.40, −0.03) |
| 10 | 0.348 | (1.24, 0.68) |
| 25 | 0.253 | (2.15, 1.41) |
| 100 | 0.211 | (3.42, 2.43) |
| 200 | 0.206 | (3.94, 2.84) |

The best possible loss is **0.205**, at about $(4.36, 3.18)$. Watch it happen. The green arrow is "downhill from here":

![Gradient descent walking down the valley](figures/fig37_gd_walk.gif)

Two things to notice:

- **Big strides early, tiny steps late.** The step is $\eta \times$ slope, and the slope flattens near the bottom, so the walker slows down on its own without being told to.
- **The path curves.** At each point it goes *steepest* downhill, which isn't always straight toward the ★. It follows the shape of the valley.

---

### 9.5 The learning rate: the one knob that can ruin everything

The update rule has one number we choose ourselves: $\eta$. Same valley, same start, 60 steps each:

![Four learning rates on the same valley](figures/fig38_learning_rates.png)

![Loss curves for the four learning rates](figures/fig39_lr_curves.png)

| $\eta$ | verdict | loss after 60 steps |
|---|---|---|
| 0.1 | **too small**: correct direction, painfully slow | 0.532 |
| 1 | **steady**: gets there, takes its time | 0.220 |
| 10 | **just right**: nearly at the bottom in a few big strides | 0.205 |
| 40 | **too big**: overshoots, lands on the far wall, bounces around | 0.516 |

At $\eta = 40$ each step is so long it jumps **past** the bottom and up the other side. The "small step" assumption behind $\Delta C \approx -\eta\lVert\nabla C\rVert^2$ is broken, so the guarantee is gone. Push $\eta$ further and the loss doesn't just bounce, it climbs: at $\eta = 300$ it ends at 4.22 after 60 steps, *worse* than where it started (2.47).

Don't read too much into the number 10. What matters is $\eta$ *relative to how curved the valley is*. This valley happens to be gentle, so big numbers are safe. On the Section 7 MNIST network, $\eta = 10$ makes the loss lurch up and down and end at 3.33 after 20 steps, worse than random guessing (2.30). That's why you'll usually see values like 0.1 or 0.001 in practice.

> 📓 **Notebook rule:** *too small wastes time, too big wastes everything.* There's no universal right value, so you **watch the loss curve**. Smoothly down means fine, flat means go bigger, jagged or rising means go smaller.

---

### 9.6 Section 4's promise: why unscaled inputs make training crawl

In Section 4.6 we said unscaled inputs also make gradient descent **zig-zag**, and promised to show it. Here it is.

Same dinner data, same information, but PG food is now rated **0–100** instead of 0–10. Nothing about *which nights you ordered* has changed. Only the ruler has. Here's the valley in these units, next to the standardised one:

![Unscaled canyon vs standardised bowl](figures/fig40_zigzag.png)

**The unscaled valley is a long, narrow canyon.** A tiny change to the PG-food weight is multiplied by numbers up to 100, so the walls across the canyon are **very steep**. The hunger weight only sees numbers up to 10, so the floor along the canyon is **very gentle**.

That puts gradient descent in an impossible spot:

- **The learning rate must be small enough for the steep direction**, or every step leaps across the canyon and up the far wall. That's the red path at $\eta = 0.03$, zig-zagging from wall to wall.
- **But then it's far too small for the gentle direction**, and progress along the canyon floor is a crawl. That's the orange path at $\eta = 0.01$: no zig-zag, barely moving.

Measured, steps to get within 0.01 of the best loss:

| inputs | best learning rate I could find | steps |
|---|---|---|
| PG food on a 0–100 scale | 0.025 (0.05 already bounces and never settles) | **198** |
| standardised (Section 4) | 5 | **12** |
| standardised | 10 | **2** |

**Between 16× and 100× slower, just from the units.** Standardising turns the canyon into a round bowl, where downhill points roughly *at* the bottom and one learning rate suits every direction.

> 📓 **Notebook rule:** *the steepest direction sets how big a step you're allowed to take. The gentlest direction sets how long the trip takes.* The more lopsided the valley, the slower gradient descent is. (The name for "how lopsided" is the **condition number**, and it comes back in Part II, along with fixes like momentum and Adam.)

So that's the full answer to Section 4: scaling your inputs isn't only about fair distances for KNN or keeping sigmoids off their flat parts. **It changes the shape of the valley you have to walk down.**

---

### 9.7 An honest caveat: real valleys aren't bowls

Our dinner neuron's valley is a single smooth bowl: one bottom, and every downhill path leads to it. A deep network's landscape isn't like that. It has flat plateaus, long ridges, saddle points (downhill in some directions, uphill in others) and many different low regions.

Surprisingly, gradient descent still works very well on these landscapes in practice, and *why* is still an active research question. For now, keep the simple picture (feel the slope, step the other way) and know it's a simplification. We'll return to the messier truth in Part III.

---

### 📓 Notebook margin: the equation so far

$$
\text{forward: } C(\mathbf{w}) \qquad\longrightarrow\qquad
\text{gradient: } \nabla C \qquad\longrightarrow\qquad
\text{update: } \mathbf{w} \leftarrow \mathbf{w} - \eta\,\nabla C
$$

| idea | what we now know |
|---|---|
| derivative | the slope of the loss along one weight: "nudge it, how does the loss move?" |
| gradient | all the slopes at once; it points steepest **uphill** |
| gradient descent | step against the gradient; $\Delta C \approx -\eta\lVert\nabla C\rVert^2 \le 0$ |
| learning rate $\eta$ | too small: crawl; too big: bounce or blow up; judge it by the loss curve |
| input scaling | turns a canyon (198 steps) into a bowl (2–12 steps) |
| real networks | bumpier landscapes, same basic idea |

---

### What comes next

There's a problem we skipped. Every step in this section computed the loss, and its gradient, over **all** the training examples. That's fine for 400 dinner nights. For 60,000 MNIST digits it means reading the whole dataset just to take **one** step, and training needs thousands of steps.

**Section 10: SGD** is about getting a *good enough* gradient from a small random handful of examples: mini-batches, why the randomness actually helps, and what "epoch" and "batch size" really mean. That will finally explain the `for i in torch.randperm(4000).split(64)` line hiding in Section 7's training loop.

---

*References: Michael Nielsen, *Neural Networks and Deep Learning*, ch. 1 ("Learning with gradient descent": the ball-rolling picture, $\Delta C \approx \nabla C \cdot \Delta v$, and choosing $\Delta v = -\eta\nabla C$ so that $\Delta C \le 0$). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 2 ("The engine of neural networks: gradient-based optimization": derivatives, gradients, learning rate too small or too large). All code in this series is PyTorch.*
