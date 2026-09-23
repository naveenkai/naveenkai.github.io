# Physics of AI — Part II · Making It Learn

## 16. Better Steps

> *Since Section 6 we've trained almost everything with `torch.optim.Adam`, described in one line as "momentum plus a separate step size for every weight". This section builds Adam properly, one idea at a time, starting from the plain gradient step of Section 9. Each idea is tested on the same lopsided canyon and then on MNIST. After that we look at the other half of taking good steps: changing the learning rate **during** training.*

---

### 16.1 Where plain SGD struggles

The plain gradient step from Section 9 is:

$$
w \;\leftarrow\; w - \eta\, g \qquad (g = \text{the gradient for this weight})
$$

It has two weaknesses, and we've met both:

1. **One learning rate for every direction.** In Section 9's canyon (PG food rated 0–100), the steep direction forced a small $\eta$ and the gentle direction then crawled: **198 steps** at the best learning rate. (All counts in this section use each optimiser's own best learning rate from a fine search: 198 for plain gradient descent at η = 0.025, as in Section 9. Section 10's 19 momentum steps were at a fixed η = 0.02; tuned, momentum needs 16.)
2. **Noise.** Mini-batch gradients jitter (Section 10), and each step follows that step's jitter.

Everything in this section is a fix for one or both.

---

### 16.2 Idea 1: momentum (keep your velocity)

We met this in Section 10.6. Instead of stepping along the current gradient, step along a **running sum** of recent gradients:

$$
v \;\leftarrow\; \beta\, v + g \qquad\qquad w \;\leftarrow\; w - \eta\, v \qquad (\beta \approx 0.9)
$$

Directions where the gradient keeps flipping sign (across the canyon) cancel out in $v$. Directions where it keeps pointing the same way (along the canyon floor) build up speed. It also smooths out mini-batch noise, since $v$ is effectively an average of the last ~10 gradients.

```python
class Momentum:
    def __init__(self, params, lr, beta=0.9):
        self.p, self.lr, self.beta = list(params), lr, beta
        self.v = [torch.zeros_like(p) for p in self.p]
    def step(self):
        with torch.no_grad():
            for p, v in zip(self.p, self.v):
                v.mul_(self.beta).add_(p.grad)      # velocity = 0.9 × old velocity + new gradient
                p -= self.lr * v
```

**Canyon: 198 → 16 steps.**

---

### 16.3 Idea 2: RMSProp (a step size for every weight)

Momentum fixes the zig-zag but still uses one $\eta$ for everything. Here's the root of the canyon problem, from the very first step:

| | gradient |
|---|---|
| hunger weight | −0.99 |
| PG-food weight | −6.05 |

The PG-food gradient is **6× bigger**, purely because its feature was measured on a 0–100 scale (Section 4). Any single $\eta$ is either too big for one weight or too small for the other.

**RMSProp**'s idea (Hinton, in a 2012 lecture): keep a running average of each weight's **squared** gradient, and divide each step by its square root, which is the gradient's typical size:

$$
s \;\leftarrow\; \beta\, s + (1-\beta)\, g^2 \qquad\qquad w \;\leftarrow\; w - \eta\,\frac{g}{\sqrt{s} + \epsilon}
$$

A weight whose gradients are usually large gets divided by something large. A weight whose gradients are usually small gets divided by something small. **Every weight ends up taking steps of roughly size $\eta$, whatever the scale of its gradient.** ($\epsilon \approx 10^{-8}$ just prevents division by zero.)

![Raw gradients differ 6×; after RMSProp's division the steps are equal](figures/fig69_per_weight.png)

```python
class RMSProp:
    def __init__(self, params, lr, beta=0.99, eps=1e-8):
        self.p, self.lr, self.beta, self.eps = list(params), lr, beta, eps
        self.s = [torch.zeros_like(p) for p in self.p]
    def step(self):
        with torch.no_grad():
            for p, s in zip(self.p, self.s):
                s.mul_(self.beta).add_((1 - self.beta) * p.grad**2)    # typical squared size
                p -= self.lr * p.grad / (s.sqrt() + self.eps)          # divide it out
```

**Canyon: 198 → 13 steps.** In effect, RMSProp undoes the bad scaling of the inputs *inside the optimiser*. That's also why Adam partly hid badly scaled or badly initialised networks earlier in the series (Section 15.6). It's a real advantage, and it's also a reason to still scale your inputs yourself.

---

### 16.4 Adam = momentum + RMSProp

**Adam** (Kingma & Ba, 2014) just does both: a running average of the gradient (momentum, called $m$) and a running average of the squared gradient (RMSProp, called $s$):

$$
m \leftarrow \beta_1 m + (1-\beta_1)\,g \qquad
s \leftarrow \beta_2 s + (1-\beta_2)\,g^2 \qquad
w \leftarrow w - \eta\,\frac{\hat m}{\sqrt{\hat s} + \epsilon}
$$

with the standard settings $\beta_1 = 0.9$, $\beta_2 = 0.999$.

There's one extra detail, the hats. $m$ and $s$ start at **zero**, so for the first few steps they're averages of "a few real gradients plus a lot of zeros", which makes them too small. Adam corrects for that by dividing by how much of the average is real so far:

$$
\hat m = \frac{m}{1 - \beta_1^{\,t}} \qquad \hat s = \frac{s}{1 - \beta_2^{\,t}} \qquad (t = \text{step number})
$$

At $t = 1$ this multiplies $m$ by 10 and $s$ by 1,000, exactly undoing the zero start. By $t \approx 50$ for $m$ (and a few thousand for $s$) the correction has faded to nothing.

```python
class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps, self.t = list(params), lr, b1, b2, eps, 0
        self.m = [torch.zeros_like(p) for p in self.p]
        self.s = [torch.zeros_like(p) for p in self.p]
    def step(self):
        self.t += 1
        with torch.no_grad():
            for p, m, s in zip(self.p, self.m, self.s):
                m.mul_(self.b1).add_((1 - self.b1) * p.grad)        # momentum
                s.mul_(self.b2).add_((1 - self.b2) * p.grad**2)     # RMSProp
                m_hat = m / (1 - self.b1**self.t)                    # bias correction
                s_hat = s / (1 - self.b2**self.t)
                p -= self.lr * m_hat / (s_hat.sqrt() + self.eps)
```

**Check it against PyTorch's own `torch.optim.Adam`** on the MNIST network, same starting weights, same batches: the weights match to $7\times10^{-9}$ after one step and $3\times10^{-7}$ after ten. After a hundred steps they've drifted apart to $5\times10^{-3}$, the same floating-point rounding story as Section 12.3. It's the same algorithm.

**Canyon: 198 → 13 steps.**

![Four optimisers on Section 9's canyon](figures/fig68_optimizers_canyon.png)

| optimiser (each at its best η) | steps to the bottom |
|---|---|
| SGD | 198 |
| Momentum | 16 |
| RMSProp | 13 |
| **Adam** | **13** |

Look at the paths. SGD bounces off the canyon walls. Momentum still swings, but each swing carries it further. RMSProp takes big, direct strides once its scaling kicks in. Adam combines both: smooth like momentum, and scaled like RMSProp.

---

### 16.5 On real data: closer than you'd think

Now MNIST (10,000 digits, 5 epochs), each optimiser given its best learning rate from a small grid:

![Training loss for four optimisers on MNIST](figures/fig70_optimizers_mnist.png)

| optimiser | best η | validation accuracy |
|---|---|---|
| SGD | 0.3 | 94.1% |
| Momentum | 0.03 | 94.5% |
| RMSProp | 0.003 | 94.8% |
| Adam | 0.003 | 94.5% |

On MNIST, with well-scaled inputs (÷255) and a small network, the valley isn't much of a canyon, and all four land within **0.7 points** of each other. Well-tuned SGD is competitive, and many image models are still trained with SGD + momentum.

So why is Adam everyone's default? Look at the "best η" column. SGD's best rate is 100× bigger than Adam's, and each optimiser's best rate changes from problem to problem. Adam's standard rate of **0.001** got 93.9% here without any search, close to its best. Across many different problems, Adam's defaults are usually *good enough*, so you spend less time tuning. That's what it's really good at.

> 📓 **Notebook rule:** *Adam is the safe first choice. SGD + momentum is worth trying when you can afford to tune it.* Whichever you choose, **tune the learning rate first**. It matters more than which optimiser you picked.

---

### 16.6 Changing the learning rate as you go

Section 10.5 showed the problem on the dinner valley: with a constant learning rate, mini-batch noise keeps the weights bouncing around the bottom forever. The fix was to **shrink the steps over time**: big steps early to cover ground, small steps late to settle.

The rule for how η changes over training is called a **learning-rate schedule**. Four common ones:

| schedule | shape |
|---|---|
| **constant** | η the whole way |
| **step decay** | η, then η/10 after 50% of training, then η/100 after 80% |
| **cosine** | glides smoothly from η down to 0 along half a cosine wave |
| **warm-up + cosine** | starts at 0, climbs linearly to η over the first epoch, then cosine down |

```python
def lr_at(t, total, peak, warmup):
    if t < warmup:
        return peak * t / warmup                                            # warm-up: ease in
    progress = (t - warmup) / (total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))                  # cosine: glide down

for pg in opt.param_groups:
    pg["lr"] = lr_at(step, total_steps, peak=0.5, warmup=steps_per_epoch)   # set it every step
```

We tested these on MNIST (50,000 digits, a 784-256-128-10 network, SGD + momentum) at three peak learning rates:

![Learning-rate schedules at peak η = 0.2 and 0.5](figures/fig71_schedules.png)

| peak η | constant | step | cosine | warm-up + cosine |
|---|---|---|---|---|
| 0.05 (comfortable, 20 epochs) | 98.24% | 98.29% | 98.21% | 98.21% |
| 0.2 (aggressive, 10 epochs) | 97.01% | 98.02% | **98.30%** | 98.24% |
| 0.5 (very aggressive, 10 epochs) | **10.99%** | **10.99%** | 51.50% | **97.21%** |

Three different stories:

- **At a comfortable η, schedules barely matter.** All four land within 0.1 points. Honestly, for small problems a constant rate is fine.
- **At an aggressive η, decay pays off.** The constant run bounces around (97.0%, and its accuracy actually *drops* in later epochs). Decaying schedules let it settle into the bottom of the valley, worth about 1.3 points.
- **At a very aggressive η, only warm-up survives.** Constant and step (identical for the first 5 epochs) collapse to 11%, which is guessing. Cosine limps to 51.5%. Warm-up + cosine reaches **97.2%**.

**Why does warm-up rescue it?** At the start, the weights are random and the gradients are large and erratic. A huge first step can shove neurons into the region where ReLU is **always off**, the "dead neuron" from Section 6.5. A dead neuron has zero slope, so it gets no gradient and can never recover. We counted, after one epoch at η = 0.5:

| after 1 epoch at peak η = 0.5 | first-layer neurons that never activate on any validation digit |
|---|---|
| constant η | **89%** |
| warm-up | 7% |

Warm-up lets the network take small, careful steps until it has found a sensible region, and *then* opens up to the big learning rate. Almost every large modern model, Transformers included, trains with warm-up followed by some kind of decay. We'll use exactly this schedule in Part VI.

---

### 📓 Notebook margin: the equation so far

$$
\underbrace{m \leftarrow \beta_1 m + (1-\beta_1) g}_{\text{momentum}}
\quad
\underbrace{s \leftarrow \beta_2 s + (1-\beta_2) g^2}_{\text{RMSProp}}
\quad
w \leftarrow w - \eta_t\,\frac{\hat m}{\sqrt{\hat s}+\epsilon}
\quad
\underbrace{\eta_t = \text{warm-up, then cosine}}_{\text{schedule}}
$$

| idea | what we now know |
|---|---|
| momentum | running sum of gradients: cancels zig-zag (canyon 198 → 16 steps) |
| RMSProp | divide by each weight's typical gradient size: equal steps for every weight (→ 13) |
| Adam | momentum + RMSProp + bias correction; matches `torch.optim.Adam` |
| on MNIST | all four within 0.7 points once tuned; Adam's strength is good defaults |
| schedules | irrelevant at a gentle η; decay helps at an aggressive η |
| warm-up | stops early giant steps killing ReLUs (89% dead vs 7%) |

---

### What comes next

Initialisation (Section 15) gets the signal the right size at step 0. Adam (this section) gives every weight its own step size. But during training, the signals flowing *between* layers keep drifting in scale as the weights change, and every layer has to keep re-adapting to the shifting output of the layer before it.

**Section 17: Keeping Things Balanced** is about **batch normalisation**, which re-standardises the signal at every layer, at every step. It's Section 4's "put everything on the same ruler" idea, moved *inside* the network. We'll see what it does to training speed, why it lets you use much bigger learning rates, and the curious way it behaves differently in training and at test time.

---

*References: Diederik Kingma & Jimmy Ba (2014), "Adam: A Method for Stochastic Optimization" (Adam and its bias correction). Geoffrey Hinton, Coursera lecture 6e (2012) (RMSProp). Ilya Loshchilov & Frank Hutter (2016), "SGDR: Stochastic Gradient Descent with Warm Restarts" (cosine schedules). Priya Goyal et al. (2017), "Accurate, Large Minibatch SGD" (linear warm-up for large learning rates). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 2 (momentum and the "variants of SGD": Adagrad, RMSprop, Adam). Michael Nielsen, *Neural Networks and Deep Learning*, ch. 3 (momentum-based gradient descent; varying the learning rate during training). All code in this series is PyTorch.*
