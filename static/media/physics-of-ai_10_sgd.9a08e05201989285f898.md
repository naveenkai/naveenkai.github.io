# Physics of AI — Part I · The Gears

## 10. SGD: Learning from a Handful at a Time

> *Section 9 gave us gradient descent: feel the slope, step the other way. But every step needed the gradient over **all** the training examples, which is fine for 400 dinner nights and hopeless for 60,000 digits, let alone the billions of words a language model reads. This section is about taking good steps after looking at only a small random handful of examples, and why that's not a compromise but the reason deep learning works at scale.*

---

### 10.1 What one full step really costs

The loss is an **average** over training examples, so its gradient is an average too:

$$
\nabla C = \frac{1}{n}\sum_{i=1}^{n} \nabla C_i
\qquad \text{(each } \nabla C_i \text{ is the gradient for one example)}
$$

PyTorch confirms it. On the Section 7 MNIST network (218,618 weights), the gradient on all 4,000 training digits matches the average of 4,000 one-digit gradients to within $4 \times 10^{-8}$.

So **one** step of Section 9's gradient descent means running every training example forward and backward through the network. With 4,000 digits that took **20 ms** per step. With 60,000 it's 15× more, and training needs thousands of steps. Each step would read the entire dataset just to move the weights a tiny bit.

---

### 10.2 Tasting the sambar

When you cook a big pot of sambar you don't drink the whole pot to check the salt. You stir it and **taste one spoonful**. A spoonful from a well-stirred pot tells you roughly what the whole pot tastes like.

Same idea here. Stir the training set (shuffle it), take a random spoonful of $B$ examples (a **mini-batch**), and average *their* gradients:

$$
\nabla C \;\approx\; \frac{1}{B}\sum_{i \,\in\, \text{batch}} \nabla C_i
$$

It's an **estimate** and it will be off. But how far off? Let's measure on the MNIST network, partway through training, by comparing many random mini-batch gradients with the true full gradient:

| batch size $B$ | error vs full gradient | cosine with full gradient (1 = same direction) | batches pointing downhill |
|---|---|---|---|
| 1 | 9.38 | 0.10 | 66% |
| 8 | 3.94 | 0.22 | 76% |
| 32 | 2.04 | 0.42 | 100% |
| 128 | 1.02 | 0.68 | 98% |
| 512 | 0.46 | 0.91 | 100% |
| 2,000 | 0.18 | 0.98 | 100% |

![Mini-batch gradient quality vs batch size](figures/fig41_batch_estimate.png)

Two lessons:

- **The error shrinks like $1/\sqrt{B}$.** Four times the batch buys only half the error, which is the usual law of averages. The grey dashed line in the figure is exactly $9.4/\sqrt{B}$, and the measurements follow it.
- **The direction is useful long before the estimate is accurate.** At $B = 32$ the estimate is individually quite noisy (error ≈ 2× its own size), yet **every single one** of the 50 batches we tried pointed downhill. You don't need the exact direction to make progress, just one that's *more down than up*.

That's the whole deal: **one mini-batch step costs a tiny fraction of a full step, and still goes roughly the right way.** Nielsen puts it neatly in his first chapter: with 60,000 training images and mini-batches of 10, each gradient estimate is **6,000× cheaper** to compute than the full one.

---

### 10.3 Epochs, batches and that line in Section 7

Now we can read the training loop from Section 7 properly:

```python
for epoch in range(30):
    for idx in torch.randperm(4000).split(64):     # ← this line
        opt.zero_grad()
        F.cross_entropy(model(Xtr[idx]), ytr[idx]).backward()
        opt.step()
```

- `torch.randperm(4000)` **stirs the pot**: a fresh random order of all training indices.
- `.split(64)` **ladles out spoonfuls**: chunks of 64 indices, which gives 63 batches (the last one has only 32).
- Each batch gives **one step**.
- Going through all the batches once is **one epoch**: every training example seen exactly once.

![One epoch: shuffle, split into batches, one step per batch](figures/fig45_epoch.png)

Why shuffle *every* epoch? If the batches were always the same, the same examples would always be grouped together and the noise would repeat. A fresh shuffle each time makes every step an independent random spoonful.

The vocabulary, pinned down:

| term | meaning | Section 7's values |
|---|---|---|
| **batch size** | examples per step | 64 |
| **step** (iteration) | one gradient estimate plus one weight update | 63 per epoch |
| **epoch** | one full pass through the training set | 30 |
| **SGD** | *stochastic* (random-batch) gradient descent | the whole loop |

---

### 10.4 The race

Enough theory. Here are four training runs on the MNIST network, each reading the training set **20 times** (20 epochs), so every run processes exactly the same 80,000 example-gradients. The only difference is how they're grouped into steps:

```python
for B, lr in [(4000, 0.5), (256, 0.2), (64, 0.1), (8, 0.02)]:
    model = make()                                    # same starting weights every time
    opt = torch.optim.SGD(model.parameters(), lr=lr)  # plain SGD, no tricks
    ...
```

| batch size | steps taken | training loss | test accuracy |
|---|---|---|---|
| 4,000 (full batch) | 20 | 2.074 | **36.3%** |
| 256 | 320 | 0.181 | 92.0% |
| 64 | 1,260 | 0.026 | 93.4% |
| 8 | 10,000 | 0.004 | **94.2%** |

![Same data read, very different progress](figures/fig42_sgd_race.png)

Same data read, and a huge gap. The full-batch run took 20 very careful steps, each using the exact gradient, and barely got started. The mini-batch runs took hundreds or thousands of rough steps and finished the job.

> 📓 **Notebook rule:** *many rough steps beat a few perfect ones.* The progress you make comes from **how many steps** you take. Precision per step has diminishing returns ($1/\sqrt{B}$).

**Being fair to full batch.** Each run got its own learning rate, because the best one depends on batch size. (Full batch at $\eta = 1.0$ fell *apart*, finishing at 8.8%.) And if we give full batch far more steps, it gets there too: 300 full steps reach 94.0%. But count the time:

| run | wall time (CPU) | test accuracy |
|---|---|---|
| B = 64, 20 epochs | **1.6 s** | 93.4% |
| full batch, 100 steps | 2.1 s | 73.5% |
| full batch, 300 steps | 6.2 s | 94.0% |

On only 4,000 digits full batch can catch up if you pay about 4× the time. On 60,000 each full step costs 15× more again, and on the billions of examples behind modern models, full-batch gradient descent simply isn't an option.

**Why not B = 1, then, if smaller batches won the race?** Because computers are built to process many numbers in parallel (Section 5.3). On this CPU a batch of 64 took **1.1 ms** and a batch of 4,000 took **20 ms**: 62× more examples for only 18× more time. Tiny batches waste that parallelism. On GPUs the effect is even bigger, which is why typical batch sizes are **32–512**: small enough for many steps, big enough to keep the hardware busy.

---

### 10.5 What the noise looks like

Back to the dinner valley, where we can see it. Same start and same learning rate, but now each step uses only **4 random nights** out of 400:

![Full batch vs noisy mini-batch paths](figures/fig43_noisy_paths.png)

- **Left, full batch:** a smooth, confident curve to the bottom, costing 400 examples per step.
- **Middle, B = 4:** the path staggers downhill (each spoonful tastes a bit different) and near the bottom it **never settles**. It keeps bouncing around the ★, because every step is still a full-size step with a noisy direction.
- **Right, B = 4 with a shrinking learning rate:** big steps early to cover ground, smaller steps later so the noise averages out. It settles close to the bottom.

That third panel is why real training runs use a **learning-rate schedule**: start large, end small. We'll use one properly in Part II.

**Does the noise ever *help*?** Possibly. Real loss landscapes (Section 9.7) have flat plateaus and saddle points where the exact gradient is close to zero. A noisy step can jostle the weights off those. There's also evidence that SGD's noise steers toward *wide* valleys, which tend to generalise better than narrow ones. These are active research ideas, not settled facts, so hold them loosely.

---

### 10.6 One upgrade: momentum

Section 9's canyon (PG food on a 0–100 scale) had gradient descent zig-zagging across the steep walls while crawling along the gentle floor. SGD's noise adds more zig-zag of its own. One classic fix is **momentum**, and it's the ball-rolling picture from Section 9 made literal.

A real ball doesn't only respond to the slope under it right now. It **keeps its velocity**. Momentum does the same: each step is the current gradient *plus a fraction of the previous step*:

$$
\mathbf{v} \leftarrow \mu\,\mathbf{v} - \eta\,\nabla C
\qquad\qquad
\mathbf{w} \leftarrow \mathbf{w} + \mathbf{v}
\qquad(\mu \approx 0.9)
$$

- **Across the canyon** the gradient flips sign every step, so those contributions **cancel out** in $\mathbf{v}$.
- **Along the canyon floor** the gradient points the same way every step, so those contributions **add up**, and the ball picks up speed.

```python
opt = torch.optim.SGD([w], lr=0.02, momentum=0.9)   # one extra argument
```

| on Section 9's canyon, $\eta = 0.02$ | steps to reach the bottom |
|---|---|
| plain gradient descent | 283 |
| with momentum 0.9 | **19** |

(Both at the same $\eta = 0.02$. Give momentum its own tuned learning rate and it needs just 16 steps; Section 16 compares every optimiser that way.)

![Momentum in the canyon](figures/fig44_momentum.png)

About **15× faster** from one extra argument. It still swings side to side, but each swing carries it further down the canyon.

This also explains the `torch.optim.Adam` in our Section 6 and 7 black boxes. **Adam** is SGD plus momentum plus a separate, automatically tuned step size for **every** weight, so the steep and gentle directions each get their own learning rate. That's a direct fix for the canyon problem. We'll build it up properly in Part II.

---

### 📓 Notebook margin: the equation so far

$$
\underbrace{\mathcal{B} \sim \text{random } B \text{ examples}}_{\text{a spoonful}}
\qquad
\mathbf{g} = \frac{1}{B}\sum_{i\in\mathcal{B}} \nabla C_i
\qquad
\mathbf{v} \leftarrow \mu\mathbf{v} - \eta\,\mathbf{g}, \;\;\; \mathbf{w} \leftarrow \mathbf{w} + \mathbf{v}
$$

| idea | what we now know |
|---|---|
| full gradient | an average over *all* examples: exact, but one step = one full pass over the data |
| mini-batch gradient | an average over a random handful: error ~ $1/\sqrt{B}$, direction good early |
| epoch | shuffle, split into batches, one step per batch |
| the race (same data read) | full batch 36.3% vs B = 8 **94.2%** |
| batch size in practice | 32–512: many steps, but still enough parallel work for the hardware |
| noise | needs a shrinking learning rate to settle; may help escape flat spots |
| momentum | cancels the zig-zag, accumulates along the floor: 283 → 19 steps |

---

### What comes next

One piece of the training loop is still a black box: `loss.backward()`. Every section since 6 has leaned on it to produce 218,618 slopes, one for every weight, in about a millisecond. Computing each one separately with the "nudge and see" trick from Section 9.2 would mean 218,618 extra forward passes for **every** step.

**Section 11: The Chain Rule Is Backprop** opens that last box. We'll trace a tiny network by hand, see how one backward sweep produces every gradient at once, and then check our hand-derived numbers against PyTorch's.

---

*References: Michael Nielsen, *Neural Networks and Deep Learning*, ch. 1 (stochastic gradient descent and mini-batches; the mini-batch gradient as an estimate of the full gradient; epochs; the speed-up from small mini-batches). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 2 ("Stochastic gradient descent": mini-batch SGD, true SGD vs batch GD, momentum as a ball rolling down the loss curve). All code in this series is PyTorch.*
