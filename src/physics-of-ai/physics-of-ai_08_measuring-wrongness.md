# Physics of AI — Part I · The Gears

## 8. Measuring Wrongness

> *Part I so far has been about what a network **is**: numbers in, moves and folds, a straight cut at the end. Every time we trained something, we hid the training inside a black box with three mysterious lines: a loss, `backward()`, `step()`. From here on we open that box. The first thing a network needs before it can improve is a way to know **how wrong it is**, as a single number it can push down.*

---

### 8.1 Why not just use accuracy?

Accuracy is the number we actually care about, so it seems like the obvious scoreboard. Let's test it with the two-input movie neuron from Section 4 (reviews and cast, standardised).

Take a reviews weight that isn't the best yet, $w_1 = 2$, and nudge it:

| reviews weight $w_1$ | accuracy | cross-entropy loss (defined below) |
|---|---|---|
| 2.00 | 83.75% | 0.3744 |
| 2.01 | 83.75% | 0.3728 |
| 2.05 | 83.75% | 0.3666 |

Accuracy **doesn't move**. The nudge made every prediction slightly better, but none crossed the 50% line, so the count of right answers stays the same. The loss, meanwhile, notices every improvement.

Sweep $w_1$ across its whole range and the difference is stark:

![Accuracy is a staircase, loss is a slope](figures/fig31_accuracy_vs_loss.png)

Zoom into the accuracy curve and it's a **staircase**: flat, then a sudden jump when one night flips from wrong to right. That's the perceptron's cliff from Section 2.4 all over again. On the flat steps the slope is zero, so there's no signal about which way to go, and at the jumps the slope is undefined.

> 📓 **Notebook rule:** *we judge the model by accuracy, but we **train** it on something smooth.* The smooth thing is the **loss**.

What we need from a loss:

1. **One number** for how wrong the model is across all the examples.
2. **Smooth**, so every tiny improvement shows up (a slope we can follow).
3. **Punishes the right things**, especially confident mistakes.

---

### 8.2 First attempt: squared error

The obvious choice, and the one Nielsen starts with in his first chapter, is the average squared distance between prediction and target:

$$
C_{\text{MSE}} = \frac{1}{2n}\sum_{i=1}^{n} \big(a_i - y_i\big)^2
\qquad a_i = \sigma(z_i) \text{ is the neuron's output, } y_i \in \{0, 1\}
$$

It's smooth, it's zero when you're perfect, and it grows as you get worse. It passes tests 1 and 2.

Test 3 is where it fails. Suppose tonight you **didn't** watch ($y = 0$) and the neuron said:

| neuron's output $a$ | how wrong is that? | squared error ½(a−0)² |
|---|---|---|
| 0.5 | unsure | 0.125 |
| 0.9 | fairly wrong | 0.405 |
| 0.99 | **confidently** wrong | 0.490 |
| 0.9999 | **absurdly** confident and wrong | 0.49990 |

Squared error **can never exceed 0.5**. Being 99.99% sure of the wrong answer costs barely more than being 90% sure. A loss that shrugs at confident mistakes is a bad teacher.

---

### 8.3 The learning slowdown

It gets worse. To learn, we follow the slope of the loss (Section 9 will make this precise). So what matters is the **slope of the loss with respect to $z$**, the neuron's raw score. By the chain rule:

$$
\frac{\partial C_{\text{MSE}}}{\partial z} = (a - y)\,\cdot\,\underbrace{\sigma'(z)}_{\text{the sigmoid's slope}}
$$

There's our old enemy, the sigmoid's slope. When the neuron is confidently wrong, $z$ is far out on the flat part of the S-curve, $\sigma'(z) \approx 0$, and **the gradient vanishes exactly when the mistake is worst.**

Here's that as an experiment. One neuron, one input, the target is 0 ("didn't watch"), trained step by step from two different starting points:

```python
def run(loss_name, w0, b0, steps=300, lr=0.15):
    w = torch.tensor(w0, requires_grad=True)
    b = torch.tensor(b0, requires_grad=True)
    x, y = torch.tensor(1.), torch.tensor(0.)
    for _ in range(steps):
        a = torch.sigmoid(w*x + b)
        loss = 0.5*(a - y)**2 if loss_name == "mse" else F.binary_cross_entropy(a, y)
        w.grad = b.grad = None
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad; b -= lr * b.grad
```

![Squared error gets stuck when the neuron is confidently wrong](figures/fig33_learning_slowdown.png)

- **Mildly wrong start** (82%): both losses learn, and squared error is just slower.
- **Confidently wrong start** (98%): squared error sits **almost motionless for over 150 steps**, still around 0.9 after 150 steps. It's stuck on the flat shoulder of the sigmoid. The orange curve on the right should look backwards to you: the neuron learns **slowest** when it is **most** wrong.

That's the opposite of how you'd want a student to learn. (Nielsen's chapter 3 runs a similar experiment. It's one of the clearest demonstrations in his book.)

---

### 8.4 Cross-entropy: measuring surprise

Here's a loss built on a different idea. Instead of asking *"how far is the output from the target?"*, ask:

> *How **surprised** should the model be by what actually happened?*

If the model gave probability $p$ to the answer that turned out to be true, its surprise is $-\log p$:

| probability given to the true answer | surprise $-\log p$ |
|---|---|
| 0.99 | 0.01 (no surprise) |
| 0.9 | 0.11 |
| 0.5 | 0.69 |
| 0.1 | 2.30 |
| 0.01 | **4.61** |
| → 0 | → ∞ |

For our yes/no neuron, the probability of the true answer is $a$ when $y = 1$ and $1 - a$ when $y = 0$. Both cases fit in one formula:

$$
C_{\text{CE}} = -\frac{1}{n}\sum_{i=1}^{n} \Big[\, y_i \log a_i + (1 - y_i)\log(1 - a_i) \,\Big]
$$

That's **binary cross-entropy**. Compare the two losses head to head:

![Squared error vs cross-entropy as a function of confidence](figures/fig32_loss_vs_p.png)

As confidence in the wrong answer grows, cross-entropy **explodes** toward infinity. Confident mistakes are now expensive, which passes test 3.

**And the slowdown disappears.** Take the slope with respect to $z$ again:

$$
\frac{\partial C_{\text{CE}}}{\partial z} = a - y
$$

The $\sigma'(z)$ term **cancelled out**. The $\log$ in the loss exactly undoes the flattening of the sigmoid. The gradient is now simply *"how wrong you are"*, so a bigger error gives a bigger push:

| $z$ | output $a$ (target 0) | slope, squared error | slope, cross-entropy |
|---|---|---|---|
| 0 | 0.500 | 0.125 | 0.500 |
| 2 | 0.881 | 0.093 | 0.881 |
| 4 | 0.982 | **0.017** | **0.982** |
| 6 | 0.998 | **0.003** | **0.998** |

Read the last two rows: the more wrong the neuron is, the *weaker* squared error pushes, and the *stronger* cross-entropy pushes. That's the blue curve in the slowdown figure diving immediately.

> 📓 **Notebook rule:** *sigmoid output → cross-entropy loss.* The log was designed to cancel the flat parts of the sigmoid, and the two belong together.

(The one-line derivation, if you want it: with $a = \sigma(z)$ and $\sigma'(z) = a(1-a)$, the chain rule gives $\frac{\partial C}{\partial z} = \left(-\frac{y}{a} + \frac{1-y}{1-a}\right)a(1-a) = -y(1-a) + (1-y)a = a - y$.)

---

### 8.5 Ten classes: softmax

Digits need 10 answers, not one yes/no. The last layer of the Section 7 network outputs **10 raw scores**, one per digit, called **logits**. They can be any real numbers, so they aren't probabilities yet.

**Softmax** turns scores into probabilities in two steps. Exponentiate (so everything is positive and bigger scores grow much bigger), then divide by the total (so they sum to 1):

$$
p_k = \frac{e^{s_k}}{\sum_{j=1}^{10} e^{s_j}}
$$

The loss is the same surprise idea: **$-\log$ of the probability given to the correct digit.**

$$
C_{\text{CE}} = -\log p_{\text{true class}}
$$

Here it is on two real test digits from our trained network:

![Softmax on a confident right answer and a confident wrong answer](figures/fig34_softmax_digits.png)

**Top, a 7.** The 7 scores 9.91, the next best (a 3) scores 5.31. That gap of about 4.6 becomes, after softmax, **98.9%** for 7 and 1% for 3. Loss = $-\log 0.989 = $ **0.01**. Almost no surprise.

**Bottom, a 4 written like a 9** (the pair from Section 7.4). The network gives **99.8%** to "9" and just **0.2%** to the true answer, "4". Loss = $-\log 0.002 = $ **6.21**, over 500 times the loss on the 7. This one example tells the network loudly *"you were sure, and you were wrong"*.

Averaged over all 1,000 test digits, the loss is **0.33**. A network guessing uniformly (10% each) would score $-\log 0.1 = 2.30$, which is a useful baseline to remember when you watch a training run start.

In PyTorch you almost never apply softmax yourself before the loss:

```python
logits = model(x)                          # raw scores, shape (batch, 10)
loss = F.cross_entropy(logits, labels)     # softmax + (−log p) inside, done safely
```

---

### 8.6 Why PyTorch wants the raw scores

Why pass **logits** and not probabilities? Floating-point numbers run out of room:

```python
z = torch.tensor([20.])
a = torch.sigmoid(z)
a.item(), (1 - a).item()          # → 1.0, 0.0      (float32 rounds 0.999999998 to 1)
torch.log(1 - a)                  # → -inf          the loss is infinite, training breaks

F.binary_cross_entropy(a, torch.tensor([0.]))              # → 100.0  (PyTorch clamps it)
F.binary_cross_entropy_with_logits(z, torch.tensor([0.]))  # → 20.0   the true answer
```

Given the raw score, PyTorch can rearrange the maths so it never computes $\log(0)$, and you get the exact answer. **Rule: give the loss function logits**. Use `BCEWithLogitsLoss` for yes/no and `cross_entropy` for many classes. That's what the black-box loop in Sections 6 and 7 was doing all along.

---

### 8.7 The loss is a landscape

One last change of view that sets up everything next.

For fixed data, the loss depends **only on the weights**. So for the two-weight movie neuron we can compute the loss at every possible $(w_1, w_2)$ and draw it as a map:

![Accuracy vs loss over the weight plane](figures/fig35_landscape.png)

- **Left, accuracy.** It only depends on the *direction* of $\mathbf{w}$ (the angle of the decision line), so it fans out from the origin in wedges. Walk outward along any ray and it doesn't change at all. There's no slope to follow.
- **Right, cross-entropy.** A smooth **valley** with a single lowest point (★). From *anywhere* on this map, "which way is downhill?" has an answer.

**Training is finding the bottom of this valley.** Our movie neuron has 3 weights, so its valley lives in 3-D. The Section 7 MNIST network has about 220,000 weights, so its valley lives in 220,000 dimensions. We can't draw that, but the question at every step is the same one we can see here: *which way is down?*

---

### 📓 Notebook margin: the equation so far

$$
\underbrace{\hat{\mathbf{p}} = \text{softmax}\big(W_{\text{out}}\,\mathbf{h}_L + \mathbf{b}_{\text{out}}\big)}_{\text{the network (Sections 2–7)}}
\qquad\qquad
\underbrace{C = -\log \hat{p}_{\text{true}}}_{\text{how wrong it is}}
$$

| idea | what we now know |
|---|---|
| accuracy | what we *report*; a staircase with no slope, so we can't train on it |
| loss | one smooth number we *train* on: $C(\text{all the weights})$ |
| squared error | caps at 0.5 and slows down exactly when most wrong |
| cross-entropy | "surprise" $-\log p$; its gradient is simply $a - y$ |
| softmax | scores → probabilities that sum to 1 |
| logits into the loss | numerically safe (`cross_entropy`, `BCEWithLogitsLoss`) |
| training | finding the bottom of the loss landscape |

---

### What comes next

We have a landscape and we want its lowest point. With 220,000 dimensions we can't look at the whole map. All we can feel is the ground under our feet: *which way slopes down, right here?*

**Section 9: Rolling Downhill** is gradient descent. We'll build it from the derivative up, watch it walk down the movie valley step by step, and break it on purpose with learning rates that are too small, too big and just right.

---

*References: Michael Nielsen, *Neural Networks and Deep Learning*, ch. 1 (quadratic cost) and ch. 3 (the learning slowdown with quadratic cost; the cross-entropy cost and why its gradient is $a - y$; softmax). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 2 (loss function as the feedback signal; `sparse_categorical_crossentropy`) and ch. 4 (choosing the loss for binary and multiclass classification). All code in this series is PyTorch.*
