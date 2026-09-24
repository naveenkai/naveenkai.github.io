# Physics of AI — Part III · Why Depth?

## 19. Any Function At All

> *Here's a claim that sounds too strong to be true: a network with just **one** hidden layer can approximate **any** continuous function (any curve, any surface, any rule mapping inputs to outputs) as closely as you like, as long as it has enough neurons. It's called the **universal approximation theorem**, and it is true. This section proves it the way Nielsen does in his chapter 4, by **building** the network by hand, with no training at all. Then it asks the question the theorem quietly skips: if one layer is enough, why does anyone go deep?*

---

### 19.1 The target

Pick any curve. Here's ours on the interval $[0, 1]$:

$$
f(x) = \sin(3\pi x)\,e^{-x} + 0.5\,x
$$

There's nothing special about it: it rises, falls, rises again, and drifts upward. The goal is to set the weights of a one-hidden-layer network **by hand** so that its output traces this curve. If we can do it for any curve, the theorem is proved (at least in one dimension, and 19.4 handles more).

---

### 19.2 Brick 1: a neuron with a huge weight is a step

Take one sigmoid neuron, $\sigma(w\,(x - s))$. From Section 2.5 we know that as $w$ grows, the S-curve sharpens into the perceptron's step:

![A steep sigmoid is a step; two steps make a bump](figures/fig82_step_bump.png)

| weight $w$ | output just left of the step ($x = 0.38$) | just right of it ($x = 0.42$) |
|---|---|---|
| 10 | 0.450 | 0.550 |
| 100 | 0.119 | 0.881 |
| 1000 | **0.000** | **1.000** |

With $w = 1000$ it's a clean step from 0 to 1, located exactly where we choose, at $x = s$ (equivalently $-b/w$ in Section 2's notation).

**Brick 2: two steps make a bump.** Add a step up at 0.3 (output weight +0.7) and a step up at 0.6 (output weight **−0.7**). Left of 0.3 both are off, so the output is 0. Between 0.3 and 0.6 only the first is on: 0.7. Right of 0.6 both are on and cancel: 0 again. **Two neurons make a rectangular bump**, with a position, width and height we set directly.

---

### 19.3 Stacking bumps

Now cut $[0, 1]$ into $N$ equal slices and put one bump on each slice, with height equal to $f$ at the slice's centre. That's $2N$ hidden neurons and one output neuron adding up the bumps:

```python
def bump_net(N, steep=2000.0):
    edges   = torch.linspace(0, 1, N + 1)
    heights = f((edges[:-1] + edges[1:]) / 2)                 # f at each slice's centre
    hidden, out = nn.Linear(1, 2 * N), nn.Linear(2 * N, 1)
    with torch.no_grad():
        hidden.weight[:] = steep                              # every neuron: a steep step
        hidden.bias[0::2] = -steep * edges[:-1]               # step up at each slice's left edge
        hidden.bias[1::2] = -steep * edges[1:]                # and at its right edge
        out.weight[0, 0::2] =  heights                        # +h at the left edge …
        out.weight[0, 1::2] = -heights                        # … −h at the right: a bump of height h
        out.bias[:] = 0
    return lambda x: out(torch.sigmoid(hidden(x[:, None]))).squeeze(1)
```

(One small detail in the full code: the outermost steps are pushed just past 0 and 1, so the ends of the interval don't get half a step.)

No training happens anywhere. We just **wrote down** the weights:

![Hand-built networks: sigmoid bumps (top) and ReLU kinks (bottom)](figures/fig83_approximations.png)

![Adding bumps one at a time](figures/fig84_bumps.gif)

| bumps | hidden neurons | largest error anywhere | average error |
|---|---|---|---|
| 5 | 10 | 0.782 | 0.185 |
| 10 | 20 | 0.457 | 0.092 |
| 50 | 100 | 0.098 | 0.017 |
| 250 | 500 | **0.020** | **0.001** |

The staircase hugs the curve more tightly with every extra bump, and **it works for any curve**: whatever $f$ is, just read off its heights. Want the error below 0.001? Add more bumps. That's the universal approximation theorem, proved by construction.

> 📓 **Notebook rule:** *one hidden layer can draw any curve, one bump at a time.* Each pair of neurons contributes one rectangle, and enough rectangles make any shape.

**With ReLU it's even easier.** Section 6 showed a ReLU neuron adds one **kink**. Put a kink at each of $N$ evenly spaced points and choose the output weights so each kink changes the slope by exactly the right amount, and the network **connects the dots** of $f$ with straight lines (bottom row above):

| ReLU neurons | 5 | 10 | 50 | 250 |
|---|---|---|---|---|
| largest error | 0.344 | 0.095 | 0.0039 | **0.00016** |

Straight segments follow a smooth curve much better than flat steps do. Five times more neurons cuts the step error by 5× but the ReLU error by about **25×**. That's another quiet reason ReLU took over (Section 6.5).

---

### 19.4 More dimensions: towers, and the catch

A real input isn't one number, it's 784 pixels. Does the construction still work? Yes, and this is where the cost shows up.

In two dimensions a bump becomes a **tower**: a flat-topped block that's high over one little square and zero elsewhere. Build it from a bump in $x_1$ plus a bump in $x_2$ (each is 1 inside its range), then feed the sum into one more steep neuron that fires only when the sum is near 2, i.e. "inside in $x_1$ **and** inside in $x_2$". (That last neuron makes it a two-hidden-layer construction. Nielsen shows how to get the same effect with one layer, but the idea is identical and two is easier to see.)

![One tower, a 2-D target, and 100 towers approximating it](figures/fig85_towers.png)

Tile the square with a grid of towers, each at the target's height, and you get a blocky but recognisable copy of any surface. So it works in 2-D. But count the towers:

| input dimensions $d$ | towers needed for a grid of 10 cells per axis |
|---|---|
| 1 | 10 |
| 2 | 100 |
| 3 | 1,000 |
| 10 | 10,000,000,000 |
| **784** (an MNIST digit) | $\mathbf{10^{784}}$ |

$10^{784}$ is not a big number, it's an absurd one. There are roughly $10^{80}$ atoms in the observable universe. This exponential blow-up with dimension is the **curse of dimensionality**, and it's the fine print on the universal approximation theorem:

> One hidden layer **can** approximate any function, but for general functions of many inputs, the number of neurons it needs can grow **exponentially** with the number of inputs.

So why did our 3-layer network get 99.25% on MNIST in Section 18 with a few hundred thousand weights, not $10^{784}$? Because digits aren't "any function". Their structure (strokes, loops, shapes built from parts) is exactly the kind of thing depth handles efficiently, and a flat grid of towers can't exploit it.

---

### 19.5 Existing isn't the same as learnable

The theorem says good weights **exist**. It doesn't say gradient descent will **find** them. Let's check, on a harder curve with six wiggles of varying height, by actually **training** networks of different shapes and sizes (Adam, cosine schedule, best of three random seeds each):

![Error vs size: hand-built shallow, trained shallow, trained deep](figures/fig86_shallow_vs_deep.png)

| network | weights | test error (MSE) |
|---|---|---|
| **1 hidden layer, weights set by hand** (connect-the-dots) | 385 | **0.000014** |
| 1 hidden layer, **trained** | 385 | 0.035 |
| 1 hidden layer, trained | 1,537 | 0.0053 |
| 4 hidden layers, trained | 505 | 0.0065 |
| 4 hidden layers, trained | 1,873 | **0.00012** |
| 4 hidden layers, trained | 7,201 | 0.00002 |

Three things stand out:

1. **The hand-built shallow network beats every trained network of its size**, and by a lot: with 385 weights, 0.000014 against 0.035, **2,500 times** better. Those weights exist, but training couldn't find them. Gradient descent doesn't know to space the kinks evenly and give each one exactly the right slope change, and a single wide layer gives it little to work with.
2. **Trained deep networks do far better than trained shallow ones at the same size.** At about 1,500–1,900 weights, 4 layers get 0.00012 while 1 layer gets 0.0053, about **44× lower error**. On the right of the figure (385 vs 505 weights), both networks flatten the two small wiggles in the middle, but the deep one tracks the other peaks closely while the shallow one undershoots most of them.
3. **But deep and narrow is less reliable.** The three seeds of the 145-weight deep network scored 0.0018, 0.10 and 0.015, which is almost a 60× spread depending on luck. Narrow deep networks can get stuck (a few dead ReLUs in a 6-wide layer is a big loss). That fragility is exactly what the rest of Part III is about.

> 📓 **Notebook rule:** *the universal approximation theorem guarantees the weights exist. It doesn't say how many neurons you'll need, whether training will find them, or whether they'll generalise.* All three of those questions are answered by depth, data and training, not by the theorem.

---

### 📓 Notebook margin: the equation so far

$$
\hat f(x) = \sum_{i=1}^{N} h_i\,\Big[\sigma\big(w(x - a_i)\big) - \sigma\big(w(x - b_i)\big)\Big]
\;\xrightarrow{\;N \to \infty\;}\; f(x)
\qquad\text{(one hidden layer, any continuous } f\text{)}
$$

| idea | what we now know |
|---|---|
| steep neuron | a step at $x = -b/w$ |
| two steps | a bump; $N$ bumps give a staircase that approaches any curve (max error 0.78 → 0.02) |
| ReLU version | connect-the-dots; error falls like $1/N^2$ (0.34 → 0.00016) |
| more dimensions | towers; a grid needs $10^d$ of them, so $10^{784}$ for a digit |
| the catch | existence ≠ affordable size ≠ findable by training |
| trained, same size | 4 layers beat 1 layer by ~44×, but narrow deep nets are fragile |

---

### What comes next

We now have two facts pulling in opposite directions. **Depth is efficient:** Section 7's folding argument, and 19.5's 44× gap. **Depth is fragile:** vanishing signals (Section 15), dead neurons (Section 16), and seed-to-seed lottery results (19.5). Part II's tools helped, but they don't fully solve it.

**Section 20: When Deeper Gets Worse** runs the experiment that puzzled the field around 2015: add more layers to a network that already trains well, and its **training** error goes *up*. That isn't overfitting; the extra layers simply can't learn to do nothing. The fix is a one-line change, $\mathbf{h} + F(\mathbf{h})$ instead of $F(\mathbf{h})$: the **residual connection**. It's the most important architectural idea between Part I and the Transformer.

---

*References: Michael Nielsen, *Neural Networks and Deep Learning*, ch. 4 ("A visual proof that neural nets can compute any function": steps from steep sigmoids, bumps from pairs of steps, towers in two dimensions, and the caveats). George Cybenko (1989), "Approximation by superpositions of a sigmoidal function". Kurt Hornik (1991), "Approximation capabilities of multilayer feedforward networks". Moshe Leshno, Vladimir Lin, Allan Pinkus & Shimon Schocken (1993) (universal approximation for any non-polynomial activation, including ReLU). All code in this series is PyTorch.*
