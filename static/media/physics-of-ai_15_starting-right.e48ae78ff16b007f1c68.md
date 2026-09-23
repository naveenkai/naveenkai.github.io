# Physics of AI — Part II · Making It Learn

## 15. Starting Right

> *Every network in this series has started life with random weights, and we've never asked **how** random. For the shallow networks of Part I it barely mattered. But in Section 11.8 a 10-layer network's learning signal shrank dramatically on its way back to the first layer, and that was only partly the activation function's fault. This section shows that the **scale** of the starting weights decides whether a deep network can learn at all, and derives the simple formula that gets it right.*

---

### 15.1 Why not start at zero?

Zero seems like the neutral, fair place to start. Try it, on a small 784 → 64 → 10 network trained for 5 epochs:

| starting weights | distinct hidden neurons after training | validation accuracy |
|---|---|---|
| all exactly **0** | 1 of 64 (none moved at all) | **11.0%** |
| all exactly **0.01** | **1 of 64** | 28.0% |
| **random** (PyTorch default) | 64 of 64 | **90.9%** |

With all zeros, nothing moves. Every hidden neuron outputs ReLU(0) = 0, so by Section 11's backprop rules every gradient reaching the first layer is zero too.

The 0.01 case is more interesting. The network **does** train, but look at what its 64 hidden neurons became:

![Constant start: 64 identical neurons. Random start: 64 different detectors.](figures/fig64_symmetry.png)

**They're all the same neuron.** If every neuron starts with the same weights, every neuron computes the same output, gets the same gradient (Section 11's rule 2 treats them identically), takes the same step, and stays identical, forever. You paid for 64 neurons and got one, repeated 64 times.

> 📓 **Notebook rule:** *randomness at the start is how neurons become different.* It's called **breaking symmetry**. Without it, a layer of $n$ neurons is really a layer of one.

So the weights must start random. The question is: random **with what size**?

---

### 15.2 The telephone game

Remember the telephone game? A message is whispered down a line of people. If everyone repeats it a little more quietly than they heard it, by the twentieth person there's silence. If everyone repeats it a little louder, by the end it's distorted shouting. It only survives if everyone speaks at **the same volume** they heard.

A deep network is that line. Each layer receives a signal (the activations from the layer before), multiplies it by weights, adds up, applies ReLU, and passes it on. If each layer makes the signal a bit smaller, it vanishes. If each makes it a bit bigger, it explodes.

Here's the experiment: 20 ReLU layers, each 256 wide, **no training at all**. Push 512 digits through and measure, at each layer, how much the signal **differs between digits**. If that goes to zero, every digit looks the same to the later layers, and there's nothing left to learn from.

![Signal strength through 20 layers for five starting scales](figures/fig65_signal_depth.png)

| starting weights ($w \sim \mathcal{N}(0, \sigma^2)$) | layer 1 | layer 10 | layer 20 |
|---|---|---|---|
| σ = 0.01 | 0.046 | $9\times10^{-11}$ | $3\times10^{-20}$ |
| σ = 0.05 | 0.23 | $9\times10^{-4}$ | $3\times10^{-6}$ |
| PyTorch default | 0.093 | $2\times10^{-5}$ | $6\times10^{-9}$ |
| **σ = √(2/256) ≈ 0.088** | **0.23** | **0.15** | **0.14** |
| σ = 0.15 | 0.70 | 52 | **9,670** |

Every scale except one sends the signal off a cliff, either toward zero or toward infinity, and it does so **exponentially**: the same factor applied twenty times over. At layer 20 of the PyTorch default network, the signals for different digits differ by six *billionths*. The layer effectively sees the same input no matter which digit went in.

Only σ ≈ 0.088 keeps the signal steady from layer 1 to layer 20. Where does that number come from?

---

### 15.3 Deriving the right size

Take one neuron in some layer. Its input is $z = \sum_{i=1}^{n} w_i h_i$, a sum of $n$ terms, where $n$ is the number of inputs, called the **fan-in**. Assume the weights are random with average 0 and variance $\sigma^2$, independent of the inputs. Then the variances add up:

$$
\text{Var}(z) \;=\; n \cdot \sigma^2 \cdot \mathbb{E}[h^2]
$$

Each input $h$ came out of a ReLU. ReLU zeroes the negative half of a symmetric signal, which halves its average square: $\mathbb{E}[h^2] = \tfrac{1}{2}\text{Var}(z_{\text{previous}})$. So from one layer to the next:

$$
\text{Var}(z_{\text{next}}) \;=\; \underbrace{\frac{n\,\sigma^2}{2}}_{\text{the per-layer factor}} \cdot\, \text{Var}(z_{\text{previous}})
$$

That factor is multiplied in at **every** layer. Anything below 1 shrinks the signal exponentially, and anything above 1 grows it exponentially. To keep it steady, set the factor to exactly 1:

$$
\boxed{\;\sigma^2 = \frac{2}{n} \qquad\text{i.e.}\qquad \sigma = \sqrt{\frac{2}{\text{fan-in}}}\;}
$$

This is **He initialisation** (Kaiming He and colleagues, 2015). For our 256-wide layers, $\sigma = \sqrt{2/256} = 0.088$, exactly the one line that stayed flat. Plug in the other scales and the table above follows. For example, σ = 0.05 gives a per-layer factor (in standard deviation) of $\sqrt{256 \times 0.0025 / 2} = 0.57$, and $0.57^{19} \approx 10^{-5}$, which matches the measured drop.

The "2" is there because of ReLU. For activations that don't cut the signal in half (like tanh, or no activation), the same argument gives $\sigma^2 = 1/n$. That's the older **Xavier / Glorot** initialisation (Glorot & Bengio, 2010), designed for the sigmoid-and-tanh era.

> 📓 **Notebook rule:** *scale each layer's starting weights by $1/\sqrt{\text{fan-in}}$*, times $\sqrt{2}$ for ReLU. More inputs means each one must speak more quietly, so their sum stays the same size.

(The same argument run **backwards** gives the same condition for the *gradients* during backprop, so He initialisation keeps both the forward signal and the backward learning signal steady.)

---

### 15.4 So what's wrong with PyTorch's default?

`nn.Linear` doesn't use He initialisation. It draws weights uniformly from $\pm 1/\sqrt{\text{fan-in}}$, which has variance $\sigma^2 = \frac{1}{3n}$. Plug that into the per-layer factor:

$$
\frac{n}{2}\cdot\frac{1}{3n} = \frac{1}{6}
\qquad\Rightarrow\qquad \text{the signal's size shrinks by } \sqrt{1/6} \approx 0.41 \text{ at every layer}
$$

Over 3 layers (Section 12's network) that's a factor of about 0.07: noticeable, but the network trains fine. Over 20 layers it's $0.41^{19} \approx 10^{-7}$, and the signal is gone. The default works fine for **shallow** networks. For deep ReLU networks you set the scale yourself:

```python
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")   # N(0, 2/fan_in): He
        nn.init.zeros_(m.bias)
```

This also explains most of **Section 11.8**, where a 10-layer ReLU network with default weights delivered a gradient 55–70× smaller to its first layer than to its last (the exact number depends on the batch). Rerun that network with He initialisation:

![Gradient size per layer: default vs He](figures/fig66_grad_layers.png)

| 10 ReLU layers | first-layer gradient ÷ last-layer gradient |
|---|---|
| PyTorch default | 0.014 (≈ 70× smaller at the front) |
| **He** | **2.2** (all layers within a small factor of each other) |

Every layer now gets a learning signal of roughly the same size.

---

### 15.5 Does it matter for training? Completely.

Here's the real test: train a **20-layer** ReLU network (256 wide) on 10,000 digits, with the same SGD settings each time (learning rate 0.01, momentum 0.9, 5 epochs). The only difference is the starting scale:

![Training a 20-layer network from four different starts](figures/fig67_deep_training.png)

| start | what happened | validation accuracy |
|---|---|---|
| σ = 0.01 | loss stuck at 2.30 | 11.0% |
| PyTorch default | loss stuck at 2.30 | 11.0% |
| **He** | **trains normally** | **92.3%** |
| σ = 0.15 | first loss **88,368**, then NaN | 10.6% |

A loss of 2.30 is $-\log(0.1)$: the network is outputting 10% for every digit, which is pure guessing (Section 8.5). With the signal gone by layer 20, the output can't depend on the input, so guessing is all it can do. With σ = 0.15 the numbers overflowed floating point within a single step. Only He initialisation gives a network that learns, and the **only** difference is one number used at the start.

This is a big part of why deep networks were considered "hard to train" for so long. Their structure was fine. They were simply being started at the wrong scale.

---

### 15.6 Honest footnotes

- **Initialisation is only the start.** It gets the signal right at step 0, but as training changes the weights the signal can drift again. The next line of defence is **normalisation layers** (Section 17), which re-balance the signal at every layer, at every step. The most powerful fix, **residual connections**, gives the signal a direct path around each layer, and that's Part III's big idea. With both in place, modern networks are much less sensitive to initialisation.
- **Adam can hide the problem, a little.** Because Adam rescales each weight's step (Section 10.6), it sometimes rescues a badly started network that plain SGD can't. We used SGD here so you could see the effect honestly.
- **Biases can start at zero.** Symmetry is already broken by the random weights.

---

### 📓 Notebook margin: the equation so far

$$
w \sim \mathcal{N}\!\left(0,\; \frac{2}{\text{fan-in}}\right),
\qquad b = 0
\qquad\Longleftarrow\qquad
\text{Var}(z_{\text{next}}) = \frac{n\sigma^2}{2}\,\text{Var}(z_{\text{prev}}) \text{ with the factor set to } 1
$$

| idea | what we now know |
|---|---|
| zero / constant start | every neuron stays identical: 64 neurons become 1 (28.0% vs 90.9%) |
| breaking symmetry | random starts make neurons different |
| signal through depth | multiplied by $n\sigma^2/2$ per layer: vanishes or explodes **exponentially** |
| He initialisation | $\sigma = \sqrt{2/\text{fan-in}}$: steady signal through 20 layers |
| PyTorch default | shrinks by 0.41 per layer: fine for 3 layers, fatal for 20 |
| 20-layer network | default: 11.0% (guessing) · He: **92.3%** |

---

### What comes next

We've been leaning on Adam since Section 6, and we've described it in one line: "momentum plus a separate step size for every weight". **Section 16: Better Steps** builds it properly, from plain SGD to momentum to RMSProp to Adam, one idea at a time, each tested on the same canyon from Section 9. Then it covers **learning-rate schedules** (warm-up, decay, cosine), the reason nearly every modern training run changes its learning rate as it goes.

---

*References: Kaiming He, Xiangyu Zhang, Shaoqing Ren & Jian Sun (2015), "Delving Deep into Rectifiers" (He initialisation for ReLU networks). Xavier Glorot & Yoshua Bengio (2010), "Understanding the difficulty of training deep feedforward neural networks" (Xavier/Glorot initialisation, and the signal-propagation argument). Michael Nielsen, *Neural Networks and Deep Learning*, ch. 3 ("Weight initialization": scaling Gaussian weights by $1/\sqrt{n_{\text{in}}}$ to avoid saturation) and ch. 5 (unstable gradients in deep networks). All code in this series is PyTorch.*
