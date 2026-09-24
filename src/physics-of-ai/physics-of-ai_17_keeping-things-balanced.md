# Physics of AI — Part II · Making It Learn

## 17. Keeping Things Balanced

> *Section 4 taught us to put every input on the same ruler before training: subtract the mean, divide by the spread. Section 15 made sure the signal starts the right size at every layer. But as training changes the weights, the signals between layers keep drifting, and each layer has to chase a moving target set by the layer before it. **Batch normalisation** applies Section 4's fix to every layer, at every step. It's one of the most useful tricks in deep learning, and it has one strange side effect you need to know about.*

---

### 17.1 The idea: standardise inside the network

Take one neuron in a hidden layer. Over a mini-batch of 64 digits, its pre-activation values $z$ form a little distribution with some mean and some spread. Batch norm does three things to it:

1. **Measure** the mean $\mu_B$ and variance $\sigma_B^2$ of $z$ over this batch.
2. **Standardise**, exactly as in Section 4: $\hat z = \dfrac{z - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
3. **Rescale and shift** with two *learnable* numbers per neuron: $y = \gamma\,\hat z + \beta$

![One neuron's values over a batch: raw, standardised, then scaled and shifted](figures/fig72_bn_mechanics.png)

Step 3 might look like it undoes the whole thing. Why standardise and then immediately rescale? Because forcing every neuron to mean 0 and spread 1 **forever** would limit the network. Some neurons may work best mostly-on or mostly-off. $\gamma$ and $\beta$ let the network choose each neuron's scale and centre **deliberately, through gradient descent**, instead of having them drift by accident as earlier weights change. (If it wanted, the network could even set $\gamma = \sigma_B$ and $\beta = \mu_B$ and undo the normalisation completely.)

Here it is from scratch, checked against PyTorch's `nn.BatchNorm1d`:

```python
class MyBatchNorm(nn.Module):
    def __init__(self, n, momentum=0.1, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n))            # learnable scale
        self.beta  = nn.Parameter(torch.zeros(n))           # learnable shift
        self.register_buffer("run_mean", torch.zeros(n))    # remembered for test time (17.5)
        self.register_buffer("run_var",  torch.ones(n))
        self.m, self.eps = momentum, eps

    def forward(self, z):
        if self.training:
            mean, var = z.mean(0), z.var(0, unbiased=False)          # THIS batch's statistics
            with torch.no_grad():                                    # keep a running average
                self.run_mean.mul_(1 - self.m).add_(self.m * mean)
                self.run_var.mul_(1 - self.m).add_(self.m * z.var(0, unbiased=True))
        else:
            mean, var = self.run_mean, self.run_var                  # remembered statistics
        z_hat = (z - mean) / torch.sqrt(var + self.eps)              # Section 4, inside the network
        return self.gamma * z_hat + self.beta
```

It matches `nn.BatchNorm1d` to $7\times10^{-7}$ in training mode and $2\times10^{-6}$ in evaluation mode. We'll come back to those two modes in 17.5, because that's where the strange side effect lives.

**Where it goes:** usually between the linear layer and the activation, i.e. `Linear → BatchNorm → ReLU`. (The linear layer's bias then becomes pointless, since batch norm subtracts the mean anyway. That's why you'll often see `nn.Linear(..., bias=False)` before a batch norm.)

---

### 17.2 What it looks like during training

Here's what the second hidden layer's neurons actually receive during the first 300 training steps, without and with batch norm (six neurons shown; line = batch mean, shaded band = ±1 spread):

![Layer 2's inputs drift without batch norm and stay put with it](figures/fig73_bn_drift.png)

Without batch norm, the ground keeps shifting. As layer 1's weights change, the values arriving at layer 2 swing around. One neuron's average jumps to +1.6 and back within 30 steps, and another drops to −1. Layer 2 is trying to learn on top of something that won't sit still.

With batch norm, every neuron's input stays centred with a spread of about 1, by construction, and moves only as fast as the network deliberately changes $\gamma$ and $\beta$.

The original batch norm paper (Ioffe & Szegedy, 2015) called this drifting **internal covariate shift** and argued that removing it is why batch norm helps. Later work (Santurkar et al., 2018) found a different explanation fits the evidence better: batch norm makes the **loss landscape smoother**, so gradients are more reliable and bigger steps are safe. The exact *why* is still debated. What it *does* is not, so let's measure that.

---

### 17.3 It rescues deep networks

Remember Section 15.5? A 20-layer ReLU network with PyTorch's default starting weights was stuck at 11%, pure guessing, because the signal vanished long before the last layer. He initialisation fixed it (92.3%). Now keep the **bad** default start, and just add a batch norm after every layer:

![20-layer network with default init: without vs with batch norm](figures/fig74_bn_deep.png)

| 20 ReLU layers, default init, same SGD | validation accuracy |
|---|---|
| no batch norm | 11.0% |
| **+ batch norm after every layer** | **94.5%** |

Batch norm doesn't care how badly the weights started. However small the signal gets, the next batch norm scales it straight back to spread 1. That's the telephone game from Section 15.2 with someone standing between every pair of players, resetting the volume.

---

### 17.4 It tolerates bigger learning rates

The second practical win. Same 3-layer network, same 5 epochs, learning rate turned up and up:

![Validation accuracy vs learning rate, with and without batch norm](figures/fig75_bn_lr.png)

| η | no batch norm | with batch norm |
|---|---|---|
| 0.01 | 92.1% | **95.5%** |
| 0.05 | 95.1% | **96.1%** |
| 0.2 | 91.3% | **95.4%** |
| 0.5 | 9.5% (dead) | **94.8%** |
| 1.0 | 11.0% (dead) | **88.8%** |
| 2.0 | 10.6% | 9.6% |

Without batch norm, the network dies somewhere between η = 0.2 and 0.5 (the dead-ReLU collapse from Section 16.6). With batch norm it still trains well at **0.5**, and hangs on at 1.0. It's not magic, since η = 2.0 kills both, but the safe range is several times wider. And it's **better at every learning rate that works**, including the gentle ones.

Why? If a big step makes some layer's outputs much larger, the next batch norm simply divides that back down. A single overshoot can't snowball through the rest of the network.

> 📓 **Notebook rule:** *batch norm is Section 4's standardisation, applied at every layer, at every step.* It rescues bad initialisations, widens the range of learning rates that work, and usually trains faster.

---

### 17.5 The strange side effect: a digit's answer depends on its batch-mates

Look again at step 1 of 17.1: batch norm uses the **batch's** mean and spread. So during training, how a digit gets normalised depends on **which other digits happen to be in the same batch**.

That's fine during training, where batches are random and the statistics average out. But at test time you might score one image at a time, or a batch that's all the same class. So batch norm has **two modes**:

| mode | normalises with | used when |
|---|---|---|
| `model.train()` | the current batch's mean and variance | training |
| `model.eval()` | a **running average** remembered during training | testing, deployment |

Forgetting to switch modes is one of the most common bugs in deep learning. Here's how bad it can get, on one trained batch-norm network:

![One network, scored three ways](figures/fig76_bn_modes.png)

| how we scored the 10,000 validation digits | accuracy |
|---|---|
| `eval()` mode | **96.1%** |
| `train()` mode, random batches of 100 | 95.5% |
| `train()` mode, batches of 100 that are **all the same digit** | **17.0%** |

In that last row, every batch contains, say, only sevens. Batch norm computes "the average seven" and subtracts it, which removes exactly the features that make them sevens. The network is left looking at the *differences between* sevens.

Even more striking, here's **one** particular seven and the network's probability that it's a 7:

| the same seven, scored… | P(7) |
|---|---|
| in `train()` mode, in a batch with 63 **other sevens** | **0.001** |
| in `train()` mode, in a batch of 63 **mixed** digits | 1.000 |
| in `eval()` mode, alone | 1.000 |

Same image, same weights: from 0.1% to 100% depending only on the company it keeps.

> 📓 **Notebook rule:** *`model.train()` before training, `model.eval()` before scoring.* With batch norm (and dropout, Section 14.4) in the network, forgetting this silently changes your answers.

**Related limits worth knowing:**

- **Tiny batches make batch norm noisy.** With a batch of 2, the "mean" is barely a mean. (In training mode, PyTorch refuses outright to batch-normalise a single example.)
- **Layer normalisation** avoids the whole issue by normalising each example **across its own neurons** instead of across the batch, so every example is handled independently and train and test behave identically. That's why Transformers use it, and we'll meet it again in Part VI.

---

### 📓 Notebook margin: the equation so far

$$
\hat z = \frac{z - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}
\qquad
y = \gamma\,\hat z + \beta
\qquad
\text{layer} = \text{ReLU}\big(\text{BN}(W\mathbf{h})\big)
$$

| idea | what we now know |
|---|---|
| batch norm | standardise each neuron over the batch, then learnable scale γ and shift β |
| during training | each layer's inputs stay centred with spread ≈ 1 instead of drifting |
| deep networks | 20 layers with bad init: 11.0% → **94.5%** |
| learning rates | still trains at η = 0.5 (94.8%) where the plain network dies |
| two modes | `train()` uses batch statistics, `eval()` uses running averages |
| the trap | same-digit batches in train mode: 96.1% → **17.0%** |
| layer norm | normalises per example: the Transformer's version (Part VI) |

---

### What comes next

Part II now has all its pieces: honest measurement (13), regularisation (14), initialisation (15), optimisers and schedules (16), and normalisation (17). **Section 18: The Recipe** closes Part II by putting them together:

- A step-by-step training **workflow**, including the sanity checks that catch most bugs before they cost you a day (overfit one batch first, check the starting loss is −log(1/10), watch the validation curve).
- Everything combined on the **full** MNIST set: He init, batch norm, AdamW, warm-up + cosine, augmentation and dropout, all tuned on validation, with the test set opened once.
- A new scoreboard against Part I's 97.87%, and an honest look at how far a network of plain layers can go before it needs the new idea of Part IV.

---

*References: Sergey Ioffe & Christian Szegedy (2015), "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift". Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas & Aleksander Mądry (2018), "How Does Batch Normalization Help Optimization?" (the smoother-landscape explanation). Jimmy Lei Ba, Jamie Ryan Kiros & Geoffrey Hinton (2016), "Layer Normalization". François Chollet, *Deep Learning with Python*, 3rd ed., ch. 9 (batch normalisation in practice: placement, and the train/inference difference). All code in this series is PyTorch.*
