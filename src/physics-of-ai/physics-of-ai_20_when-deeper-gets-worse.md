# Physics of AI — Part III · Why Depth?

## 20. When Deeper Gets Worse

> *Section 19 left us with a tension. Depth is efficient (a trained 4-layer network beat a 1-layer one by 44×), but depth is fragile. Part II's tools (He initialisation, batch norm, Adam, warm-up) were supposed to fix the fragility. This section pushes them to the limit, finds that they aren't enough, and then fixes the problem with one small change: **add the input back**. That change, the residual connection, made 100-layer networks routine, and it sits inside every Transformer we'll build in Part VI.*

---

### 20.1 The experiment that shouldn't fail

Here's a thought experiment. Take a network with 20 layers that trains well. Now build a 50-layer network: copy the 20 good layers, and make the 30 extra layers do **nothing**, just pass their input straight through. This deeper network computes exactly the same function as the 20-layer one, so it can't be *worse*.

So if we train a 50-layer network from scratch, it should do at least as well as the 20-layer one, at least on the **training** data. Let's check. These are plain stacks of `Linear → BatchNorm → ReLU` layers, 128 wide, with He initialisation and Adam (everything from Part II), trained for 3 epochs on 50,000 digits:

![More layers, worse training loss](figures/fig87_degradation.png)

| hidden layers | **training** loss | **training** accuracy | validation accuracy |
|---|---|---|---|
| 4 | 0.040 | 98.8% | 97.1% |
| 10 | 0.065 | 98.1% | 96.8% |
| 20 | 0.124 | 96.6% | 95.2% |
| 50 | **2.18** | **17.3%** | 18.0% |
| 100 | **3.41** | **12.6%** | 11.9% |

The deeper the network, the **worse it does on its own training data**. At 50 layers it has essentially stopped learning.

Be careful to see what this is *not*:

- **It's not overfitting** (Section 13). Overfitting means great training scores and poor validation scores. Here the *training* score collapses too.
- **It's not the vanishing signal from Section 15.** We used He initialisation and batch norm, which were built to fix exactly that.

This is the **degradation problem**, and it puzzled researchers around 2015 (He, Zhang, Ren & Sun saw the same thing with 20- vs 56-layer image networks). The extra layers *could* learn to do nothing, and that solution exists by construction. Gradient descent just can't find it.

![Training curves: 50 plain layers barely move](figures/fig88_curves.png)

---

### 20.2 Why "do nothing" is hard to learn

Let's test the heart of the thought experiment directly. Give a stack of plain layers the simplest possible job: **output exactly what came in**, $y = x$, for random 32-dimensional inputs.

| layers | error after 2,000 steps |
|---|---|
| 1 plain layer | 0.021 |
| 5 plain layers | 0.053 |
| 20 plain layers | **0.106** |

![Learning to copy the input: plain vs residual](figures/fig90_identity.png)

Twenty layers of `Linear → ReLU` can't learn to copy their input, and more layers make it worse. For a plain layer, "do nothing" means finding weights that make $\text{ReLU}(W\mathbf{h} + \mathbf{b}) = \mathbf{h}$: an exact identity matrix, zero bias, and no neuron ever dipping below zero. That's a very specific, narrow target. Random starting weights are nowhere near it, and every layer's error feeds into the next.

---

### 20.3 The fix: add the input back

**He and colleagues' idea (2015):** don't ask a layer to *produce* the new representation. Ask it to produce a **change** to add to the old one:

$$
\text{plain:}\quad \mathbf{h} \leftarrow F(\mathbf{h})
\qquad\qquad
\text{residual:}\quad \mathbf{h} \leftarrow \mathbf{h} + F(\mathbf{h})
$$

![Plain layer vs residual block](figures/fig89_residual_block.png)

$F$ is the same `Linear → BatchNorm → ReLU` as before. The only change is the **skip connection** that carries $\mathbf{h}$ around it and adds it back. In PyTorch it's literally one character, `h + `:

```python
class Residual(Plain):
    def forward(self, x):
        h = torch.relu(self.inp(x))
        for lin, bn in zip(self.layers, self.norms):
            h = h + torch.relu(bn(lin(h)))        # plain version: h = torch.relu(bn(lin(h)))
        return self.out(h)
```

Now "do nothing" is easy: it just means $F(\mathbf{h}) = 0$, which you get by making the weights small, the easiest thing in the world for weight decay and random initialisation to land near. The copying test from 20.2, with residual blocks and ordinary random starting weights (each branch's output scaled by 0.1, a common way to start residual branches small):

| layers | plain | **residual** |
|---|---|---|
| 1 | 0.021 | **0.000000** |
| 5 | 0.053 | **0.000004** |
| 20 | 0.106 | **0.00017** |

The residual stack actually *started* worse at 20 layers (its random branches add up to a lot of noise), but it learns to quieten them and copies the input almost perfectly. The plain stack never gets close.

Think of editing an exam answer. A plain layer is like a teacher who must **rewrite the whole answer from scratch** each time. A residual block is a teacher who writes **corrections in the margin**. If the answer is already right, the margin stays empty, and nothing is lost.

---

### 20.4 The degradation problem, fixed

Same experiment as 20.1, same settings, with `h +` added:

| hidden layers | plain: training acc | **residual: training acc** | residual: validation acc |
|---|---|---|---|
| 4 | 98.8% | 98.8% | 97.2% |
| 10 | 98.1% | 99.0% | 97.3% |
| 20 | 96.6% | 98.5% | 97.0% |
| 50 | 17.3% | **98.0%** | 96.4% |
| 100 | 12.6% | **98.3%** | 96.7% |

**A 100-layer network now trains as well as a 4-layer one.** (In this small 3-epoch test, extra depth doesn't *beat* 4 layers on MNIST. Digits don't need 100 layers. The point is that depth has stopped *hurting*. On harder problems, like the images in Part IV, that turns depth from a liability into the thing that wins.)

It even works without batch norm. At 50 layers with no normalisation at all, after one epoch:

| 50 layers, no batch norm | training accuracy |
|---|---|
| plain | 62.3% |
| residual (branch weights started 10× smaller) | **95.8%** |

---

### 20.5 Why it works: a highway for the gradient

Look at what one residual block does to the gradient during backprop (Section 11). The chain rule through $\mathbf{h}_{\text{next}} = \mathbf{h} + F(\mathbf{h})$ gives:

$$
\frac{\partial \mathbf{h}_{\text{next}}}{\partial \mathbf{h}} \;=\; \underbrace{I}_{\text{the skip}} \;+\; \frac{\partial F}{\partial \mathbf{h}}
$$

In a plain network, the gradient reaching layer 1 is a **product** of 50 layer-by-layer factors, and a product of 50 numbers either explodes or vanishes unless every one of them is almost exactly 1 (Sections 11.8 and 15). In a residual network every factor is **"1 + something"**, so the product always contains a direct term: the gradient from the loss flows straight down the skip connections to every layer, untouched. Each branch's own gradient rides on top of it.

Here's the learning signal each layer gets at the start of training, 50 layers, both with batch norm and He init:

![Gradient size per layer at initialisation: plain vs residual](figures/fig91_grad_flow.png)

| 50 layers, at initialisation | gradient at layer 1 ÷ gradient at layer 50 |
|---|---|
| plain | **9,151** |
| residual | 11 |

Interesting twist: in the plain network the gradient doesn't *vanish* here, it **explodes**, growing about 9,000-fold on its way back to the first layer. Batch norm stopped the forward signal from vanishing (Section 17), but in a very deep plain stack it makes the backward signal blow up instead. (This is a known effect, analysed by Yang and colleagues in 2019.) Either way, early and late layers receive wildly different signals, and no single learning rate suits them all. With skip connections every layer gets a learning signal of about the same size.

There's another way to see it. Unroll the residual network and its output is a **sum**:

$$
\mathbf{h}_L \;=\; \mathbf{h}_0 + F_1(\cdot) + F_2(\cdot) + \dots + F_L(\cdot)
$$

The input travels all the way to the output unchanged, and each block adds a small correction on top. A deep residual network behaves less like one enormous chain and more like many shallower paths added together, which is much easier to train.

> 📓 **Notebook rule:** *don't make a layer rebuild the signal, let it add a correction.* $\mathbf{h} + F(\mathbf{h})$ gives the forward signal a way through and the backward gradient a highway home.

---

### 20.6 Where you'll meet it again

In 2015, residual networks ("ResNets") won the ImageNet image-recognition competition with **152 layers**, when most networks before them had struggled past about 20 to 30. Today skip connections are nearly everywhere, and in particular inside every Transformer. A Transformer block is, at heart, two residual additions:

$$
\mathbf{h} \leftarrow \mathbf{h} + \text{Attention}\big(\text{LayerNorm}(\mathbf{h})\big)
\qquad
\mathbf{h} \leftarrow \mathbf{h} + \text{MLP}\big(\text{LayerNorm}(\mathbf{h})\big)
$$

That's `h + F(h)`, with the layer norm from Section 17.5 inside the branch. When we build one in Part VI, this structure is what lets us stack dozens of blocks.

(A small honesty note on our block: we used $\mathbf{h} + \text{ReLU}(\text{BN}(W\mathbf{h}))$, the simplest version. The original ResNet puts two layers inside $F$ and applies the ReLU after the addition, and Transformers normalise *before* the branch. The idea, adding a learned correction to an untouched copy, is the same in all of them.)

---

### 📓 Notebook margin: the equation so far

$$
\mathbf{h}_{\ell+1} = \mathbf{h}_\ell + F_\ell(\mathbf{h}_\ell)
\qquad\Longrightarrow\qquad
\frac{\partial \mathbf{h}_{\ell+1}}{\partial \mathbf{h}_\ell} = I + \frac{\partial F_\ell}{\partial \mathbf{h}_\ell}
$$

| idea | what we now know |
|---|---|
| degradation | deeper plain nets get worse **on training data**: 50 layers → 17.3% |
| not overfitting, not init | happens even with He init + batch norm |
| "do nothing" is hard | 20 plain layers can't learn $y = x$ (0.106 vs 0.00017) |
| residual block | $\mathbf{h} + F(\mathbf{h})$: layers learn corrections, not replacements |
| result | 100 residual layers train as well as 4 (98.3% training accuracy) |
| gradient highway | layer-1 ÷ layer-50 gradient: 9,151 (plain) vs 11 (residual) |
| everywhere | ResNets (152 layers), and every Transformer block |

---

### What comes next

**Section 21: The Modern Block** closes Part III by assembling the building block that modern networks, Transformers included, actually stack: a residual connection, **layer norm** placed *before* the branch ("pre-norm"), a wider hidden layer inside the branch, and a smoother activation (GELU). We'll test each choice on the same deep-network setup, and end Part III with a 100-block network that trains cleanly with no special care. That block is, almost exactly, the MLP half of the Transformer.

---

*References: Kaiming He, Xiangyu Zhang, Shaoqing Ren & Jian Sun (2015), "Deep Residual Learning for Image Recognition" (the degradation problem, residual learning, 152-layer ResNets). Kaiming He et al. (2016), "Identity Mappings in Deep Residual Networks" (why the clean identity path matters for gradient flow). Andreas Veit, Michael Wilber & Serge Belongie (2016), "Residual Networks Behave Like Ensembles of Relatively Shallow Networks". Greg Yang et al. (2019), "A Mean Field Theory of Batch Normalization" (gradient explosion in deep plain networks with batch norm). Michael Nielsen, *Neural Networks and Deep Learning*, ch. 5 (unstable gradients in deep networks). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 9 (residual connections). All code in this series is PyTorch.*
