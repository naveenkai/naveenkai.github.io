# Physics of AI — Part II · Making It Learn

## 18. The Recipe

> *Part II gave us a toolbox: honest measurement, regularisation, initialisation, better optimisers, schedules and normalisation. This last section of Part II turns the toolbox into a routine. First, the cheap sanity checks that catch most bugs in minutes. Then every tool combined on the full MNIST set, with an honest test of which ones actually earn their place. Finally, the test set is opened once to see how far plain layers have come since Part I.*

---

### 18.1 Before training: three sanity checks

Most training bugs don't crash. The code runs, the loss goes down a bit, and you get a disappointing number with no idea why. These three checks take a couple of minutes and catch a surprising share of them.

**Check 1: is the starting loss right?** A freshly initialised network knows nothing, so it should give roughly equal probability to all 10 digits. From Section 8.5, that means a loss of $-\log(1/10) = 2.303$.

```python
model = build(); acc, loss = evaluate(model, X_val, Y_val)
# default init: 2.302    He init + batch norm: 2.313      ← both ≈ 2.303 ✓
```

If the starting loss is 5 or 50, something is already wrong: the inputs aren't scaled (Section 4), the initialisation is off (Section 15), or the labels don't match the outputs.

**Check 2: can the model memorise one batch?** Take just 64 digits and train on them over and over. Section 13 showed these networks can memorise *random* labels, so 64 real ones should reach almost zero loss quickly. If they don't, the bug is in the code, not the data.

![Overfitting one batch: correct code vs two classic bugs](figures/fig77_overfit_batch.png)

| code | loss after 0 / 50 / 100 / 300 steps |
|---|---|
| **correct** | 2.34 → **0.000** → 0.000 → 0.000 |
| softmax applied before `cross_entropy` | 2.30 → **1.476** → 1.476 → 1.476 (stuck forever) |
| forgot `opt.zero_grad()` | 2.34 → 2.34 → **2.61** → 0.000 (goes *up* first) |

The middle row is one of the most common bugs in PyTorch. `F.cross_entropy` already applies softmax internally (Section 8.6), so applying it yourself squashes the outputs twice, and the loss can never fall below about 1.46. The third row is sneakier: without `zero_grad()`, gradients pile up from every previous step (PyTorch *adds* into `.grad`), so each step uses a growing, stale sum. Adam's rescaling eventually muddles through here, but the early loss **rises**, which is a red flag.

**Check 3: which learning rates are sensible?** Run one short pass where the learning rate grows a little every step, from $10^{-6}$ up to 10, and watch the loss:

![Learning-rate range test](figures/fig78_lr_range.png)

Too small and nothing happens (flat at 2.4). In the useful range, roughly $10^{-3}$ to $10^{-2}$ here, the loss drops fastest. Past about $10^{-1}$ it starts climbing, and by 1 it explodes. Pick a peak learning rate inside the good range, a bit below where the curve bottoms out. (This isn't exact, since the loss also falls simply because training is happening, but it reliably rules out values that are ten times too big or too small in one short run.)

---

### 18.2 The full recipe

Here's every Part II tool, applied to the MNIST network:

```python
def build(bn=True, drop=0.2, he=True, widths=(512, 256)):
    layers, d = [nn.Flatten()], 784
    for w in widths:
        layers += [nn.Linear(d, w, bias=not bn)]                 # bias redundant before BN (17.1)
        layers += [nn.BatchNorm1d(w)] if bn else []              # Section 17
        layers += [nn.ReLU()]                                    # Section 6
        layers += [nn.Dropout(drop)] if drop else []             # Section 14.4
        d = w
    model = nn.Sequential(*layers, nn.Linear(d, 10))
    if he:                                                       # Section 15
        for m in model:
            if isinstance(m, nn.Linear) and m is not model[-1]:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    return model

opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)     # Sections 14.3, 16.4
# every step: lr = peak × (warm-up for 1 epoch, then cosine to 0)           # Section 16.6
# every batch: random rotation ±12° and shift ±2.5 px                        # Section 14.5
```

It trains on 50,000 digits, with 10,000 held out as validation (Section 13.5), for 15 epochs. The test set stays sealed.

---

### 18.3 Which ingredients actually earn their place?

Adding everything isn't the same as knowing what works. So we run an **ablation**: take the full recipe, remove **one** ingredient at a time, and see how much validation accuracy drops. To know what counts as a real difference, three of the configurations were run with three different random seeds (Section 12.3's lesson: small gaps can be noise).

![Ablation: removing one ingredient at a time](figures/fig79_ablation.png)

| configuration (15 epochs) | validation accuracy |
|---|---|
| **full recipe** | 98.81% (seeds: 98.81 / 98.85 / 99.00) |
| − augmentation | **98.38%** |
| − batch norm | 98.77% |
| − dropout | **98.93%** (seeds: 98.93 / 98.95 / 99.04) |
| − weight decay | 98.91% |
| − warm-up + cosine | 98.75% |
| − He init | 98.87% |
| **Part I style** (none of the above, Adam, constant lr) | **97.72%** (seeds: 97.72 / 97.77 / 97.87) |

Read it with the seed noise in mind: the same configuration moves by about **0.2 points** from seed to seed.

- **Together, the recipe is clearly better:** about 98.9% against 97.8%, a gap roughly five times the noise.
- **Augmentation is the one ingredient that clearly matters on its own.** Removing it costs about 0.5 points. Same lesson as Section 14.5.
- **Batch norm, He init and the schedule** each change things by less than the noise. That fits what we learned: they matter most in **deep** networks (Sections 15 and 17), and this one has only two hidden layers.
- **Removing dropout slightly *helped*.** Look at why in the table: with augmentation on, training accuracy (99.1%) is barely above validation (98.8%). There's hardly any overfitting left for dropout to fix, so it only gets in the way.

> 📓 **Notebook rule:** *regularisation only helps if you're overfitting.* Check the training-vs-validation gap first. If it's already small, more regularisation just holds the network back.

So the recipe we keep: **augmentation + batch norm + He init + AdamW + warm-up/cosine, without dropout.** One more choice made on validation: 30 epochs instead of 15 lifts validation to **99.25%**, so we keep 30.

---

### 18.4 Opening the envelope

Every decision so far used only validation data. Now the standard last step: retrain the chosen recipe on **all 60,000** training digits (train + validation, since there's nothing left to choose), and open the test set **once**:

| | test accuracy | mistakes out of 10,000 | test loss |
|---|---|---|---|
| Part I, Way 3 (Section 12) | 97.87% | 213 | 0.081 |
| **Part II recipe** | **99.25%** | **75** | **0.021** |

![MNIST scoreboard, Part I → Part II](figures/fig80_scoreboard2.png)

**Almost two-thirds of the mistakes are gone**, with the same kind of network: fully connected layers of neurons, no new architecture. (Batch norm is the only new layer, and 18.3 showed it contributes little here.) Almost every extra point came from *how* it was trained.

And the network isn't only more accurate, it's **more honest about its mistakes**:

| on the test digits it gets wrong | Part I model | Part II recipe |
|---|---|---|
| average confidence in the wrong answer | 81% | 73% |
| wrong answers given with > 99% confidence | 13% | **3%** |

Here are the mistakes it's still most confident about:

![Most confident of the 75 remaining mistakes](figures/fig81_remaining_mistakes.png)

Many of these would make a person hesitate too: an 8 whose loops are nearly closed like a 9, a 7 that's mostly one stroke like a 1, a 5 with a looped belly like a 6. The network hasn't become perfect. It has become reasonable.

---

### 18.5 The recipe, as a routine

| step | what | why (section) |
|---|---|---|
| 1 | split off a validation set; seal the test set | 13.5 |
| 2 | scale the inputs | 4 |
| 3 | check the starting loss ≈ −log(1/classes) | 8.5, 15 |
| 4 | overfit one batch to ~0 loss | 13.2 |
| 5 | learning-rate range test; pick a peak rate in the good range | 9.5, 16 |
| 6 | train with a sensible default: Adam(W), warm-up + cosine; He init and batch norm for deep nets | 15–17 |
| 7 | compare training vs validation: **big gap** → regularise (augmentation first); **small gap** → train longer or use a bigger model | 13–14 |
| 8 | ablate: remove each ingredient; keep only what beats seed noise | 12.3, 18.3 |
| 9 | retrain on train + validation; open the test set **once** | 13.5 |

---

### 📓 Notebook margin: Part II in one table

| section | idea | biggest measured effect |
|---|---|---|
| 13 | memorising vs learning | random labels: 100% train, 10.7% test |
| 14 | regularisation | 1,000 digits: 89.8% → 95.6% (augmentation did most of it) |
| 15 | initialisation | 20 layers: 11.0% → 92.3% (He init) |
| 16 | optimisers and schedules | canyon: 198 → 13 steps; warm-up saves η = 0.5 (11% → 97.2%) |
| 17 | batch norm | 20 layers, bad init: 11.0% → 94.5% |
| 18 | the recipe | **MNIST test: 97.87% → 99.25%** |

---

## End of Part II

Part I built a machine that learns. Part II made it learn the *right* thing, reliably. Along the way we kept bumping into one question and putting it off: **why does depth work at all?**

In Section 7 we saw that stacking folds multiplies them. In Section 11.8 and Section 15 we saw that depth is also what makes training fragile. Part III takes the question head on:

- **Section 19: Any Function At All.** Nielsen's visual proof that one hidden layer can, in principle, approximate *any* function, and why "in principle" hides an enormous cost.
- Then why deep beats wide in practice, why very deep networks still broke even with Part II's tools, and the one-line fix, **residual connections**, that made 100-layer networks trainable and sits inside every Transformer we'll build in Part VI.

---

*References: Andrej Karpathy (2019), "A Recipe for Training Neural Networks" (the overfit-one-batch check, verifying the initial loss, and building up from simple baselines). Leslie Smith (2017), "Cyclical Learning Rates for Training Neural Networks" (the learning-rate range test). Ilya Loshchilov & Frank Hutter (2019), "Decoupled Weight Decay Regularization" (AdamW). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 6 ("The universal workflow of machine learning"). Michael Nielsen, *Neural Networks and Deep Learning*, ch. 3 ("How to choose a neural network's hyper-parameters?"). All code in this series is PyTorch.*
