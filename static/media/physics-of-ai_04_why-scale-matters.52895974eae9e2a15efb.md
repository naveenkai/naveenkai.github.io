# Physics of AI — Part I · The Gears

## 4. Why Scale Matters

> *Section 3 ended with a small unease. We divided every MNIST pixel by 255 and moved on without asking why. Then we lined up two honest inputs, review score (0–10) and number of ratings (15,000–1,00,000), and wondered what happens when one is ten thousand times bigger than the other. This section answers that. The answer turns out to explain why data scientists do half the "boring" preprocessing steps they do.*

---

### 4.1 A new input, an honest mistake

Let's go back to the movie decision with 600 past movie nights. Each night now has three measurements:

| input | range | encoded as |
|---|---|---|
| review score | 0 – 10 | a number ✓ |
| how much I like the cast | 0 – 10 | a number ✓ |
| number of ratings | 15,000 – 1,00,000 | a number ✓ |

Every rule from Section 3 is followed: these are real measurements, so they stay as numbers, with no fake categories. We'll use **synthetic** data so that we know the truth. Watching depends mostly on reviews and cast, and how many people rated it matters only a tiny bit:

```python
import torch
torch.manual_seed(0)

n = 600
reviews   = torch.rand(n) * 10
cast      = torch.rand(n) * 10
n_ratings = 15_000 + torch.rand(n) * 85_000

logit   = 0.9*reviews + 0.7*cast - 8 + 0.00002*(n_ratings - 57_500)
watched = (logit + torch.randn(n) > 0).long()          # a little noise, like real life

X = torch.stack([reviews, cast, n_ratings], dim=1)     # shape (600, 3)
X_train, X_test = X[:400], X[400:]
y_train, y_test = watched[:400], watched[400:]
```

About half the nights end with the movie watched (49%), so **a coin flip scores 50%**. Anything useful has to beat that.

---

### 4.2 A model with no weights: ask your neighbours

Before we train neurons, here's the simplest "learning" idea there is, **k-nearest neighbours (KNN)**:

> *To decide about tonight, find the $k$ past nights most similar to it and do whatever you did on most of them.*

There's no $\mathbf{w}$, no $b$ and no training. The model is just **the data plus a notion of distance**. The distance is ordinary straight-line (Euclidean) distance, the same thing Section 3's `cdist` computed:

$$
d(\mathbf{a}, \mathbf{b}) = \sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2 + (a_3 - b_3)^2}
$$

In PyTorch the whole model is three lines:

```python
def knn_predict(X_train, y_train, X_query, k=7):
    d   = torch.cdist(X_query, X_train)               # every query vs every past night
    idx = d.topk(k, largest=False).indices            # the k closest
    return y_train[idx].mode(dim=1).values            # majority vote

acc = lambda pred, y: (pred == y).float().mean().item()
acc(knn_predict(X_train, y_train, X_test), y_test)    # → 0.50
```

**50%.** That's exactly a coin flip. The model learned nothing, even though the pattern in the data is strong.

---

### 4.3 Who's actually doing the talking?

Take two movies that should feel completely different:

| | reviews | cast | ratings |
|---|---|---|---|
| movie A | 1 | 2 | 40,000 |
| movie B | 9 | 8 | 40,600 |

Movie A is panned and has a cast you don't care about. Movie B is acclaimed and stars people you love. Those are opposite decisions. Now look at what goes into the distance:

$$
d^2 = \underbrace{(1-9)^2}_{64} + \underbrace{(2-8)^2}_{36} + \underbrace{(40{,}000 - 40{,}600)^2}_{\mathbf{360{,}000}}
$$

A trivial difference of 600 ratings outweighs everything else by a factor of 3,600.

![Share of distance by feature, raw vs standardised](figures/fig11_distance_share.png)

**Distance doesn't know about units.** It doesn't know that "8 points of review score" is a huge difference and "600 more ratings" is nothing. It only sees the raw size of the numbers, and the rating counts are huge. So KNN's "nearest nights" are really just **movies with a similar number of ratings**:

![The 7 nearest neighbours, raw vs standardised](figures/fig12_neighbours.png)

On the left, the 7 "nearest" nights to tonight (★) are scattered across the reviews–cast plane. Their rating counts are what's close: all within a few hundred of 50,000. Two of them are poorly reviewed movies with a cast you don't care for, which tell us nothing about tonight. This particular query got lucky with 5 of 7 correct. Across the whole test set, luck averages out to a coin flip.

> 📓 **Notebook rule:** *a feature's influence on distance is proportional to its numeric spread, not its importance.* Whatever has the biggest numbers gets the loudest voice.

---

### 4.4 The fix: put every feature on the same ruler

We want each feature measured in **its own natural unit**: "how unusual is this value, *for this feature*?" That unit is the standard deviation. **Standardisation** (the z-score) rescales each column to mean 0 and spread 1:

$$
\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}
$$

After this, "reviews = +1" means *one typical spread above the average review score*, and "ratings = +1" means *one typical spread above the average number of ratings*. The two are finally comparable.

```python
mean, std = X_train.mean(0), X_train.std(0)
# mean ≈ [4.95, 5.07, 57,777]    std ≈ [2.89, 2.92, 25,190]

Xtr_s = (X_train - mean) / std
Xte_s = (X_test  - mean) / std          # ← test uses TRAIN statistics

acc(knn_predict(Xtr_s, y_train, Xte_s), y_test)     # → 0.875
```

**From 50% to 87.5% with one line**, the same model on the same data. Only the ruler changed. Look back at the right half of the distance-share figure: reviews and cast now make up nearly all of the distance, and those 600 ratings contribute 0.005%.

It isn't a lucky choice of $k$ either:

![KNN accuracy vs k, raw vs standardised](figures/fig13_knn_accuracy.png)

**Why test uses *train* statistics.** Pretend the test nights are tomorrow's. You can't compute tomorrow's average before tomorrow happens. If you standardise with statistics that include the test set, information from the "future" leaks into your preprocessing, and your accuracy estimate becomes optimistic. So: fit the scaler on train and apply it everywhere.

**The other common ruler: min-max scaling.** It squeezes each feature into $[0, 1]$:

$$
\tilde{x}_j = \frac{x_j - \min_j}{\max_j - \min_j}
$$

It also scores **87.5%** here. That's what `images / 255` from Section 3 was doing: pixels run from 0 to 255, so dividing by 255 *is* min-max scaling.

---

### 4.5 Equal voice ≠ equal importance

Here's an experiment that should bother you. **Drop the number of ratings entirely**:

```python
acc(knn_predict(X_train[:, :2], y_train, X_test[:, :2]), y_test)   # → 0.905
```

Two features beat three. Standardisation gave the rating count an **equal voice**, but it barely matters to the decision, so it now adds a little noise to every distance.

Scaling solves one problem: *no feature is loud just because of its units.* It can't tell you *which features deserve a voice at all*. For that you have to look at the data. Hold that thought until 4.8.

---

### 4.6 It's not just KNN: scale breaks the neuron too

KNN has no weights, so you might hope a neuron could "learn around" a big feature by giving it a small weight. Sometimes it can, *eventually*. But look at what happens on the very first step.

A fresh `nn.Linear` starts with small random weights, roughly between −0.6 and +0.6 here. Multiply one of those by a rating count of 57,000 and you get a $z$ in the thousands:

```python
import torch.nn as nn
torch.manual_seed(1)
neuron = nn.Linear(3, 1)

for name, inputs in [("raw", X_train), ("standardised", Xtr_s)]:
    neuron.zero_grad()
    p = torch.sigmoid(neuron(inputs).squeeze())
    loss = nn.functional.binary_cross_entropy(p, y_train.float())
    loss.backward()
    print(name, neuron.weight.grad)
```

| | mean $\lvert z \rvert$ | nights with σ(z) stuck at ~0 or ~1 | gradient on the weights |
|---|---|---|---|
| **raw** | 6,466 | **100%** | **[0, 0, 0]** |
| **standardised** | 0.4 | 0% | [−0.27, −0.27, −0.10] |

A gradient of **exactly zero** means the neuron can't learn at all. Here's why, as a picture:

![Sigmoid saturation: raw vs standardised z](figures/fig14_saturation.png)

Remember Section 2's "sharpness" figure, where scaling $z$ up turned the sigmoid back into a step? The raw rating count did exactly that. Every night lands far out on a flat part of the curve, where the slope is zero. **No slope, no learning**, the same lesson as Section 2 arriving from a different direction.

With standardised inputs, $z$ lands on the slope, where small weight changes produce small, useful output changes. That's where learning happens.

(There's a second, subtler problem. Even when the neuron isn't saturated, the gradient for each weight is proportional to its input, so the rating count's weight would get pushed thousands of times harder than the reviews weight. That makes gradient descent zig-zag. We'll see it properly when we draw loss landscapes in Section 9.)

---

### 4.7 So why did we divide MNIST by 255?

Here's a subtle point. All 784 pixels share **the same scale**, 0 to 255. Scaling every feature by the same number shrinks all distances by the same factor, so **the nearest neighbours don't change**:

```python
# 4,000 train / 1,000 test MNIST digits, k = 3
acc(knn_predict(train_px,       ...), ...)     # → 0.935
acc(knn_predict(train_px / 255, ...), ...)     # → 0.935   identical
```

So `/255` was never for KNN. **It's for the neuron.** A digit has around 150 inked pixels at values up to 255. Dot those with random weights and $z$ lands deep in the flat zone of the sigmoid, exactly like the rating count did. After `/255`, the pixels sit in $[0, 1]$ and $z$ stays near the slope.

> 📓 **Notebook rule:** *different scales across features → distance-based models break. Big scale overall → neurons saturate. Scaling fixes both.*

---

### 4.8 EDA, reverse-engineered

Most tutorials hand you a preprocessing checklist: *check ranges, look at distributions, handle outliers, encode categories.* It feels like ritual.

We just derived every item from the maths:

| the maths forced us to ask… | so in EDA you… | we saw it in |
|---|---|---|
| Are features on wildly different scales? | look at `min`, `max`, `mean`, `std` per column | 4.3: rating count drowned out reviews |
| Which ruler: z-score or min-max? | look at the **distribution** and **outliers** | below |
| Is this number secretly a category? | check what each column *means* | 3.7: genres as 1, 2, 3 |
| Does this feature deserve a voice? | check how it relates to the target | 4.5: rating count was mostly noise |

**The outlier case, to make it concrete.** Suppose one blockbuster in the data has 10,00,000 ratings. Min-max divides every movie by that one movie's range, so everyone else's rating count gets squeezed into $[0, 0.09]$, and the feature is flattened to almost nothing. The z-score is shifted too (the mean and std move), but it's less fragile. You only notice this if you **look at the column first**, which is what EDA is.

> 📓 **Notebook rule:** *EDA isn't a checklist. It's the set of questions the maths forces you to ask before you trust a distance or a gradient.*

---

### 📓 Notebook margin: the equation so far

$$
\tilde{\mathbf{x}} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}
\qquad\longrightarrow\qquad
\hat{y} = \sigma(\,\mathbf{w}\cdot\tilde{\mathbf{x}} + b\,)
$$

| idea | what we now know |
|---|---|
| distance | only as fair as the scales you feed it |
| standardisation | every feature measured in "typical spreads", using **train** stats only |
| `/255` | min-max for pixels; it keeps the neuron's $z$ on the slope |
| EDA | the questions that come out of distance and gradients |

The input $\mathbf{x}$ is now in good shape: **numbers, correctly encoded, on a fair scale.**

---

### What comes next

We've spent three sections getting $\mathbf{x}$ right. Now it's time to look hard at what the neuron **does** to it.

In Section 2 we saw that $\mathbf{w}\cdot\mathbf{x} + b$ draws a line. But a whole *layer* of neurons, $W\mathbf{x} + \mathbf{b}$, does something much more physical: it **moves space**. It stretches it, rotates it, shears it and slides it. Every trick a neural network has is built from those few motions.

That's **Section 5: Tensor Operations as Physics**.

---

*References: François Chollet, *Deep Learning with Python*, 3rd ed., ch. 2 (data preprocessing and normalisation of MNIST) and ch. 4–5 (feature normalisation, data leakage). Michael Nielsen, *Neural Networks and Deep Learning*, ch. 3 (saturation and learning slowdown in sigmoid neurons). All code in this series is PyTorch.*
