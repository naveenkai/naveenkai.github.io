# Physics of AI — Part II · Making It Learn

## 13. Memorising vs Learning

> *Part I ended with a network that gets 99.4% of its training digits right and 97.9% of new ones. That 1.5-point gap looks small, but it's a symptom of the most important question in machine learning. We never actually care how well a network does on the examples it trained on. We care how it does on examples it has **never seen**. This section is about the difference between memorising and learning, how to measure it honestly, and a famous experiment showing just how good neural networks are at memorising.*

---

### 13.1 Two students

Two students are preparing for the same university exam.

The first gets hold of the last ten years' question papers and memorises every answer, word for word. The second works through the same papers but focuses on *why* each answer is right.

On any question copied from an old paper, both score perfectly. On a question they haven't seen before, the first student is stuck, while the second can work it out.

A neural network can become either student. **Training accuracy can't tell them apart**, because both are perfect on the practice papers. Only questions from outside the practice set reveal which one you have.

That's why every accuracy number in this series has been reported on **test** digits the network never trained on. The ability to do well on new data is called **generalisation**, and it's the only thing that matters in the end.

---

### 13.2 The experiment: memorising nonsense

Here's an experiment that makes the problem impossible to ignore. It follows a well-known 2017 paper by Zhang and colleagues, *"Understanding deep learning requires rethinking generalization"*.

Take 5,000 MNIST digits. Now **shuffle the labels**, so each image gets the label of some other random image. A "7" might now be labelled "2", and another "7" labelled "5". Only 9.8% of the labels are still correct, which is about chance. There's no pattern left to learn. The labels are pure noise.

Train the same network on real labels and on shuffled labels:

```python
Y_random = Y[torch.randperm(5000)]                 # destroy the link between image and label

model = nn.Sequential(nn.Linear(784, 512), nn.ReLU(),
                      nn.Linear(512, 512), nn.ReLU(),
                      nn.Linear(512, 10))          # 669,706 weights
# trained with Adam, batch 128, 100 epochs, once on Y and once on Y_random
```

![Real labels vs shuffled labels](figures/fig54_random_labels.png)

| | training accuracy (epoch 100) | test accuracy |
|---|---|---|
| **real labels** | 100% | **94.6%** |
| **shuffled labels** | **100%** | 10.7% (chance) |

Look at the right panel. With labels that mean *nothing*, the network still reaches **100% training accuracy**. It memorised an arbitrary label for each of 5,000 images. It needed longer (about 50 epochs, against about 10 for real labels), but it got there. Test accuracy stayed at chance the whole time, because there was never anything to generalise.

Two lessons, one reassuring and one uncomfortable:

- **A perfect training score proves nothing.** This network can hit 100% on training data whether it learned something or nothing.
- **These networks have enough capacity to memorise.** 669,706 weights and only 5,000 examples: there's room to store every answer separately. Yet with *real* labels the same network generalises to 94.6%. So why doesn't it just memorise the real labels too? That question is still being researched. The best current answer is that real data has patterns that are **easier** to learn than memorising, and gradient descent finds the easy patterns first. It's not a complete explanation, so hold it loosely.

---

### 13.3 What overfitting looks like

With the shuffled labels, memorising is *all* the network can do. With real labels it does both: it learns the patterns, and then, if we let it, it keeps going and memorises the leftovers, including the **noise** in the training data.

Here's that in a picture we can see. Take 400 noisy two-moons points (the Section 6 dataset, with some points on the "wrong" side because of noise), and three models:

![Big net trained long vs stopped early vs small net](figures/fig57_moons_overfit.png)

| model | training accuracy | new points |
|---|---|---|
| big net (256 neurons per layer), trained long | **99.0%** | 87.8% |
| same big net, stopped early | 93.0% | **90.1%** |
| small net (8 neurons per layer), trained long | 91.5% | 89.9% |

The left panel is **overfitting** you can see. The boundary grows tiny islands and tendrils to grab individual orange crosses sitting inside the blue region and vice versa. Those points are noise, and next time the noise will be somewhere else. The boundary learned the *particular* 400 points, not the *shape* of the moons.

The middle and right panels are smoother, get **worse** training scores, and do **better** on new points. Accepting some mistakes on the training data was the right call, because those mistakes were on noise.

> 📓 **Notebook rule:** *overfitting is fitting the noise.* The network can't tell which details are real pattern and which are accidents of this particular sample, so if you let it, it will learn both.

The middle and right panels also preview two of the fixes: **stop earlier**, or **limit the capacity**. We'll build the full toolkit next section.

---

### 13.4 Watching it happen over time

We can watch overfitting unfold in a training run. Train the same big network on just **1,000** MNIST digits and, after every epoch, check it on 10,000 **other** digits it never trains on:

![Training vs validation over 150 epochs](figures/fig55_overfit_curves.png)

| | epoch 7 | epoch 150 |
|---|---|---|
| training loss | 0.138 | **0.00009** |
| validation loss | **0.395** (lowest) | 0.648 (+64%) |
| validation accuracy | 88.5% | 89.5% |

The two loss curves tell the story. Training loss falls forever: the network drives it toward zero by becoming ever more certain about digits it has already memorised. **Validation loss bottoms out at epoch 7 and then climbs.** From that point on, extra training makes the network worse at new digits in one important way.

Look at the accuracy panel, though: validation accuracy doesn't fall, it just stalls around 89%. So what's getting worse? **Confidence.** Remember from Section 8 that cross-entropy punishes confident mistakes very heavily. The network keeps getting the same ~11% of validation digits wrong, but it grows more and more **sure** of those wrong answers. That's the "100% sure and wrong" behaviour we saw in Section 12.6, and it's a direct side effect of memorising.

> 📓 **Notebook rule:** *watch the validation loss, not just the validation accuracy.* Loss notices overconfidence long before accuracy does.

---

### 13.5 Three sets, three jobs

So far we've been checking against "digits it didn't train on". To do this properly you need **three** separate piles of data, each with its own job. Back to exams:

| set | exam analogy | used for | touched how often |
|---|---|---|---|
| **training** | practice papers | learning the weights | every step |
| **validation** | mock exams | choosing *everything else*: architecture, learning rate, when to stop | many times |
| **test** | the final exam, sealed until the end | one honest estimate of real-world performance | **once** |

For MNIST the usual split (and Nielsen's) is to hold 10,000 of the 60,000 training digits back as validation, train on 50,000, and leave the official 10,000 test digits sealed until the very end.

(A confession: in Section 12 we printed *test* accuracy after every epoch. That's a mild peek. It didn't matter there because we never used it to choose anything, but from here on, choices get made on validation data only.)

**Why not just use the test set to make choices?** Because every time you use a set to make a decision, you slowly fit *it* too. Here's that effect at its most extreme. Take 1,000 "models" that just **guess randomly** (a fixed random answer per image), score each on 100 test digits, and pick the winner:

![Selection bias: the luckiest of 1,000 random guessers](figures/fig58_peeking.png)

| | accuracy |
|---|---|
| average guesser on the 100-digit check set | 10.1% |
| **the "best" guesser on that check set** | **20%** |
| that same "best" guesser on 9,900 fresh digits | **10.3%** |

The winner looks twice as good as chance. It isn't. It's the one that got **lucky** on those particular 100 digits. On fresh digits it's back to chance.

Real model selection isn't this extreme, since real models have real differences. But the same force is always there: **choose among enough options using the same data, and your best score will be partly luck.** That's why the choosing happens on the validation set, and the test set is opened once, at the end, to find out what you really built.

> 📓 **Notebook rule:** *the test set is a sealed envelope.* Every peek turns it a little more into a validation set, and you lose your only honest number.

---

### 13.6 The best cure: more data

The most reliable fix for memorising is simply **more data**. With more examples there's more real pattern to find and less room to memorise every one. Same network, same recipe, different amounts of training data:

![More data closes the gap](figures/fig56_more_data.png)

| training digits | training accuracy | test accuracy | gap |
|---|---|---|---|
| 500 | 100% | 87.5% | 12.5 |
| 1,000 | 100% | 89.8% | 10.2 |
| 5,000 | 100% | 94.5% | 5.5 |
| 20,000 | 100% | 97.3% | 2.7 |
| 50,000 | 99.8% | **98.2%** | **1.6** |

The network memorises the training set perfectly almost every time. What changes is how much of that memory is *pattern* and how much is *particular*. From 500 to 50,000 digits, the gap shrinks from 12.5 to 1.6 points.

Real data is often expensive, though: you have to collect it, label it and clean it. Part of the next section is about getting the effect of more data **for free**, by creating new training examples from the ones you already have (**data augmentation**).

---

### 📓 Notebook margin: what we now know

$$
\text{generalisation gap} \;=\; \text{training accuracy} - \text{test accuracy}
$$

| idea | what we now know |
|---|---|
| generalisation | performance on data never seen: the only number that matters |
| memorising | a big network hits **100%** on 5,000 images with *random* labels (and 10.7% on test) |
| overfitting | fitting the noise: islands in the boundary, better training score, worse on new data |
| over time | validation loss bottoms out (epoch 7), then climbs as the network grows overconfident |
| train / validation / test | practice papers / mock exams / sealed final exam |
| peeking | the "best" of 1,000 random guessers scores 20% on 100 digits and 10.3% on fresh ones |
| more data | the gap shrinks from 12.5 points (500 digits) to 1.6 (50,000) |

---

### What comes next

We've seen the disease. Now the medicine. **Section 14: Holding the Network Back** covers the standard regularisation toolkit, each piece tested on the same overfitting setup from 13.4 so we can compare them fairly:

- **Early stopping:** keep the weights from the best validation epoch.
- **Weight decay:** a penalty for large weights, and why that smooths the boundary.
- **Dropout:** randomly switch neurons off during training, so no neuron can rely on another.
- **Data augmentation:** small shifts and rotations of each digit to make "new" data for free.

---

*References: Zhang, Bengio, Hardt, Recht & Vinyals (2017), "Understanding deep learning requires rethinking generalization" (the random-labels experiment). Michael Nielsen, *Neural Networks and Deep Learning*, ch. 3 ("Overfitting and regularization": training on 1,000 images, the rising validation cost, the 50,000/10,000 train/validation split, "more training data" as a remedy). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 5 ("Fundamentals of machine learning": generalisation, the random-labels demonstration, training/validation/test sets, information leaks). All code in this series is PyTorch.*
