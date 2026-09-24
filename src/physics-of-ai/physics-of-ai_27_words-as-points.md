# Physics of AI — Part V · Remembering

## 27. Words as Points in Space

> *Section 26 left each word as a one-hot vector: 20,000 numbers, one of them 1, every word equally far from every other. This section gives each word a short list of numbers, an **embedding**, and lets training decide what those numbers are. We'll see that the numbers depend entirely on the job they were trained for. Trained on sentiment, the embedding collapses to a single "how positive" number. Trained to predict neighbouring words, it discovers genders, genres, numbers and synonyms, and it helps most when labelled data is scarce. That second idea, learning from plain text before learning the task, is the seed of every modern language model.*

---

### 27.1 One learned row per word

Section 26.3 showed that multiplying a one-hot vector by a matrix $E$ just picks out one row: $\text{onehot}(i)\,E = E[i]$. So give $E$ only $d$ columns (16 instead of 20,000), and each word gets its own **short, dense, learned vector**:

![Embedding lookup: a one-hot vector times the table is just one row](figures/fig125_lookup.png)

```python
emb = nn.Embedding(20002, 16)        # a 20,002 × 16 table of weights, trained like any other layer
emb(torch.tensor([5]))               # → row 5: the 16 numbers for "excellent"
```

That's the whole idea. An embedding isn't a new kind of layer. It's the first dense layer on a one-hot input, implemented as a lookup (Section 26.3). Its rows start random and are adjusted by backprop (Section 11) like any other weight.

For a review, look up every word's row and **average** them into one 16-number summary, then put a linear layer on top:

```python
class EmbMean(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.emb  = nn.EmbeddingBag(V, d, mode="mean")    # look up every word, average the rows
        self.head = nn.Linear(d, 1)
    def forward(self, word_ids, offsets):
        return self.head(self.emb(word_ids, offsets))[:, 0]
```

| trained on all 39,582 labelled reviews | validation accuracy |
|---|---|
| bag of words, linear (Section 26) | 89.6% |
| **embedding d = 16, averaged, linear** | **90.5%** |
| bag of words + word pairs, MLP (Section 26's best) | 91.2% |

A little better than the plain bag of words, not as good as word pairs. That makes sense, because **averaging is still a bag**. Shuffle the words and the average doesn't change. So embeddings on their own don't fix Section 26's blindness to order. What they change is **how words relate to each other**. Let's look.

---

### 27.2 What a sentiment-trained embedding learns

Train the same model with **d = 2**, so every word becomes a point we can plot directly. It still gets **90.0%**, almost as good as d = 16:

![A 2-number sentiment embedding: every word on one line](figures/fig126_embedding_2d.png)

The 3,000 most common words don't spread across the plane. They lie on **one line**, ordered from *worst, waste, awful* through *the, movie, hitchcock* to *excellent, perfect, 7*. Measured, **99.7%** of the spread is along a single direction. Even with d = 16, one direction holds **97.3%** of it.

That's the job speaking. Sentiment classification needs one fact per word: *how positive is it?* The average over the review is then projected onto the classifier's weight vector (the arrow), and anything perpendicular to the arrow is ignored, so training never shapes it. The model spent 16 numbers per word to learn **one**.

So in this embedding, "good" and "bad" point in **opposite** directions (cosine similarity −0.97). But its nearest neighbours of "actor" are *estranged, mason's, disappears, janet*: noise. Nothing in sentiment training required actors to sit near actresses.

(A detail worth spotting: "10" sits near the neutral middle, while "7" and "8" are strongly positive. Reviews write "10/10" but also "1/10", and our tokenizer splits both into separate numbers, so "10" turns up in the worst reviews too.)

> 📓 **Notebook rule:** *an embedding learns what its training signal rewards, and nothing else.* A sentiment embedding is a positivity scale.

---

### 27.3 Learning words from their neighbours: word2vec

Here's an idea from linguistics, J. R. Firth, 1957: *"You shall know a word by the company it keeps."* Words that appear in similar contexts tend to mean similar things. "Superb" and "outstanding" both show up next to "performance", "film", "absolutely". **Neighbouring words are free labels**, and every sentence ever written supplies them.

**word2vec** (Mikolov et al., 2013) turns that into a training task. In the **skip-gram** version, the model is shown a word and trained to recognise the words that appear within a few positions of it.

- **Positive examples**: a real (word, neighbour) pair from the text, like (*superb*, *performance*).
- **Negative examples**: the same word paired with 5 random words from the vocabulary, like (*superb*, *tax*).

Each word gets two vectors, $\mathbf{v}$ (when it's the centre) and $\mathbf{u}$ (when it's the neighbour). The score of a pair is their dot product, pushed up for real pairs and down for random ones:

$$
\text{loss} = -\log\sigma(\mathbf{u}_{\text{neighbour}}\cdot\mathbf{v}_{\text{word}}) \;-\; \sum_{k=1}^{5} \log\sigma(-\mathbf{u}_{\text{random}_k}\cdot\mathbf{v}_{\text{word}})
$$

That's Section 8's cross-entropy for a yes/no question, asked six times per pair. Here it is in PyTorch:

```python
Win, Wout = nn.Embedding(V, 100), nn.Embedding(V, 100)            # v (centre) and u (neighbour) tables
for centre, neighbour in pairs_in_batches:                        # pairs within ±5 words of each other
    negs = torch.multinomial(noise_dist, len(centre) * 5, replacement=True).view(-1, 5)
    v   = Win(centre)                                             # (B, 100)
    pos = (v * Wout(neighbour)).sum(1)                            # real pairs: push the score up
    neg = torch.bmm(Wout(negs), v[:, :, None])[:, :, 0]           # random pairs: push the scores down
    loss = -(F.logsigmoid(pos).mean() + F.logsigmoid(-neg).sum(1).mean())
```

(Two standard tricks from the paper: very frequent words like "the" are randomly dropped during pair-making, and negatives are drawn with probability ∝ frequency$^{0.75}$.)

I trained it on the **text of the 39,582 training reviews, with no sentiment labels at all**: 100 dimensions, about 25.5 million word pairs per epoch, 3 epochs, 2.5 minutes each on 2 CPUs. Then look up each word's nearest neighbours by cosine similarity:

| word | nearest neighbours in word2vec |
|---|---|
| excellent | outstanding, superb, terrific, fantastic, wonderful, exceptional, great |
| boring | dull, pointless, uneventful, tedious, uninteresting, unoriginal |
| funny | hilarious, quotable, amusing, humorous, comical, witty, **unintentionally** |
| husband | wife, boyfriend, husband's, father, unfaithful, estranged, spouse |
| hitchcock | hitchcock's, alfred, vertigo, palma, … |
| disney | disney's, walt, pixar, cartoon, talespin, dreamworks |
| 8 | 10, 7, 6, 4, 5, 3, 9 |

From nothing but "which words appear near which", it has found synonyms, a director's first name and his films (and Brian De **Palma**, often compared to him), rival studios, and the numbers. "Unintentionally funny" is a common enough phrase that "unintentionally" became a neighbour of "funny".

![word2vec space: words cluster by meaning; the same word pairs in two embeddings](figures/fig127_word2vec_space.png)

Projected to 2-D (left), the groups mostly separate: praise, criticism, family, film jobs and numbers each form their own cluster, and horror words sit close to criticism ("gore" and "slasher" next to "boring" is a fair summary of many reviews). Two dimensions can't show everything in 100, but it's clearly a map of meaning, learned without a single label.

But look at the right panel. In word2vec, **"good" and "bad" are close** (similarity 0.71). In fact "bad" is the third-nearest neighbour of "good", after "decent" and "nice". It makes sense once you think about contexts: "the acting was **good**" and "the acting was **bad**" share every neighbour. Opposites are used in the same places, so the distributional idea puts them together. That's a real limitation: word2vec knows "good" and "bad" are the **same kind of word**, but not that they're opposite in the way sentiment cares about.

---

### 27.4 king − man + woman = ?

The most famous word2vec result is that directions in the space carry meaning: $\text{king} - \text{man} + \text{woman} \approx \text{queen}$. Let's test it honestly on our small model (trained on movie reviews only), with 16 analogies. Here's a selection; the rest (husband→wife, boy→girl, father→mother, three→four) are all correct:

| analogy | top answer |
|---|---|
| man : woman :: king : ? | **queen** ✓ |
| man : woman :: actor : ? | **actress** ✓ |
| he : she :: his : ? | **her** ✓ |
| good : better :: bad : ? | **worse** ✓ |
| good : best :: bad : ? | **worst** ✓ |
| big : bigger :: small : ? | **smaller** ✓ |
| go : went :: see : ? | **saw** ✓ |
| comedy : funny :: horror : ? | **scary** ✓ |
| man : woman :: brother : ? | daughter (sister is 2nd) |
| france : paris :: italy : ? | *t'aime* ✗ |
| walk : walked :: play : ? | playing ✗ |
| actor : actors :: film : ? | movie ✗ |

Overall **12 of 16** correct as the top answer and 13 in the top 5. The failure on France is funny and instructive: in movie reviews "paris" appears mostly in the film title *Paris, je t'aime*, so the model learned the film, not the city.

Two honest caveats that most demos skip:

- **The standard test excludes the three input words** from the answers. Without that rule the top answer is often just the input: "king − man + woman" returns "king". Only **7 of 16** are correct without the exclusion.
- **Some "analogies" are just neighbours.** In 4 of the 16, the correct answer is already the nearest neighbour of the third word (for example "her" is simply nearest to "his"), so the arithmetic isn't doing much work.

The directions are real, but they're weaker and noisier than the famous example suggests.

---

### 27.5 Why this matters: learning before labels

Labels are expensive, because someone has to read each review and mark it. Text is cheap. So here's the practical test. Take the classifier from 27.1, give it only a **few hundred labelled reviews**, and compare three starting points:

- the bag of words (Section 26)
- an embedding learned from scratch on the labelled reviews
- the **word2vec** embedding, learned from unlabelled text and then **frozen**, with a small MLP (100 → 64 → 1) on top. It gets the MLP because its vectors can't adapt, so the head has to do a little more of the work

![Validation accuracy vs number of labelled reviews](figures/fig128_labelled_data.png)

| labelled reviews | bag of words | embedding from scratch | **word2vec (frozen)** |
|---|---|---|---|
| 500 | 81.0% | 79.4% | **85.4%** |
| 2,000 | 85.9% | 85.6% | **86.4%** |
| 39,582 (all) | 89.6% | **90.5%** | 87.1% |

(At 500 and 2,000 reviews each number is the average over 3 different random subsets. The error bars show the range. The full-data row is one run.)

**With 500 labels, pre-trained embeddings win by more than 4 points.** The classifier doesn't have to learn from 500 reviews that "superb" and "outstanding" are similar, because word2vec already put them together from 9 million words of unlabelled text. With 500 reviews, a from-scratch embedding has seen most words only a handful of times.

**With all the labels, the frozen word2vec comes last.** That's 27.3's limitation showing up: word2vec put "good" next to "bad", and a frozen embedding can't move them apart. Let training adjust the word2vec vectors too (**fine-tuning**) and the full-data score rises to **90.7%**, the best embedding result here.

This is the pattern behind modern NLP: **pre-train** on huge amounts of unlabelled text, then **fine-tune** on the small labelled task. BERT and GPT do the same thing with Transformers instead of a lookup table, and with "predict the missing or next word" instead of "predict the neighbours". The idea is the same one we just ran in 7 minutes on a CPU.

> 📓 **Notebook rule:** *plain text is free supervision.* Predicting a word's neighbours teaches meaning without a single label, and that knowledge transfers to tasks where labels are scarce.

---

### 27.6 Still a bag

None of these models beat Section 26's word-pairs model (91.2% validation). The best embedding model reached 90.7%. So the test set **stays closed** this time: our standing test result is still Section 26's **90.26%**.

The reason is the averaging step. Whether a word is a one-hot slot or a 100-number vector, averaging over the review throws the order away. "Expected great but it was terrible" and "expected terrible but it was great" still get identical scores. Embeddings changed **what a word is**. They didn't change **how the words are combined**.

To combine them *in order*, the model needs to read word 1, then word 2, then word 3, carrying something forward as it goes: a **memory**.

---

### 📓 Notebook margin: the equation so far

$$
\text{word } i \;\mapsto\; \mathbf{e}_i = E[i] \in \mathbb{R}^{d}
\qquad
\text{review} \;\mapsto\; \bar{\mathbf{e}} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{e}_{w_t}
\qquad
P(\text{positive}) = \sigma(\mathbf{w}\cdot\bar{\mathbf{e}} + b)
$$

$$
\text{word2vec: }\; \max\; \log\sigma(\mathbf{u}_{\text{neighbour}}\cdot\mathbf{v}_{\text{word}}) + \sum_{k}\log\sigma(-\mathbf{u}_{\text{random}_k}\cdot\mathbf{v}_{\text{word}})
$$

| idea | what we now know |
|---|---|
| embedding | a learned row per word; the same thing as a dense layer on one-hot |
| sentiment embedding | collapses to one direction (99.7% of the spread at d = 2); good · bad = −0.97 |
| word2vec | trained on neighbouring words, no labels; finds synonyms, genres, family, numbers |
| its blind spot | antonyms share contexts: good · bad = +0.71 |
| analogies | 12 / 16 top-1, but only 7 / 16 without excluding the input words |
| pre-training | 500 labels: 85.4% (word2vec) vs 81.0% (bag of words) |
| still a bag | averaging ignores order; best 90.7% val, below Section 26's word pairs |

---

### What comes next

**Section 28: Reading in Order** builds the first network with a memory, the **recurrent neural network (RNN)**. It reads a review one embedding at a time and updates a hidden state $\mathbf{h}_t = \tanh(W\mathbf{h}_{t-1} + U\mathbf{e}_t + \mathbf{b})$, the same neuron from Section 2 but fed its own previous output. We'll build it from scratch, check that it finally tells "great then terrible" from "terrible then great", and then discover its fatal flaw: gradients through 200 time steps behave exactly like gradients through 200 layers (Sections 15 and 20).

---

*References: J. R. Firth (1957), "A synopsis of linguistic theory 1930–1955" ("You shall know a word by the company it keeps"). Tomas Mikolov, Kai Chen, Greg Corrado & Jeffrey Dean (2013), "Efficient Estimation of Word Representations in Vector Space"; Tomas Mikolov et al. (2013), "Distributed Representations of Words and Phrases and their Compositionality" (skip-gram, negative sampling, subsampling, the 0.75 power, analogies). Omer Levy & Yoav Goldberg (2014), "Linguistic Regularities in Sparse and Explicit Word Representations" (analogy evaluation). Tal Linzen (2016), "Issues in evaluating semantic spaces using word analogies" (excluding input words; answers that are already nearest neighbours). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 14 (word embeddings: learned vs pre-trained, the Embedding layer). Jacob Devlin et al. (2019), "BERT" (pre-train, then fine-tune). All code in this series is PyTorch.*
