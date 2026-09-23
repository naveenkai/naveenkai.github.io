# Physics of AI — Part V · Remembering

## 26. Text as Numbers

> *Every input so far has been a grid of numbers with a fixed size: 784 pixels, or 3 × 32 × 32. Language is different. A review can be 7 words or 2,000. Its "pixels" are words, and there's no obvious number for "terrible". This section turns text into tensors in the simplest possible way, trains a classifier that gets 90% right, and then shows exactly what that simple way throws away. That missing piece, **word order**, is what the rest of this series is about.*

---

### 26.1 The data: 50,000 movie reviews

The **IMDB** dataset (Maas et al., 2011) is 50,000 movie reviews, each labelled **positive** or **negative**, half and half. Chollet uses it for his text chapters, and so will we. The task is **sentiment classification**: read a review, output the probability it's positive.

```
"Superb film with no actual spoken dialogue, which enhances the level of suspense.
 The whole approach gives a completely different twist to a war film. Well worth watching again…"   → positive
```

![Review lengths; Zipf's law for word frequencies](figures/fig121_lengths_zipf.png)

The left plot shows the first problem. The median review is 175 words, but they range from **7** to **2,150**. There's no fixed-size grid here. The input is a **sequence**, and sequences come in every length.

(Honest data notes. The official Stanford download was blocked in my environment, so I used a public GitHub copy of the same 50,000 reviews. That copy has the original train/test halves merged, so I removed 418 duplicate reviews and made my own split: **39,582 training / 5,000 validation / 5,000 test**, with seed 0. The test set is opened once, at the end.)

---

### 26.2 Step 1: split text into tokens

A **token** is the unit we'll turn into a number. Here, a token is a word: lowercase everything, remove the HTML line breaks, and keep runs of letters, digits and apostrophes (plus `!` and `?`, which carry feeling):

```python
def tokenize(s):
    s = s.lower().replace("<br />", " ")
    return re.findall(r"[a-z0-9']+|[!?]", s)

tokenize("Not bad at all. I loved it!")   # ['not', 'bad', 'at', 'all', 'i', 'loved', 'it', '!']
```

The training reviews contain **9.2 million** tokens but only **109,883 distinct words**. The right-hand plot above is one of the most reliable facts about language, **Zipf's law**: a word's frequency is roughly proportional to 1 / its rank. "The" alone is 5.7% of all tokens. The top 100 words cover **50%** of the text, and the top 20,000 cover **97%**. Meanwhile **45,458 words** appear exactly once (typos, names, rare words).

So we keep a **vocabulary** of the 20,000 most frequent training words and map everything else to a single `[UNK]` ("unknown") token:

```python
counts = Counter(w for review in train_tokens for w in review)          # training reviews only
itos = ["[PAD]", "[UNK]"] + [w for w, _ in counts.most_common(20000)]   # id → word
stoi = {w: i for i, w in enumerate(itos)}                               # word → id
encode = lambda toks: [stoi.get(w, 1) for w in toks]                    # unknown words → 1
```

Build the vocabulary from **training data only**, just as Section 4 standardised using training statistics only. Otherwise information from the validation and test reviews leaks in.

---

### 26.3 Step 2: ids → vectors

Now each word is an integer, but an integer is a bad input. id 76 ("bad") isn't "more" than id 21 ("not"). The ids are sorted by frequency, not meaning. We need a representation where no word is secretly bigger than another.

The standard answer is **one-hot encoding**: word number $i$ becomes a vector of 20,000 zeros with a single 1 in position $i$. It's exactly how we encoded digit **labels** back in Section 8.

![Text → tokens → ids → multi-hot vector](figures/fig122_pipeline.png)

One-hot vectors have two properties worth pausing on.

**1. Every word is equally far from every other word.** The distance between any two one-hot vectors is $\sqrt{2}$. So "good" is exactly as far from "great" as it is from "terrible". The representation knows nothing about meaning, and whatever the network learns about similar words it has to learn separately for each one.

**2. Multiplying a one-hot vector by a weight matrix just picks out one row.** If $W$ is a $20{,}000 \times d$ matrix, then $\text{onehot}(i)\,W = W[i]$. There's no real multiplication, only a lookup:

```python
W = torch.randn(10, 4); ids = torch.tensor([3, 7, 3])
torch.equal(F.one_hot(ids, 10).float() @ W, W[ids])     # True
```

That second fact is the key to Section 27. The first dense layer on a one-hot input is secretly **a table with one learned row per word**. PyTorch has a layer that does the lookup directly without building the huge one-hot vector: `nn.Embedding`, and its summing cousin `nn.EmbeddingBag`, which we use below.

---

### 26.4 The bag of words

Now for a whole review. The simplest idea is to throw every word's one-hot vector into a bag and record **which words appear**: a 20,000-long **multi-hot** vector with a 1 for each word present (bottom row of the figure above). A 200-word review lights up about 124 slots on average (words repeat), and the other ~19,880 are 0.

Put a linear layer on top (Section 5) and that's logistic regression on words. Every word gets **one weight**, and the review's score is the sum of the weights of the words it contains:

```python
class BagLinear(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.w = nn.EmbeddingBag(V, 1, mode="sum")      # one weight per word, summed over the bag
        self.b = nn.Parameter(torch.zeros(1))
    def forward(self, word_ids, offsets):              # word_ids: every review's unique ids, concatenated
        return self.w(word_ids, offsets)[:, 0] + self.b
```

`EmbeddingBag` computes exactly `multi_hot @ W`, but by looking up and summing rows instead of multiplying by 19,880 zeros. That makes it fast: each model below trains in about 10 seconds on 2 CPUs.

| bag of words, 20,000-word vocabulary | best validation accuracy |
|---|---|
| linear (logistic regression) | 89.9% |
| MLP: 20,000 → 16 → 1, dropout 0.5 (Chollet's baseline) | **90.1%** |

**90% from just knowing which words appear.** Sentiment is mostly carried by individual words, and a single linear layer reads them. Because the model is linear, we can open it and read every weight:

![The words with the most negative and most positive weights](figures/fig123_word_weights.png)

Most of this is what you'd expect: *worst, waste, awful, boring* against *excellent, superb, perfect, flawless*. But look closer, because it's Section 25's lesson again: **the model learns whatever predicts the label**.

- **"7" is the single most positive word**, and "8" is fifth. Reviewers write scores like "7/10". Of training reviews containing "7", **72%** are positive. It's a real clue, but it's about the rating, not the language.
- **"mst3k"** is strongly negative. It stands for *Mystery Science Theater 3000*, a show that mocks bad films, so people mention it when a film is bad.
- **"olds"** is strongly negative, almost certainly from "year-olds" (our tokenizer splits on the hyphen), as in "maybe 12-year-olds will enjoy this".
- **"good" is barely positive** (weight 0.19, while "great" is 0.66). The word appears in **14,925** training reviews, and **49.8%** of them are positive. It turns up in phrases like "good actors wasted" or "no good" about as often as in "a good film". On its own, "good" carries almost no information.
- "the" gets a weight of 0.007: frequent and meaningless.

---

### 26.5 What the bag throws away

Here's Section 22's experiment again, now for text. There, we shuffled the **pixels** of every image and the dense network didn't notice. Here, shuffle the **words** of every validation review:

```
original:  "i expected it to be terrible but it was great"
shuffled:  "great it be but to was i terrible expected it"
```

![Validation accuracy with and without word shuffling](figures/fig124_shuffle_words.png)

| model | original reviews | words shuffled |
|---|---|---|
| bag of words, linear | 89.86% | **89.86%** |
| bag of words, MLP | 90.06% | **90.06%** |

Identical to the last digit, and this time it isn't even an experiment, it's **guaranteed**. Shuffling changes the order, and the multi-hot vector has no order in it. The bag of words is the text version of the dense network on shuffled pixels: it is **blind to arrangement** by construction.

How much does that matter? Ask the trained MLP about sentences where the order changes the meaning:

| sentence | bag-of-words MLP: P(positive) |
|---|---|
| "i expected it to be great but it was terrible" | 36.6% |
| "i expected it to be terrible but it was great" | **36.6%** |
| "this movie was good" | 48.0% |
| "this movie was not good" | 41.2% |

The first two sentences contain exactly the same words, so they **must** get the same score. The model can't tell a pleasant surprise from a disappointment. And "not" barely dents "good", because the model has no way to know that "not" applies to the *next* word.

---

### 26.6 A little bit of order: word pairs

A cheap fix is to add **pairs of neighbouring words** to the bag: "not good", "was great", "waste of". These are called **bigrams** (single words are unigrams). Pairs from "it was great" are "it was" and "was great". Keep the 20,000 most frequent unigrams and bigrams together (12,688 of them turn out to be pairs), and train the same models:

```python
def ngrams(toks):
    return toks + [a + " " + b for a, b in zip(toks, toks[1:])]   # words, plus each neighbouring pair
```

| model (validation) | words only | **words + pairs** | words + pairs, then shuffled |
|---|---|---|---|
| linear | 89.9% | 90.6% | 86.6% |
| MLP (16 hidden) | 90.1% | **91.2%** | 87.6% |

Pairs add about a point. Now shuffling **hurts**, by 3.6 points, because the pairs of a shuffled review are nonsense. The model has started to use order. And the sentences:

| sentence | words only | **words + pairs** |
|---|---|---|
| "i expected it to be great but it was terrible" | 36.6% | **17.5%** |
| "i expected it to be terrible but it was great" | 36.6% | **56.4%** |
| "this movie was not good" | 41.2% | **17.2%** |
| "this movie was not bad" | 18.4% | 20.6% |

The model now tells the two "expected" sentences apart, thanks to the pairs "was terrible" and "was great". The pair "not good" gets its own strongly negative weight (−0.54 in the linear version).

But look at the last row. **"Not bad" is still judged negative**, and here the model is simply faithful to its data: of the 223 training reviews containing "not bad", only **36%** are positive. People write "not bad" about films they found mediocre. The model isn't wrong about how the phrase is used, just about how we meant it in that sentence.

Pairs are a patch, not a cure. They see two words of context and nothing more. "Not **exactly** good" or "I **didn't** think it was **good**" put more than one word between the negation and what it negates. You could add triples, then quadruples, but the vocabulary explodes (Zipf again: most long word sequences appear once). We need a model that reads a sequence **in order** and remembers what it has seen.

---

### 26.7 Opening the envelope

The best validation model (words + pairs, MLP) goes to the test set, once:

> **Test accuracy: 90.26%** (validation was 91.22%)

That's about a point below validation. With several models and 6 epochs each compared on the same 5,000 validation reviews, a small optimistic bias is expected (Section 13's warning about tuning on validation). It's in line with Chollet's bag-of-bigrams result on this dataset (about 90%).

This number is now the **baseline to beat**. Everything in Part V has to earn its complexity against 90.3% from a bag of words and pairs that trains in 15 seconds. As Chollet points out, on short-ish sentiment tasks like this one, that's a surprisingly hard baseline.

> 📓 **Notebook rule:** *a bag of words is a dense network for text: it knows **what** is there, never **where**.* It's fast, strong, and blind to order by construction. Every sequence model after this one is an attempt to read the order too.

---

### 📓 Notebook margin: the equation so far

$$
\text{review} \xrightarrow{\text{tokenize}} (w_1, \dots, w_T)
\xrightarrow{\text{vocabulary}} (i_1, \dots, i_T)
\xrightarrow{\text{bag}} \mathbf{x} \in \{0,1\}^{20{,}000}
\qquad
P(\text{positive}) = \sigma\!\Big(b + \!\!\sum_{\text{word } i \,\in\, \text{review}}\!\! w_i\Big)
$$

| idea | what we now know |
|---|---|
| text is a sequence | lengths 7 to 2,150 words; no fixed grid |
| tokens, vocabulary | 109,883 distinct words; top 20,000 cover 97% (Zipf's law) |
| one-hot | every word equally far from every other ($\sqrt{2}$); times a matrix = a row lookup |
| bag of words | 90.1% validation; its weights read like a dictionary of sentiment, plus shortcuts ("7", "mst3k") |
| order blindness | shuffled words give exactly the same score, by construction |
| word pairs | 91.2% validation; shuffling now costs 3.6 points |
| test (opened once) | **90.26%**: the baseline for Part V |

---

### What comes next

One-hot vectors treat "great" and "excellent" as strangers. They're as different as "great" and "boring". Yet the bag-of-words model learned nearly the same weight for both (0.66 and 0.83), **separately**, from thousands of reviews each. What if words started out close to their synonyms?

**Section 27: Words as Points in Space** replaces the one-hot vector with a short, learned vector for each word, an **embedding**. It's the lookup table from 26.3, with $d$ = 16 or 64 columns instead of 20,000, trained so that words used in similar ways end up near each other. We'll train embeddings, look at which words land next to "excellent" and "boring", test the famous "king − man + woman" arithmetic honestly, and check whether embeddings alone beat our 90.3%.

---

*References: Andrew Maas et al. (2011), "Learning Word Vectors for Sentiment Analysis" (the IMDB dataset). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 14 ("Text classification": tokenization, vocabulary indexing, one-hot and multi-hot encoding, the bag-of-words and bag-of-bigrams baselines on IMDB). George Kingsley Zipf (1935/1949), *The Psycho-Biology of Language* / *Human Behavior and the Principle of Least Effort* (Zipf's law). All code in this series is PyTorch.*
