# Physics of AI — Part V · Remembering

## 30. The Bottleneck

> *So far every sequence model has read a sequence and produced one number. The problems that made recurrent networks famous, like translation, produce a **whole sequence**: English in, French out, with different lengths and a different word order. The standard design for that, the **encoder–decoder**, uses one LSTM to read and a second to write. Between them there's a single handoff: one fixed-size vector that has to carry everything. This section builds it, gives it a task where we can check every symbol, and measures exactly where that one vector runs out of room. That limit is what attention was invented to remove, and it closes Part V.*

---

### 30.1 Reading, then writing

The design (Sutskever, Vinyals & Le, 2014; Cho et al., 2014) has two halves:

- An **encoder** LSTM reads the input, one symbol at a time, exactly like Section 29. We throw away its outputs and keep only its **final state** $(\mathbf{h}, \mathbf{c})$.
- A **decoder** LSTM starts from that state and **writes** the output one symbol at a time. At each step it predicts the next symbol with a softmax over the vocabulary (Section 8), and that symbol becomes its next input.

![Encoder–decoder: the encoder's final state is the only link to the decoder](figures/fig138_encoder_decoder.png)

```python
class Seq2Seq(nn.Module):
    def __init__(self, H=128, d=32):
        super().__init__()
        self.emb = nn.Embedding(VOC, d)
        self.enc = nn.LSTM(d, H, batch_first=True)     # reader
        self.dec = nn.LSTM(d, H, batch_first=True)     # writer
        self.out = nn.Linear(H, VOC)                   # scores for the next symbol
    def forward(self, src, tgt):
        _, state = self.enc(self.emb(src))             # (h, c) after the last input symbol: THE handoff
        dec_in = torch.cat([BOS_column, tgt[:, :-1]], 1)   # decoder input: <start>, then the true previous symbols
        o, _ = self.dec(self.emb(dec_in), state)
        return self.out(o)                             # one prediction per output position
```

Two details:

- **Teacher forcing.** During training, the decoder is fed the *correct* previous symbol, not its own guess, so all output positions train in parallel and early mistakes don't snowball. At test time there's no correct answer to feed, so it feeds back its own predictions (**greedy decoding**: take the most likely symbol each step).
- **The handoff.** Look at `state`. Whatever the encoder learned from the input, the decoder only ever sees these $2 \times H$ numbers. With $H = 128$, that's **256 numbers** for the whole input, however long it is.

(Both LSTMs use Section 29's forget-gate bias of +1.)

---

### 30.2 A task we can check exactly

Translation would take days on a CPU, and a wrong translation is a matter of opinion. So we use a task where the right answer is unambiguous and every symbol can be checked: **reverse a sequence of digits**.

```
input:   4 5 4 0 5 7 4 5 9 6 3 4 7 0 2 9 2 6 0 2 2 6 9 9 1 7 2 6 4 1
output:  1 4 6 2 7 1 9 9 6 2 2 0 6 2 9 2 0 7 4 3 6 9 5 4 7 5 0 4 5 4
```

Each digit is random, 10 possibilities, so the input can't be compressed. A 30-digit input holds $30 \times \log_2 10 \approx 100$ bits of information, and all of it must pass through the handoff. Training uses lengths 5 to 30, fresh random sequences every batch (so no memorising a fixed dataset), cross-entropy per output symbol, Adam with warm-up and cosine decay (Section 16), and 4,000 to 5,000 steps (about 4 minutes on 2 CPUs for $H = 128$). Testing uses 1,000 new sequences at each length, including lengths **35 and 40**, which training never showed.

---

### 30.3 Where the one vector runs out

Here are four test inputs for the $H = 128$ model (256 numbers of memory):

| length | wanted output | model's output | wrong |
|---|---|---|---|
| 8 | 83206574 | 83206574 | 0 |
| 20 | 63917492799703882109 | 63917492799703882109 | 0 |
| 30 | 1462719962206**29207**436954750454 | 1462719962206**22907**436954750454 | 2 |
| 35 | 30547704249793534993907176276493509 | 30574407274993539493017697276945304 | 19 |

Up to 20 digits it's flawless. At 30, a couple of digits in the middle get swapped: the memory is blurry, not blank. At 35, longer than anything in training, it gets the first few digits and then produces confident nonsense.

Across all 1,000 test sequences per length, for three memory sizes:

![Exact-match and per-symbol accuracy vs input length, for three memory sizes and for copying](figures/fig139_bottleneck_length.png)

| sequences exactly right | 10 digits | 20 | 25 | 30 | 35 (unseen length) |
|---|---|---|---|---|---|
| $H = 64$ (128 numbers) | 93.5% | 28.2% | 3.4% | 0.0% | 0.0% |
| $H = 128$ (256 numbers) | 100% | 99.3% | 92.3% | 50.7% | 0.1% |
| $H = 256$ (512 numbers) | 100% | 100% | 99.3% | 89.9% | 0.0% |

Three things to read off this.

**1. Every model has a breaking point, and it moves with the size of the handoff.** Doubling the memory from 128 to 256 numbers moves the length where exact-match drops below 50% from about 18 digits to about 30. Doubling again keeps 30-digit inputs at 90%. There's no length a fixed vector can handle for free: more input means more to squeeze through the same pipe.

**2. The breaking point comes long before the information limit.** 512 numbers could in principle hold far more than 100 bits. But the network has to *learn* an encoding using an LSTM's gated, squashed updates, and in practice it packs much less than the theoretical maximum. The bottleneck is a learning problem as much as a storage problem.

**3. Past the training lengths, everything fails, even the biggest model.** At 35 digits, exact match is 0% for all three. That's partly the bottleneck and partly **generalisation**: the models never saw a sequence that long, and they learned the patterns of the lengths they were shown, not a general "reverse" procedure. (Honest caveat: our measurement can't cleanly separate those two causes at 35 and 40.)

This matches what researchers found with real translation. Cho et al. (2014) reported that encoder–decoder translation quality **dropped steadily as sentences got longer**, and pointed to the fixed-length vector as the likely cause. It was the direct motivation for attention.

---

### 30.4 Which digits go wrong?

Break the 30-digit results down by output position:

![Accuracy at each output position, for inputs of length 30](figures/fig140_position_accuracy.png)

| accuracy at output position | 1st | 5th | 10th | 20th | 30th (last) |
|---|---|---|---|---|---|
| reverse, $H = 64$ | 93.0% | 67.8% | 45.2% | 28.1% | 15.5% |
| reverse, $H = 128$ | 98.8% | 94.2% | 89.5% | 86.4% | 74.3% |
| reverse, $H = 256$ | 100% | 99.2% | 98.3% | 98.9% | 97.6% |

Accuracy falls steadily as the decoder writes. For reversal, the last digits written are the **first** digits read: the oldest information in the state, with the most updates layered on top of it. The decoder also has to keep track of where it is, with every step of its own state drifting a little further.

---

### 30.5 Order matters: reverse vs copy

Now train the same $H = 128$ model to **copy** instead of reverse: same inputs, same output digits, just in the original order. You might expect it to be no harder, since it's the same information.

| $H = 128$, exactly right | 15 digits | 20 | 25 | 30 |
|---|---|---|---|---|
| reverse | 99.9% | 99.3% | 92.3% | **50.7%** |
| copy | 94.9% | 73.1% | 26.1% | **1.4%** |

**Copying is much harder.** When copying, the first digit written is the first digit read, which is 30 steps back through the encoder. Reversal puts each early output right next to the input that produced it: the last digit read is the first one written, with only one step between them. Short paths are easier to learn than long ones (Sections 28–29 again).

Sutskever and colleagues found exactly this in real translation, and turned it into a famous trick: **feed the source sentence in reverse order**. The first words of the French output are then close to the first words of the English input, which the encoder has just read. That simple change raised their English-to-French BLEU translation score from 25.9 to 30.6. It works, but it's a patch. It shortens *some* paths by lengthening others, and it can't help when the needed input could be anywhere.

> 📓 **Notebook rule:** *a fixed-size vector is a fixed-size pipe.* However long the input, everything the decoder will ever know has to fit through $2H$ numbers, and the network is far from perfect at packing them. Longer inputs, or inputs whose useful parts are far from where they're needed, fail first.

---

### 30.6 The fix, in one sentence

Look at the encoder again. As it reads, it produces a hidden state **at every input position**, $\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_T$, and each one is a good summary of the input *around that position*. The encoder–decoder throws them all away except the last.

What if the decoder could keep all of them, and at each output step **look back** at whichever input positions it needs, reversing digit 17 by looking directly at input position 14? No single vector would have to hold everything, and every path from input to output would be one step long.

That's **attention** (Bahdanau, Cho & Bengio, 2014).

---

### 📓 Notebook margin: Part V in one table

$$
\text{encoder: } (\mathbf{h}_T, \mathbf{c}_T) = \text{LSTM}(\mathbf{x}_1, \dots, \mathbf{x}_T)
\qquad
\text{decoder: } P(y_t \mid y_{<t}) = \text{softmax}\big(W\,\text{LSTM}(y_{t-1};\ \text{state starts at } (\mathbf{h}_T, \mathbf{c}_T))\big)
$$

| section | idea | biggest measured effect |
|---|---|---|
| 26 | bag of words | 90.26% test; shuffled words score identically |
| 27 | embeddings, word2vec | 500 labels: 85.4% (pre-trained) vs 81.0% (bag of words) |
| 28 | RNN | gradient 18,080× smaller at the first of 200 words; memory fails by 40 |
| 29 | LSTM, forget bias | 80-word memory solved in 50–150 steps; IMDB 89.9% |
| 30 | encoder–decoder | 30-digit reversal: 0% (128 numbers) → 51% (256) → 90% (512); copying 1.4% |

---

## End of Part V

Part V taught networks to read in order. **Embeddings** gave words meaning (27). **Recurrence** gave the network a memory (28), and **gates** made that memory last (29). The **encoder–decoder** let a network write sequences as well as read them (30).

But every one of these designs moves information through a **chain**: word 1's influence reaches word 200 only by passing through 199 updates, and it reaches the decoder only through one final vector. The measurements all say the same thing: chains leak, and single vectors overflow. Recurrent models are also **slow**, because step 200 can't start until step 199 has finished.

**Part VI: Attention** removes the chain. **Section 31: Looking Back** adds attention to this exact encoder–decoder and task. At every output step the decoder scores all encoder states, turns the scores into weights with a softmax, and reads a weighted average. We'll see whether 30-digit reversal becomes easy, look at the attention weights (they should form a clean diagonal), and test the lengths that broke every model here.

---

*References: Ilya Sutskever, Oriol Vinyals & Quoc Le (2014), "Sequence to Sequence Learning with Neural Networks" (the LSTM encoder–decoder; reversing the source sentence: BLEU 25.9 → 30.6). Kyunghyun Cho et al. (2014), "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation"; Kyunghyun Cho, Bart van Merriënboer, Dzmitry Bahdanau & Yoshua Bengio (2014), "On the Properties of Neural Machine Translation: Encoder–Decoder Approaches" (quality drops with sentence length). Dzmitry Bahdanau, Kyunghyun Cho & Yoshua Bengio (2014), "Neural Machine Translation by Jointly Learning to Align and Translate" (the fixed-length bottleneck, and attention as the fix). Ronald Williams & David Zipser (1989), "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks" (teacher forcing). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 15 (sequence-to-sequence learning; the RNN encoder–decoder and its limits). All code in this series is PyTorch.*
