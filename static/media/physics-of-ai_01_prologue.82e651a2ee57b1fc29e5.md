# Physics of AI

## 1. Prologue: Weights and Biases

![Paul Dirac](figures/paul-dirac-quote.png)

> *Paul Dirac, one of the great physicists of the twentieth century, once wrote that "God is a mathematician of a very high order, and He used very advanced mathematics in constructing the universe." He meant that mathematics isn't just a tool humans use to **describe** the universe. It's woven into the actual structure of reality, at a deep level.*

---

### 1.1 Why "physics"?

Everything in this world can be represented in numbers. A photo is a grid of numbers. A song is a wave of numbers. A movie review is a sequence of words, and we'll see that words can become numbers too. Maybe everything is predictable, and even uncertainty is simply a number we don't know yet, or don't understand yet.

Physics works by finding the smallest pieces of the world, atoms and forces, and building everything else up from them. This series does the same for AI. It focuses on describing the **true, atomic blocks of AI**, and then builds upward one piece at a time, from a single neuron to the Transformer behind today's language models (*"Attention Is All You Need"*).

And the most atomic block of all is surprisingly simple. AI is fundamentally built on two things: **weights and biases**.

---

### 1.2 A weight is just "how much does this matter?"

To start with, a **weight** is nothing but a number that represents how important a certain "thing" is. That "thing" could be anything.

Take a decision everyone makes: **"Should I watch this movie tonight?"**

What are the factors that matter?

- Does it have **great reviews**?
- Is my **favourite actor** in it?
- Is it on the **OTT platform I pay for**?

But how much does each of them matter? Let's say reviews matter to you the most, that's priority no. 1. Second, the actor. And last, whether it's on a platform you already pay for.

To represent everything in mathematical terms, each factor becomes an input, **1 for yes and 0 for no**:

| factor | input |
|---|---|
| great reviews? | $x_1$ |
| favourite actor? | $x_2$ |
| on my subscription? | $x_3$ |

To represent importance, each input gets a weight:

$$
w_1 = 5 \;\;(\text{reviews matter most}) \qquad w_2 = 3 \qquad w_3 = 2
$$

Then take the **weighted sum**, multiplying each input by its weight and adding them up:

$$
\sum_i w_i x_i = 5\cdot\text{reviews} + 3\cdot\text{actor} + 2\cdot\text{subscription}
$$

And let's say I'll only watch it if the sum is **greater than 6**. That's my **threshold**.

---

### 1.3 A few scenarios

| reviews? | actor? | on subscription? | weighted sum | watch? |
|---|---|---|---|---|
| 1 | 0 | 0 | 5 | No |
| 1 | 0 | 1 | 7 | **Yes** |
| 0 | 1 | 1 | 5 | No |
| 1 | 1 | 0 | 8 | **Yes** |
| 1 | 1 | 1 | 10 | **Yes** |

Take the first row:

$$
\sum_i w_i x_i = 1\times 5 + 0\times 3 + 0\times 2 = 5, \qquad 5 \le 6 \;\Rightarrow\; \textbf{No}
$$

Great reviews alone aren't enough. But great reviews *plus* it being on my plan (7) clears the bar. And look at row three: my favourite actor, on my plan, still a no (5), because without good reviews, nothing else quite makes up for it. The weights encode my priorities, and the threshold encodes how picky I am.

Drawn as a picture, it's this:

![One perceptron deciding whether to watch a movie](figures/fig00a_prologue_perceptron.png)

Inputs come in on the left, each travels along a connection with its own weight, they're summed in the circle, and one decision comes out. **That's how a perceptron works!** It was invented by Frank Rosenblatt in 1958, and it's the ancestor of every neuron in every network in this series.

All eight possible movies, sorted by their weighted sum:

![All 8 input combinations and their weighted sums; the bias as a dial](figures/fig00b_prologue_bias.png)

---

### 1.4 Now, the bias

Look at the equation we've been using again. It had a **threshold**, and the threshold controls how easily I say yes:

- ↑ **higher threshold** → fewer chances of a positive result (watching the movie)
- ↓ **lower threshold** → more chances of a positive result

The **bias** we use in modern AI is just the threshold moved to the other side of the inequality:

$$
\sum_i w_i x_i > \text{threshold}
\quad\Longleftrightarrow\quad
\sum_i w_i x_i - \text{threshold} > 0
$$

Call $b = -\text{threshold}$, and the rule becomes:

$$
\text{output} =
\begin{cases}
0 & \text{if } \mathbf{w}\cdot\mathbf{x} + b \le 0 \\
1 & \text{if } \mathbf{w}\cdot\mathbf{x} + b > 0
\end{cases}
$$

Technically, $b \equiv -\text{threshold}$. Same decisions, written a different way:

| | classical (Rosenblatt) | modern (bias form) |
|---|---|---|
| rule | fires if $\sum_i w_i x_i > \text{threshold}$ | fires if $\sum_i w_i x_i + b > 0$ |
| for my movie neuron | threshold $= 6$ | $b = -6$ |

Why bother moving it? Because now the bias is **just another number the neuron owns**, like the weights. Later in this series, networks *learn* their weights and biases automatically, and it's much simpler when every learnable number has the same role: something you add or multiply, then compare with zero.

In simple terms, the bias tells us: **"how easy is it for this neuron to fire, independent of the inputs?"**

- A **very negative bias** = hard to trigger. A **skeptical neuron**.
- A **very positive bias** = fires almost regardless of the input. An **eager neuron**.

The right-hand panel above shows it as a dial. With $b = -10$ or lower, my movie neuron says no to all 8 kinds of movie. At $b = -6$ it watches 3. Near $b = 0$ it watches almost anything.

In PyTorch, the whole perceptron is three lines:

```python
import torch
w = torch.tensor([5., 3., 2.])          # how much each factor matters
b = -6.0                                # how picky I am (threshold 6)
x = torch.tensor([1., 0., 1.])          # great reviews, not my actor, on my plan
watch = (w @ x + b > 0)                 # 5 + 0 + 2 - 6 = 1 > 0  →  True
```

---

### 1.5 The catch

That's it: **weights and biases, the fundamental mathematical model behind AI.** Every network in this series, up to the Transformer, is built out of this one equation, repeated millions of times.

But notice who chose the numbers 5, 3, 2 and −6: **I did.** For movie night that's easy, since I know my own priorities. Now try something harder. Look at a scribbled handwritten "7". You recognise it instantly, but try writing down the weights: which of its 784 pixels matter, and by how much? You can't. Nobody can write those rules down.

So the rest of this series is about one idea: **instead of choosing the weights and biases ourselves, we let the machine learn them from examples.** Section 2 takes the first step: it shows what a neuron needs before it *can* learn.

---

### 📓 Notebook margin: the equation, at the very start

$$
\text{output} = \begin{cases} 1 & \text{if } \sum_i w_i x_i + b > 0 \\ 0 & \text{otherwise} \end{cases}
$$

| idea | what it means |
|---|---|
| weight $w_i$ | how much input $i$ matters (reviews 5, actor 3, subscription 2) |
| weighted sum | $\sum_i w_i x_i$: all the evidence added up |
| threshold | how much evidence is enough (6) |
| bias $b$ | the threshold moved across: $b = -\text{threshold} = -6$ |
| perceptron | weighted sum + bias, then a yes/no decision (Rosenblatt, 1958) |
| the catch | for real problems, nobody can hand-pick the weights, so the machine must learn them |

---

### What comes next

**Section 2: One Neuron** gives the perceptron a smoother decision (the sigmoid instead of a hard yes/no), keeps the same movie-night question (with one new input that pushes *against* watching: work early tomorrow), and shows why the smooth version is what makes learning possible. A few sections later, the weights are learned from 600 past movie nights instead of being set by hand. That's the first step from *writing* rules to *learning* them.

---

*References: P. A. M. Dirac (1963), "The Evolution of the Physicist's Picture of Nature", Scientific American ("God is a mathematician of a very high order, and He used very advanced mathematics in constructing the universe"). Frank Rosenblatt (1958), "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain". Michael Nielsen, *Neural Networks and Deep Learning*, ch. 1 (perceptrons; the bias as a measure of how easy it is to get the perceptron to fire). François Chollet, *Deep Learning with Python*, 3rd ed., ch. 1 (what "learning" means: finding the right weights). All code in this series is PyTorch.*
