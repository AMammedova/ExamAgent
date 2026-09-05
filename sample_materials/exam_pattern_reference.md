# Exam pattern reference (ML + DL final)

Ingest this file as **EXAM_SAMPLES** on the Materials page. It records the question
formats and worked patterns observed in the previous midterm, so the app can calibrate
question style and difficulty before your own past papers are uploaded.

Replace or supplement it with the actual exam PDFs as soon as you have them — real
papers always beat a reconstruction.

---

## Question format A — Assertion and Reason

An Assertion (A) and a Reason (R) are given. Choose:

- **A** — Assertion true, Reason true, Reason explains the Assertion
- **B** — Assertion true, Reason true, Reason does NOT explain the Assertion
- **C** — Assertion true, Reason false
- **D** — Assertion false, Reason true
- **E** — Assertion false, Reason false

The examiner favours the traps. Evaluate the truth of A and of R **separately** first,
then ask whether R is the actual *cause* of A. A true statement is not automatically an
explanation, and a true reason can accompany a false assertion.

Worked example.

> **Assertion:** Reducing the learning rate is an effective remedy for overfitting.
> **Reason:** The learning rate controls the size of each parameter update during
> gradient descent.

The Reason is a correct definition, so R is true. The Assertion is false: the learning
rate governs optimisation dynamics, not the capacity of the hypothesis class; a smaller
learning rate reaches the same overfitted solution more slowly. **Answer: D.**

---

## Question format B — Mathematical calculation

Typical subjects: neural network parameter counting, forward propagation, loss
calculation, backpropagation, gradient and weight/bias updates, tensor dimensions, CNN
output dimensions and parameter counts, receptive fields, attention calculations,
probabilities, distances, classification metrics, regression.

### Worked pattern: 2-2-1 MLP, one gradient descent step

A 2-2-1 multilayer perceptron is trained with binary cross-entropy.

- Input `x = (1, 2)^T`, target `y = 1`, learning rate `eta = 0.1`
- Layer 1 (ReLU): `W1 = [[1, 1], [1, -1]]`, `b1 = [0, 0]^T`
- Layer 2 (sigmoid): `W2 = [1, 2]`, `b2 = -3`

**Forward pass**

```
z1 = W1 x + b1 = [1*1 + 1*2, 1*1 + (-1)*2] = [3, -1]
a1 = ReLU(z1)  = [3, 0]
z  = W2 . a1 + b2 = 1*3 + 2*0 - 3 = 0
y_hat = sigmoid(0) = 0.5
L  = -[y log y_hat + (1-y) log(1-y_hat)] = -log(0.5) = 0.6931
```

**Backward pass**

With a sigmoid output and binary cross-entropy the output gradient collapses:

```
dL/dz   = y_hat - y = 0.5 - 1 = -0.5
dL/dW2  = dL/dz * a1 = [-1.5, 0]
dL/db2  = dL/dz = -0.5
dL/da1  = dL/dz * W2 = [-0.5, -1.0]
dL/dz1  = dL/da1 * 1[z1 > 0] = [-0.5, 0]      # ReLU derivative
dL/dW1  = outer(dL/dz1, x) = [[-0.5, -1.0], [0, 0]]
dL/db1  = [-0.5, 0]
```

**Update** (`w <- w - eta * dL/dw`)

```
W2 <- [1, 2] - 0.1 * [-1.5, 0] = [1.15, 2]
b2 <- -3 - 0.1 * (-0.5) = -2.95
```

**Which hidden unit receives zero gradient, and why?**

Hidden unit 2. Its pre-activation is `z1_2 = -1 <= 0`, so ReLU is inactive there and its
local derivative is 0. The chain rule multiplies by that 0, so no gradient reaches its
incoming weights and they are unchanged by this update. Sustained across all inputs this
is the *dying ReLU* problem.

---

## Question format C — Conceptual reasoning

Asks *why* a mechanism produces an effect. Examples seen:

- Why is a nonlinear activation function necessary?
- Why does removing nonlinear activations collapse an MLP into an affine transformation?
- Why does dropout reduce overfitting?
- How does L2 regularization change the objective?
- Why does the learning rate not fix overfitting?
- Why does batch normalization help optimisation?
- Why does a vanilla RNN suffer from vanishing gradients?
- Why does an LSTM help with long-term dependencies?
- Why is attention useful?
- Why is scaled dot-product attention divided by `sqrt(d_k)`?

Marks go to the **causal chain**, not the definition. Compare:

> Weak: "Dropout helps because it makes the model more robust."
>
> Full marks: "Dropout randomly removes units during training, so the network cannot
> rely on any particular unit being present; this prevents co-adaptation between units,
> reduces effective capacity and therefore reduces overfitting."

---

## Question format D — Architecture interpretation

Given a diagram or a textual description of an architecture, state for each component:
what it computes, what information flows through it, why it exists, what dimensions are
involved, and what breaks if it is removed.

---

## Question format E — Graph interpretation

Training loss versus validation loss; identifying overfitting, underfitting, the correct
early-stopping point, learning-rate pathologies, and what the curve implies about what
the model is learning.

---

## Question format F — Scenario reasoning

> A model performs very well on training data but poorly on validation data.
> What is happening? Why? Which intervention would help? Which intervention would
> **not** directly address the problem?

The final part is where marks are lost. Naming an intervention that does not address the
cause — and saying *why* it does not — is explicitly examined.

---

## Question format G — Comparison

LSTM vs GRU · training vs validation · dropout vs L2 · KNN vs SVM · bagging vs boosting ·
CNN vs fully connected · BERT vs GPT · encoder vs decoder · self-attention vs
cross-attention.

Answer along explicit axes: mechanism, cost, assumptions, failure modes, when each is
preferred.

---

## Question format H — "What happens if..."

What happens if activation functions are removed? Skip connections removed? Padding
changed? Stride increased? Learning rate too large? Bottleneck made smaller? Attention
removed? The LSTM cell state removed?

State the consequence **and** the mechanism that causes it.

---

## CNN calculation patterns

```
output size   = floor((H + 2P - K) / S) + 1
conv params   = (K * K * C_in + 1) * C_out          # +1 is the bias per filter
pooling params= 0                                    # fixed operation, nothing learned
FC params     = in * out + out
receptive field: RF_l = RF_{l-1} + (k_l - 1) * jump_{l-1},   jump_l = jump_{l-1} * s_l
```

Also examined: channels, feature-map resolution, encoder/decoder resolution, skip
connections by **concatenation** (channels increase) versus **element-wise addition**
(shape preserved), and the effect of total stride on the detectability of small objects.

A convolution's parameter count is independent of input resolution because of weight
sharing; only the compute scales.

---

## Transformer calculation and reasoning patterns

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
d_head = d_model / h
attention params = 4 * (d_model^2 + d_model)     # W_Q, W_K, W_V, W_O with biases
FFN params       = d_model*d_ff + d_ff + d_ff*d_model + d_model
```

**Why `sqrt(d_k)`?** The dot product of two vectors with roughly unit-variance components
is a sum of `d_k` terms, so its variance grows with `d_k` and its standard deviation with
`sqrt(d_k)`. Unscaled logits push softmax into a saturated, near one-hot regime where its
Jacobian is almost zero and gradients vanish. Dividing by `sqrt(d_k)` restores a logit
standard deviation near 1. Dividing by `d_k` would over-correct and flatten attention
toward uniform.

Also examined: self-attention versus cross-attention (where Q, K and V come from), causal
masking, residual connections, layer normalisation, the position-wise feed-forward
network, positional encodings (self-attention is permutation-equivariant without them),
BERT (encoder-only, masked LM, bidirectional) versus GPT (decoder-only, next-token,
causal), pretraining versus fine-tuning, and in-context learning.

---

## Recurrent network patterns

```
RNN cell params  = (input + hidden) * hidden + hidden
GRU cell params  = 3 * [(input + hidden) * hidden + hidden]
LSTM cell params = 4 * [(input + hidden) * hidden + hidden]
```

Vanishing gradients: backpropagation through time multiplies `(T - t)` Jacobians
`W_h^T diag(f'(z_k))`, so the magnitude decays or explodes exponentially with sequence
length. The LSTM cell state updates additively, `c_t = f_t * c_{t-1} + i_t * g_t`, so
`dc_t/dc_{t-1} = f_t`; with the forget gate near 1 the gradient travels many steps almost
undamped — the constant error carousel. Gradient clipping addresses **exploding**, not
vanishing, gradients.

---

## Classification metric patterns

```
accuracy    = (TP + TN) / (TP + TN + FP + FN)
precision   = TP / (TP + FP)          # conditions on the prediction
recall      = TP / (TP + FN)          # conditions on the truth
F1          = 2PR / (P + R)           # harmonic mean, not arithmetic
specificity = TN / (TN + FP)
```

State which denominator you used. Precision and recall differ only in what you condition
on, and swapping them is the most common error on this question type.
