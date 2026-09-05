"""Seed question bank.

These are hand-written in the style of the university exam samples: reasoning
first, definitions never. They bootstrap the app on day one; the generators and
(when configured) the LLM produce the rest.

Assertion-Reason items always carry the three internal truth flags so the
evaluator can derive the correct option mechanically.
"""
from __future__ import annotations

from typing import Any

#: Standard option set for every Assertion-Reason question.
AR_OPTIONS: list[dict[str, str]] = [
    {"key": "A", "text": "Assertion true, Reason true, and the Reason correctly explains the Assertion"},
    {"key": "B", "text": "Assertion true, Reason true, but the Reason does NOT explain the Assertion"},
    {"key": "C", "text": "Assertion true, Reason false"},
    {"key": "D", "text": "Assertion false, Reason true"},
    {"key": "E", "text": "Assertion false, Reason false"},
]


def ar_key(a_true: bool, r_true: bool, explains: bool) -> str:
    """Derive the correct A-E option from the three truth flags."""
    if a_true and r_true:
        return "A" if explains else "B"
    if a_true and not r_true:
        return "C"
    if not a_true and r_true:
        return "D"
    return "E"


def _ar(qid: str, topic: str, category: str, assertion: str, reason: str,
        a_true: bool, r_true: bool, explains: bool, explanation: str,
        difficulty: int = 5, subtopic: str = "") -> dict[str, Any]:
    return dict(
        id=qid, topic=topic, subtopic=subtopic, category=category,
        question_type="assertion_reason", difficulty=difficulty, priority="CRITICAL",
        prompt="",  # rendered from assertion/reason by the quiz UI
        assertion=assertion, reason=reason,
        assertion_truth=a_true, reason_truth=r_true, reason_explains_assertion=explains,
        options=AR_OPTIONS,
        correct_option=ar_key(a_true, r_true, explains),
        model_answer=explanation,
        expected_concepts=[],
        estimated_time=120,
        source_basis="seed",
    )


def _open(qid: str, topic: str, category: str, qtype: str, prompt: str,
          model_answer: str, concepts: list[str], reasoning: str = "",
          difficulty: int = 4, subtopic: str = "", time: int = 240,
          priority: str = "CRITICAL") -> dict[str, Any]:
    return dict(
        id=qid, topic=topic, subtopic=subtopic, category=category,
        question_type=qtype, difficulty=difficulty, priority=priority,
        prompt=prompt, model_answer=model_answer,
        expected_concepts=concepts, expected_reasoning=reasoning,
        estimated_time=time, source_basis="seed",
    )


ML = "Machine Learning"
DL = "Deep Learning"

SEED_QUESTIONS: list[dict[str, Any]] = [

    # =================== 5 ML conceptual reasoning ==========================
    _open("ml_c1", "overfitting", ML, "conceptual_reasoning",
          "A model achieves 99% accuracy on the training set and 62% on the validation set. "
          "Explain precisely what is happening, why it happens, and which single intervention "
          "you would try first. Also name one intervention that would NOT address the problem, "
          "and say why not.",
          "The model is overfitting: it has enough capacity to fit noise and sample-specific "
          "structure in the training data, so training error keeps falling while the gap to "
          "validation error widens - it is fitting variance, not signal. First intervention: "
          "reduce effective capacity or add regularisation (L2 / dropout / fewer parameters), "
          "or obtain more training data, because both shrink the hypothesis space the model can "
          "exploit to memorise. Lowering the learning rate would NOT address it: the learning "
          "rate controls how the optimiser traverses the loss surface, not the capacity of the "
          "hypothesis class - a smaller learning rate will simply converge more slowly to the "
          "same overfitted solution.",
          ["overfitting", "variance", "capacity", "generalization gap", "regularization",
           "more data", "learning rate does not control capacity"],
          reasoning="Must separate optimisation from generalisation.",
          difficulty=4, subtopic="Evaluation"),

    _open("ml_c2", "feature_scaling", ML, "conceptual_reasoning",
          "For which of these algorithms does feature scaling change the result, and why: "
          "KNN, decision trees, SVM with RBF kernel, linear regression fitted by the normal "
          "equation? Justify each answer by the mechanism, not by memorised rules.",
          "KNN: yes - predictions depend on Euclidean distances, so a feature with a larger "
          "numeric range dominates the distance and effectively silences the others. "
          "SVM with RBF: yes - the kernel exp(-gamma||x-x'||^2) is a function of the same "
          "distance, so unscaled features distort the similarity and the effective gamma. "
          "Decision trees: no - splits are chosen by thresholding one feature at a time and any "
          "monotone rescaling preserves the ordering, hence the same splits. "
          "Linear regression by the normal equation: the fitted predictions are unchanged "
          "(coefficients simply rescale inversely), because OLS has a closed-form solution that "
          "is equivariant to invertible linear rescaling; scaling only matters there for "
          "numerical conditioning and for regularised variants, where the penalty is not "
          "scale-invariant.",
          ["distance-based methods", "kernel depends on distance", "trees are threshold based",
           "monotone invariance", "OLS equivariance", "regularisation breaks scale invariance"],
          difficulty=5, subtopic="Data"),

    _open("ml_c3", "bias_variance", ML, "conceptual_reasoning",
          "Explain why increasing k in K-Nearest Neighbours moves the model along the "
          "bias-variance tradeoff. What happens in the two extreme cases k = 1 and k = n?",
          "Each prediction averages over k neighbours. Larger k averages more labels, which "
          "reduces the variance of the prediction (individual noisy labels matter less) but "
          "increases bias, because the neighbourhood grows and includes points that are less "
          "similar to the query, smoothing over genuine local structure. At k = 1 the model "
          "interpolates the training data: zero training error, very high variance, decision "
          "boundary highly sensitive to noise. At k = n every prediction is the global majority "
          "class (or global mean): zero variance, maximal bias - the model ignores the input "
          "entirely.",
          ["averaging reduces variance", "larger neighbourhood increases bias", "k=1 interpolates",
           "k=n constant predictor", "noise sensitivity"],
          difficulty=4, subtopic="Evaluation"),

    _open("ml_c4", "cross_validation", ML, "conceptual_reasoning",
          "You standardise your entire dataset, then run 5-fold cross-validation. Your CV score "
          "is optimistic compared with the true test performance. Explain the mechanism of the "
          "error and state the correct procedure.",
          "This is data leakage. The scaler's mean and standard deviation were computed using "
          "all samples, including those that later serve as validation folds, so information "
          "about the held-out data has entered the training process; the folds are no longer "
          "independent of the fitted preprocessing. The correct procedure is to fit the scaler "
          "inside each fold on that fold's training portion only, and apply the stored statistics "
          "to the held-out fold - i.e. put preprocessing inside the cross-validation pipeline, "
          "not before it.",
          ["data leakage", "statistics computed on held-out data", "fold independence",
           "fit inside the fold", "pipeline"],
          difficulty=5, subtopic="Evaluation"),

    _open("ml_c5", "pca", ML, "conceptual_reasoning",
          "PCA is described as 'unsupervised dimensionality reduction that keeps the most "
          "important directions'. Explain in what precise sense a direction is 'important', and "
          "give a concrete case where the direction PCA discards is exactly the one you needed.",
          "PCA ranks directions by the variance of the projected data: the principal components "
          "are the eigenvectors of the covariance matrix, ordered by eigenvalue, so 'important' "
          "means 'high variance', nothing more. It never looks at labels. If the class-"
          "discriminative signal lies along a low-variance direction - for example two elongated, "
          "parallel class clusters that are separated by a small offset perpendicular to their "
          "long axis - PCA keeps the long (high-variance) within-class direction and discards the "
          "small between-class one, destroying separability. LDA, which maximises the between-"
          "class to within-class scatter ratio, is the supervised alternative for that case.",
          ["variance maximisation", "eigenvectors of covariance", "ignores labels",
           "high variance is not high discriminative power", "LDA alternative"],
          difficulty=5, subtopic="Dimensionality Reduction"),

    # =================== 5 ML calculation ==================================
    _open("ml_x1", "model_evaluation", ML, "calculation",
          "", "", ["precision", "recall", "F1", "confusion matrix"],
          difficulty=3, subtopic="Evaluation", time=240) | {"calc_generator": "metrics"},
    _open("ml_x2", "knn", ML, "calculation",
          "", "", ["Euclidean distance", "majority vote"],
          difficulty=3, subtopic="Classification", time=240) | {"calc_generator": "knn"},
    _open("ml_x3", "naive_bayes", ML, "calculation",
          "", "", ["Bayes rule", "prior", "likelihood", "posterior"],
          difficulty=4, subtopic="Classification", time=240) | {"calc_generator": "bayes"},
    _open("ml_x4", "linear_regression", ML, "calculation",
          "", "", ["least squares", "slope", "intercept", "R^2"],
          difficulty=3, subtopic="Regression", time=360) | {"calc_generator": "linreg"},
    _open("ml_x5", "decision_trees", ML, "calculation",
          "", "", ["entropy", "information gain", "Gini"],
          difficulty=4, subtopic="Classification", time=360) | {"calc_generator": "entropy"},
    _open("ml_x6", "pca", ML, "calculation",
          "", "", ["covariance", "eigenvalues", "explained variance"],
          difficulty=5, subtopic="Dimensionality Reduction", time=420) | {"calc_generator": "pca"},
    _open("ml_x7", "kmeans", ML, "calculation",
          "", "", ["assignment step", "update step", "WCSS"],
          difficulty=4, subtopic="Clustering", time=360) | {"calc_generator": "kmeans"},

    # =================== 5 ML assertion-reason ==============================
    _ar("ml_ar1", "regularization_ml", ML,
        "L2 regularization reduces the variance of a linear model.",
        "L2 regularization removes features from the model by setting their coefficients exactly "
        "to zero.",
        True, False, False,
        "The assertion is true: shrinking coefficients constrains the hypothesis space and lowers "
        "variance (at the cost of some bias). The reason is false: it describes L1/Lasso. L2 "
        "shrinks coefficients smoothly toward zero but, because its penalty is differentiable at "
        "the origin, it does not produce exact zeros. Correct answer: C.",
        subtopic="Regression"),

    _ar("ml_ar2", "knn", ML,
        "KNN requires no training phase.",
        "KNN stores the training data and defers all computation to prediction time.",
        True, True, True,
        "Both statements are true and the reason is exactly the mechanism: KNN is a lazy learner, "
        "so 'training' is just storing the data and the whole cost appears at query time when "
        "distances are computed. Correct answer: A.",
        subtopic="Classification", difficulty=3),

    _ar("ml_ar3", "cross_validation", ML,
        "K-fold cross-validation gives a lower-variance estimate of generalisation performance "
        "than a single train/test split.",
        "K-fold cross-validation trains the model on the entire dataset at once.",
        True, False, False,
        "The assertion is true - averaging over k folds reduces the variance of the performance "
        "estimate because every sample is used for validation exactly once. The reason is false: "
        "each fold trains on k-1 folds only; the model is never trained on all the data in a "
        "single fit during the CV procedure. Correct answer: C.",
        subtopic="Evaluation"),

    _ar("ml_ar4", "kmeans", ML,
        "K-Means always converges to the globally optimal clustering.",
        "The K-Means objective (within-cluster sum of squares) decreases monotonically at every "
        "iteration.",
        False, True, False,
        "The reason is true - both the assignment and update steps can only decrease WCSS, which "
        "is why the algorithm terminates. But the assertion is false: monotone decrease only "
        "guarantees convergence to a local optimum, and the result depends on initialisation "
        "(hence k-means++ and multiple restarts). Correct answer: D. Note the trap: a true "
        "statement about convergence does not imply global optimality.",
        subtopic="Clustering"),

    _ar("ml_ar5", "naive_bayes", ML,
        "Naive Bayes often performs well on text classification despite its independence "
        "assumption being violated.",
        "In text data, words in a document are statistically independent of each other given the "
        "class label.",
        True, False, False,
        "The assertion is true and well documented. The reason is false: words are clearly not "
        "conditionally independent given the class ('New' and 'York' co-occur). The actual "
        "explanation is that classification only needs the correct *argmax*, and the ranking of "
        "class posteriors is often preserved even when the estimated probabilities themselves are "
        "badly calibrated. Correct answer: C.",
        subtopic="Classification"),

    _ar("ml_ar6", "pca", ML,
        "PCA should be applied after standardising features that have different units.",
        "PCA selects directions that maximise variance, and variance depends on the measurement "
        "scale of each feature.",
        True, True, True,
        "Both true, and the reason is the mechanism: a feature measured in millimetres will have "
        "a numerically enormous variance compared to one measured in metres, so unstandardised "
        "PCA would pick the unit choice rather than the structure. Correct answer: A.",
        subtopic="Dimensionality Reduction"),

    # =================== 5 DL conceptual reasoning =========================
    _open("dl_c1", "affine_transformation", DL, "conceptual_reasoning",
          "Show why a multilayer perceptron with all nonlinear activations removed collapses "
          "into a single affine transformation, and explain what this implies about the function "
          "class the network can represent.",
          "With no nonlinearity, layer l computes h_l = W_l h_{l-1} + b_l. Composing two layers "
          "gives W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2), which has exactly the form "
          "W'x + b'. By induction any depth collapses to a single affine map W' = W_L...W_1 with "
          "an accumulated bias. The implication: depth buys nothing in representational power - "
          "the network can only represent linear decision boundaries / linear regressions, "
          "exactly the same function class as a single-layer linear model, no matter how many "
          "layers or units are added. Nonlinear activations are what make the composition strictly "
          "more expressive and give the universal approximation property.",
          ["composition of affine maps is affine", "W' = product of weight matrices",
           "accumulated bias", "depth gives no extra expressivity", "linear decision boundary",
           "universal approximation requires nonlinearity"],
          reasoning="Requires the algebraic composition argument, not just 'it becomes linear'.",
          difficulty=5, subtopic="Foundations"),

    _open("dl_c2", "scaled_dot_product", DL, "conceptual_reasoning",
          "Why is the dot product in scaled dot-product attention divided by sqrt(d_k)? What "
          "specifically goes wrong if the scaling is removed, and why sqrt(d_k) rather than d_k?",
          "If the components of q and k are roughly independent with zero mean and unit variance, "
          "then q.k is a sum of d_k such products, so it has variance proportional to d_k and "
          "standard deviation proportional to sqrt(d_k). Without scaling, the logits fed to the "
          "softmax grow with the head dimension; large-magnitude logits push softmax into a "
          "saturated, near one-hot regime where its Jacobian is almost zero, so gradients vanish "
          "and training stalls. Dividing by sqrt(d_k) normalises the standard deviation of the "
          "logits back to roughly 1, keeping the softmax in a responsive regime. Dividing by d_k "
          "would over-correct - it would shrink the logits by a factor sqrt(d_k) too much, "
          "flattening the attention distribution toward uniform and destroying the model's "
          "ability to be selective.",
          ["variance grows with d_k", "std proportional to sqrt(d_k)", "softmax saturation",
           "vanishing gradients", "sqrt matches the standard deviation", "d_k would over-flatten"],
          difficulty=6, subtopic="Transformers"),

    _open("dl_c3", "dropout", DL, "conceptual_reasoning",
          "Explain the mechanism by which dropout reduces overfitting. Then explain why dropout "
          "is disabled at test time and what must be done to keep the network's outputs consistent.",
          "During training dropout deactivates each unit independently with probability p, so no "
          "unit can rely on the presence of any particular other unit. This prevents co-adaptation: "
          "the network is forced to distribute the representation across redundant features rather "
          "than building fragile detectors that depend on specific partners, which reduces "
          "effective capacity and therefore variance. It can also be read as training an implicit "
          "ensemble of exponentially many thinned subnetworks that share weights. At test time we "
          "want a deterministic prediction and the full ensemble, so dropout is switched off. "
          "Because the expected input to a unit was scaled by the keep probability during "
          "training, the activations must be compensated: either multiply activations by (1-p) at "
          "test time, or - as in standard inverted dropout - divide by (1-p) during training so "
          "that no change is needed at inference.",
          ["random deactivation", "prevents co-adaptation", "reduces effective capacity",
           "implicit ensemble", "deterministic inference", "expectation must be matched",
           "inverted dropout"],
          difficulty=5, subtopic="Training"),

    _open("dl_c4", "vanishing_gradients", DL, "conceptual_reasoning",
          "Explain mathematically why a vanilla RNN suffers from vanishing gradients over long "
          "sequences, and why the LSTM cell state alleviates the problem.",
          "Backpropagation through time multiplies Jacobians along the sequence: the gradient of "
          "the loss at time T with respect to the hidden state at time t contains the product "
          "prod_{k=t+1..T} (W_h^T diag(f'(z_k))). This is a product of (T - t) matrices, so its "
          "magnitude behaves roughly like the (T-t)-th power of the dominant factor. If the "
          "relevant singular values times the activation derivatives are below 1 the product "
          "decays exponentially - vanishing gradients, and dependencies more than a few dozen "
          "steps back receive essentially no learning signal; above 1 it explodes. The LSTM adds "
          "a cell state whose update is additive, c_t = f_t * c_{t-1} + i_t * g_t, so the "
          "derivative dc_t/dc_{t-1} is the forget gate f_t rather than a repeated weight-matrix "
          "product. When the forget gate stays near 1 the gradient can travel many steps almost "
          "undamped along this 'constant error carousel', which is why long-range dependencies "
          "become learnable.",
          ["product of Jacobians", "exponential decay", "repeated multiplication",
           "additive cell state update", "gradient path is the forget gate",
           "constant error carousel", "exploding gradients need clipping"],
          difficulty=6, subtopic="Sequences"),

    _open("dl_c5", "batch_normalization", DL, "conceptual_reasoning",
          "Batch normalization is often said to 'fix internal covariate shift'. Give a more "
          "defensible account of why it helps optimisation, and describe how it behaves "
          "differently at training and inference time.",
          "The internal-covariate-shift story is the historical motivation but is contested. The "
          "better-supported account is that normalising each pre-activation to roughly zero mean "
          "and unit variance makes the loss surface better conditioned: it reduces the dependence "
          "of each layer's gradient on the scale of the previous layers' weights, which smooths "
          "the loss landscape and bounds the gradient magnitude, so larger learning rates become "
          "stable and optimisation converges faster. The learnable gamma and beta restore "
          "representational capacity that pure normalisation would remove. The minibatch noise it "
          "injects also has a mild regularising effect. At training time the statistics come from "
          "the current minibatch; at inference the batch is not available (and predictions must be "
          "deterministic and independent of batch composition), so running averages of mean and "
          "variance accumulated during training are used instead.",
          ["conditioning of the loss surface", "reduces sensitivity to weight scale",
           "allows larger learning rates", "gamma and beta restore capacity",
           "regularising minibatch noise", "running statistics at inference"],
          difficulty=5, subtopic="Training"),

    _open("dl_c6", "activation_functions", DL, "conceptual_reasoning",
          "Compare ReLU and sigmoid as hidden-layer activations for a deep network. Address "
          "gradient behaviour, computational cost, and the failure mode specific to each.",
          "Sigmoid squashes into (0,1) and its derivative peaks at 0.25 and approaches zero in "
          "both tails, so stacking layers multiplies many sub-unit factors and gradients vanish; "
          "its outputs are also not zero-centred, which biases the gradient updates of the next "
          "layer in a common direction and slows convergence. ReLU has derivative exactly 1 for "
          "positive pre-activations, so it does not attenuate the gradient in its active region, "
          "and both the function and its derivative are trivial to compute - these are the main "
          "reasons deep networks became trainable. ReLU's failure mode is the dying ReLU: a unit "
          "whose pre-activation is negative for all inputs has zero gradient forever and can never "
          "recover, which motivates Leaky ReLU / ELU / careful initialisation.",
          ["sigmoid derivative max 0.25", "saturation in both tails", "vanishing gradient",
           "not zero-centred", "ReLU derivative is 1 when active", "cheap to compute",
           "dying ReLU", "leaky ReLU as a fix"],
          difficulty=4, subtopic="Foundations"),

    # =================== 5 DL calculation ==================================
    _open("dl_x1", "backpropagation", DL, "calculation",
          "", "", ["chain rule", "ReLU derivative", "sigmoid+BCE gradient", "weight update"],
          difficulty=6, subtopic="Training", time=600) | {"calc_generator": "mlp_backprop"},
    _open("dl_x2", "mlp", DL, "calculation",
          "", "", ["dense layer parameters", "biases"],
          difficulty=3, subtopic="Foundations", time=180) | {"calc_generator": "mlp_params"},
    _open("dl_x3", "loss_functions", DL, "calculation",
          "", "", ["softmax", "cross entropy", "p - y gradient"],
          difficulty=4, subtopic="Training", time=240) | {"calc_generator": "softmax"},
    _open("dl_x4", "gradient_descent", DL, "calculation",
          "", "", ["derivative", "update rule", "stability"],
          difficulty=3, subtopic="Training", time=240) | {"calc_generator": "gd"},
    _open("dl_x5", "logistic_regression", ML, "calculation",
          "", "", ["sigmoid", "cross entropy gradient"],
          difficulty=4, subtopic="Classification", time=300) | {"calc_generator": "logistic"},
    _open("dl_x6", "dropout", DL, "calculation",
          "", "", ["keep probability", "expectation", "train vs test"],
          difficulty=3, subtopic="Training", time=200) | {"calc_generator": "dropout"},

    # =================== 5 DL assertion-reason =============================
    _ar("dl_ar1", "activation_functions", DL,
        "A multilayer perceptron without nonlinear activation functions cannot solve the XOR "
        "problem.",
        "The composition of affine transformations is itself an affine transformation.",
        True, True, True,
        "Both true and the reason is precisely the explanation: without nonlinearity the whole "
        "network reduces to a single affine map, which can only realise a linear decision "
        "boundary, and XOR is not linearly separable. Correct answer: A.",
        subtopic="Foundations"),

    _ar("dl_ar2", "learning_rate", DL,
        "Reducing the learning rate is an effective remedy for overfitting.",
        "The learning rate controls the size of each parameter update during gradient descent.",
        False, True, False,
        "The reason is a correct definition. The assertion is false: the learning rate governs "
        "optimisation dynamics (speed and stability of convergence), not the capacity of the "
        "hypothesis class. A smaller learning rate reaches essentially the same overfitted "
        "solution more slowly. Overfitting is addressed by regularisation, more data, reduced "
        "capacity or early stopping. Correct answer: D. This is the classic 'true reason, false "
        "assertion' trap.",
        subtopic="Training"),

    _ar("dl_ar3", "batch_normalization", DL,
        "Batch normalization allows the use of higher learning rates.",
        "Batch normalization removes the need for any other form of regularization.",
        True, False, False,
        "The assertion is true - normalising pre-activations improves the conditioning of the "
        "loss surface, so larger steps remain stable. The reason is false: BN provides only a "
        "mild regularising effect from minibatch noise and networks with BN still routinely use "
        "weight decay, data augmentation and sometimes dropout. Correct answer: C.",
        subtopic="Training"),

    _ar("dl_ar4", "residual_connections", DL,
        "Residual connections make very deep networks easier to optimise.",
        "The identity path provides a route along which gradients reach earlier layers without "
        "being repeatedly multiplied by weight matrices.",
        True, True, True,
        "Both true, and the reason is the mechanism. The derivative through a residual block is "
        "(I + dF/dx), so the identity term guarantees a gradient path of magnitude ~1 even when "
        "the residual branch attenuates it - this is what removes the degradation problem in very "
        "deep stacks. Correct answer: A.",
        subtopic="CNN"),

    _ar("dl_ar5", "dropout", DL,
        "Dropout is applied during inference to make predictions more robust.",
        "Dropout reduces co-adaptation between hidden units during training.",
        False, True, False,
        "The reason is true and is dropout's actual mechanism. The assertion is false: standard "
        "dropout is a training-time-only procedure; at inference the full network is used "
        "deterministically (with the appropriate scaling). Correct answer: D. (Monte-Carlo dropout "
        "at inference exists but is a separate uncertainty-estimation technique, not standard "
        "practice.)",
        subtopic="Training"),

    _ar("dl_ar6", "weight_initialization", DL,
        "Initialising all weights of a neural network to zero prevents the network from learning "
        "useful features.",
        "With identical initial weights, all units in a layer receive identical gradients and "
        "therefore remain identical throughout training.",
        True, True, True,
        "Both true and causally linked: zero (or any constant) initialisation fails to break "
        "symmetry, so every unit in a layer computes the same function and receives the same "
        "update forever - the layer has the expressive power of a single unit. This is why random "
        "initialisation schemes such as Xavier/He exist. Correct answer: A.",
        subtopic="Training"),

    # =================== 5 CNN questions ===================================
    _open("cnn_1", "cnn_parameter_count", DL, "calculation",
          "", "", ["output size formula", "weight sharing", "parameter counting"],
          difficulty=4, subtopic="CNN", time=300) | {"calc_generator": "cnn_shape"},

    _open("cnn_2", "receptive_field", DL, "calculation",
          "", "", ["receptive field recursion", "jump"],
          difficulty=5, subtopic="CNN", time=240) | {"calc_generator": "receptive_field"},

    _open("cnn_3", "cnn_basics", DL, "comparison",
          "Compare a convolutional layer with a fully connected layer applied to the same image "
          "input. Address parameter count, the inductive bias each encodes, and what each one "
          "assumes about the data.",
          "A fully connected layer connects every input pixel to every unit, so its parameter "
          "count scales with the number of pixels times the number of units and it treats each "
          "pixel position as an independent, unrelated coordinate - it has essentially no spatial "
          "prior and must learn from data that neighbouring pixels are related. A convolutional "
          "layer applies a small kernel across all positions, so it has two structural priors: "
          "**local connectivity** (a unit only sees a small neighbourhood, encoding the assumption "
          "that relevant image structure is local) and **weight sharing** (the same detector is "
          "applied everywhere, encoding translation equivariance - a feature is worth detecting "
          "regardless of where it appears). The consequences are far fewer parameters "
          "(K*K*C_in*C_out, independent of the input resolution), better sample efficiency, and "
          "far less overfitting on image data. The cost is that the prior is wrong for data with "
          "no spatial locality, where an FC layer or attention may be preferable.",
          ["local connectivity", "weight sharing", "translation equivariance",
           "parameters independent of resolution", "inductive bias", "sample efficiency"],
          difficulty=4, subtopic="CNN"),

    _open("cnn_4", "stride_padding", DL, "what_happens_if",
          "A convolutional layer currently uses kernel 3x3, stride 1, 'same' padding. Describe "
          "precisely what happens to (a) the output spatial size, (b) the receptive field growth "
          "per layer, (c) the computational cost, and (d) the information available to later "
          "layers, if the stride is increased to 2. Then state one situation where this is "
          "desirable and one where it is harmful.",
          "(a) The output side becomes roughly half: floor((H + 2P - K)/S) + 1 with S = 2 halves "
          "the resolution. (b) The receptive field of subsequent layers grows faster, because the "
          "jump (accumulated stride) doubles and every later kernel step now covers two input "
          "pixels. (c) Cost drops by about 4x for the following layers, since the spatial extent "
          "of the feature map falls by 4. (d) Fine spatial detail is discarded - the layer "
          "subsamples rather than aggregating, so high-frequency information is permanently lost "
          "to later layers. Desirable: in the early/middle stages of a classification backbone "
          "where you want a large receptive field and cheap computation and only need global "
          "semantics. Harmful: in dense prediction tasks (segmentation, small-object detection) "
          "where precise localisation is required - which is why such models use dilated "
          "convolutions or skip connections to recover resolution.",
          ["output size halves", "jump doubles", "faster receptive field growth",
           "4x cheaper", "loss of fine detail", "bad for dense prediction / small objects"],
          difficulty=5, subtopic="CNN"),

    _open("cnn_5", "residual_connections", DL, "what_happens_if",
          "A 50-layer CNN with residual connections is retrained after all skip connections are "
          "removed, with everything else identical. Predict what happens during training and "
          "explain the mechanism. Would the same removal matter in a 6-layer network?",
          "Training degrades: the deep plain network converges more slowly and typically reaches "
          "a *higher* training error than the residual version - the degradation problem, which "
          "is an optimisation failure, not overfitting. Mechanism: in a residual block the "
          "Jacobian is (I + dF/dx), so the identity term keeps a gradient path of magnitude ~1 "
          "back through every block; without it the gradient is a product of 50 weight-matrix "
          "Jacobians and attenuates (or amplifies) exponentially with depth. Residual blocks also "
          "make the identity function trivially representable (F = 0), so extra depth cannot hurt "
          "by construction, whereas a plain stack must learn identity mappings explicitly. In a "
          "6-layer network the effect is small: the product is short enough that gradients still "
          "propagate, so skip connections give little benefit at that depth.",
          ["degradation problem", "training error increases", "not overfitting",
           "I + dF/dx", "exponential attenuation with depth",
           "identity is easy to represent", "depth-dependent effect"],
          difficulty=6, subtopic="CNN"),

    _open("cnn_6", "pooling", DL, "diagram_interpretation",
          "A CNN is described as: Input 32x32x3 -> [Conv 3x3, 32 filters, pad 1] -> ReLU -> "
          "[MaxPool 2x2, stride 2] -> [Conv 3x3, 64 filters, pad 1] -> ReLU -> "
          "[MaxPool 2x2, stride 2] -> Flatten -> FC(10). For each stage state the output shape "
          "and the number of trainable parameters, then explain what role each component plays "
          "and what breaks if the pooling layers are removed.",
          "Conv1: output 32x32x32; params (3*3*3 + 1)*32 = 896. Pool1: output 16x16x32; 0 params. "
          "Conv2: output 16x16x64; params (3*3*32 + 1)*64 = 18496. Pool2: output 8x8x64; 0 params. "
          "Flatten: 8*8*64 = 4096. FC: 4096*10 + 10 = 40970 params. Roles: the convolutions "
          "extract local features with shared weights; ReLU supplies the nonlinearity without "
          "which the whole stack would collapse to one affine map; max pooling downsamples, giving "
          "small translation invariance, a larger effective receptive field and cheaper later "
          "layers; the FC layer maps the learned representation to class scores. Removing the "
          "pooling layers leaves the feature maps at 32x32, so the flattened vector becomes "
          "32*32*64 = 65536 and the FC layer alone jumps to ~655k parameters - a large increase in "
          "capacity and overfitting risk, higher compute, and a smaller receptive field, so later "
          "layers see less context.",
          ["output shapes", "conv parameter formula", "pooling has no parameters",
           "flatten size", "FC parameter explosion", "receptive field", "translation invariance"],
          difficulty=5, subtopic="CNN", time=420),

    # =================== 5 RNN / LSTM questions ============================
    _open("rnn_1", "lstm", DL, "comparison",
          "Compare LSTM and GRU: gate structure, parameter count, memory mechanism, and when you "
          "would choose one over the other.",
          "An LSTM has three gates (forget, input, output) plus a candidate update, and maintains "
          "two states: the hidden state h_t and a separate cell state c_t, with c_t = f_t*c_{t-1} "
          "+ i_t*g_t. A GRU merges the forget and input gates into a single update gate z_t and "
          "adds a reset gate r_t, and keeps only one state: h_t = (1-z_t)*h_{t-1} + z_t*h~_t. "
          "Because a GRU has three weight blocks against the LSTM's four, it has roughly 3/4 of "
          "the parameters for the same hidden size, so it trains faster and needs less data. Both "
          "solve the vanishing-gradient problem the same way - via an additive, gated state update "
          "that gives a near-identity gradient path. Empirically their accuracy is usually "
          "comparable; prefer GRU for smaller datasets and tighter compute budgets, prefer LSTM "
          "when the task benefits from separating what is remembered internally (c_t) from what is "
          "exposed to the rest of the network (h_t), typically on long, complex sequences.",
          ["three gates vs two", "separate cell state vs single state", "4 vs 3 weight blocks",
           "3/4 parameters", "additive update solves vanishing gradients",
           "comparable accuracy", "GRU cheaper"],
          difficulty=5, subtopic="Sequences"),

    _open("rnn_2", "lstm", DL, "what_happens_if",
          "What happens if the cell state is removed from an LSTM, keeping the gates operating "
          "only on the hidden state? Explain the consequence for gradient flow.",
          "You lose the additive, largely uninterrupted memory path. The cell state's update "
          "c_t = f_t*c_{t-1} + i_t*g_t has derivative dc_t/dc_{t-1} = f_t, so with the forget "
          "gate near 1 gradients traverse many timesteps almost undamped - the constant error "
          "carousel. If the only state is the hidden state, which is passed through a squashing "
          "nonlinearity and a weight matrix at every step, backpropagation again multiplies many "
          "Jacobians and gradients decay exponentially with sequence length. The result is a model "
          "that behaves much closer to a vanilla RNN: it can still learn short-range dependencies "
          "but long-range ones become untrainable. (A GRU avoids this not by keeping a cell state "
          "but by making its single hidden-state update itself additive and gated.)",
          ["additive memory path", "dc_t/dc_{t-1} = forget gate", "constant error carousel",
           "repeated Jacobian products", "exponential decay", "long-range dependencies lost"],
          difficulty=6, subtopic="Sequences"),

    _open("rnn_3", "rnn", DL, "calculation",
          "", "", ["recurrent parameter formula", "gate count"],
          difficulty=4, subtopic="Sequences", time=240) | {"calc_generator": "lstm_params"},

    _ar("rnn_4", "vanishing_gradients", DL,
        "Gradient clipping solves the vanishing gradient problem in recurrent networks.",
        "Gradient clipping rescales the gradient vector when its norm exceeds a threshold.",
        False, True, False,
        "The reason is a correct description of clipping. The assertion is false: clipping bounds "
        "gradients from *above*, so it addresses **exploding** gradients. It does nothing for "
        "vanishing gradients, where the problem is that the signal is already too small - "
        "rescaling a near-zero gradient does not restore the lost long-range information. "
        "Vanishing gradients are addressed architecturally (LSTM/GRU gating, residual paths) or "
        "via initialisation and activation choices. Correct answer: D.",
        subtopic="Sequences"),

    _open("rnn_5", "seq2seq", DL, "scenario",
          "A seq2seq model without attention translates short sentences well but degrades sharply "
          "on long ones. Diagnose the cause, explain the mechanism, and state the intervention "
          "plus why it works. Name one intervention that would NOT fix this.",
          "Cause: the fixed-size context vector bottleneck. A vanilla encoder-decoder compresses "
          "the entire source sentence into one final hidden vector of fixed dimension; as the "
          "sentence lengthens, the amount of information that must be squeezed into that fixed "
          "budget grows, so early tokens are effectively overwritten, and the decoder has no route "
          "back to the source representation. Intervention: add attention. The decoder then "
          "computes, at every output step, a query against all encoder hidden states and consumes "
          "a weighted sum of them, so the representation it can access grows with the input length "
          "instead of being constant, and gradients also reach the encoder through short paths. "
          "An intervention that would NOT fix it: simply increasing the number of encoder layers "
          "(or training longer) - the bottleneck is the fixed-width single context vector, so a "
          "deeper encoder still funnels everything through the same constant-size channel.",
          ["fixed-size context vector", "bottleneck", "information loss grows with length",
           "attention gives access to all encoder states", "length-adaptive representation",
           "shorter gradient paths", "more layers does not remove the bottleneck"],
          difficulty=5, subtopic="Sequences"),

    # =================== 5 Attention / Transformer questions ===============
    _open("tfm_1", "scaled_dot_product", DL, "calculation",
          "", "", ["dot product", "scaling", "softmax", "weighted values"],
          difficulty=5, subtopic="Transformers", time=360) | {"calc_generator": "attention"},

    _open("tfm_2", "transformer", DL, "calculation",
          "", "", ["multi-head splitting", "projection matrices", "FFN"],
          difficulty=5, subtopic="Transformers", time=300) | {"calc_generator": "tfm_params"},

    _open("tfm_3", "self_vs_cross_attention", DL, "comparison",
          "Distinguish self-attention from cross-attention in an encoder-decoder Transformer. "
          "For each, state where Q, K and V come from, and what the mechanism accomplishes.",
          "In self-attention, Q, K and V are all linear projections of the *same* sequence's "
          "representations. In the encoder this lets every source token gather context from every "
          "other source token, building contextualised representations in a single step regardless "
          "of distance. In the decoder, self-attention is causally masked so a position may only "
          "attend to positions at or before itself, preserving the autoregressive property. In "
          "cross-attention (decoder layers only), the queries come from the decoder's own "
          "representations while the keys and values come from the *encoder output*. That is the "
          "mechanism by which the decoder consults the source sequence: at each generation step it "
          "asks 'which parts of the input are relevant to the token I am producing now?' - it "
          "replaces the single fixed context vector of a classical seq2seq model. Structurally the "
          "computation is identical; only the origin of Q versus K/V differs.",
          ["self: Q,K,V from the same sequence", "cross: Q from decoder, K/V from encoder",
           "causal masking in the decoder", "replaces the fixed context vector",
           "same computation different inputs"],
          difficulty=5, subtopic="Transformers"),

    _ar("tfm_4", "transformer", DL,
        "Positional encodings are necessary in a Transformer but were not needed in an RNN.",
        "Self-attention is permutation-equivariant: without positional information it computes the "
        "same set of outputs for any reordering of the input tokens.",
        True, True, True,
        "Both true, and the reason is exactly the cause. Attention aggregates a weighted sum over "
        "all positions with no inherent notion of order, so token order must be injected "
        "explicitly. An RNN processes tokens sequentially, so order is encoded in the computation "
        "itself. Correct answer: A.",
        subtopic="Transformers"),

    _open("tfm_5", "bert", DL, "comparison",
          "Compare BERT and GPT: architecture, pretraining objective, direction of context, and "
          "the class of task each is naturally suited to.",
          "BERT is encoder-only, pretrained with masked language modelling (predict randomly "
          "masked tokens, historically with next-sentence prediction). Because it is not "
          "autoregressive, each token attends to context on **both** sides, giving deeply "
          "bidirectional representations - ideal for understanding tasks (classification, NER, "
          "extractive QA, sentence similarity), typically by fine-tuning a small head on the [CLS] "
          "or token representations. GPT is decoder-only with causal masking, pretrained to "
          "predict the next token, so context is strictly left-to-right. That makes it naturally "
          "generative and enables in-context/few-shot learning, where the task is specified in the "
          "prompt with no weight updates. The trade-off is direct: bidirectionality gives BERT "
          "better representations for understanding but makes it unable to generate "
          "autoregressively; causal masking makes GPT a generator but denies each token access to "
          "its right-hand context.",
          ["encoder-only vs decoder-only", "masked LM vs next-token prediction",
           "bidirectional vs causal/left-to-right", "understanding vs generation",
           "fine-tuning vs in-context learning"],
          difficulty=4, subtopic="Transformers"),

    _open("tfm_6", "attention", DL, "what_happens_if",
          "In a Transformer encoder block, what happens if (a) the residual connections are "
          "removed, (b) the layer normalisation is removed, (c) the feed-forward sublayer is "
          "removed? Answer each with the mechanism.",
          "(a) Without residuals the gradient must pass through every attention and FFN Jacobian "
          "in sequence, so deep stacks become hard to optimise and training degrades; the "
          "identity path that guarantees a ~unit-magnitude gradient route is gone, and the block "
          "can no longer trivially represent the identity function. (b) Without LayerNorm the "
          "activation scales drift between layers and the loss surface is worse conditioned, so "
          "training becomes unstable and typically requires much smaller learning rates and "
          "careful warmup; deep Transformers frequently fail to converge at all. (c) Without the "
          "feed-forward sublayer each block reduces to attention followed by a linear projection, "
          "so its per-token transformation is essentially linear. The FFN is where the block's "
          "position-wise nonlinear capacity lives (typically 4x expansion then projection back), "
          "and it holds roughly two-thirds of the block's parameters - removing it sharply reduces "
          "expressive power, leaving a model that can mix tokens but barely transform them.",
          ["residual gradient path", "identity representable", "LayerNorm conditions activations",
           "training instability without normalisation", "FFN provides positionwise nonlinearity",
           "FFN holds most parameters", "attention alone only mixes tokens"],
          difficulty=6, subtopic="Transformers", time=420),

    # =================== extra scenario / graph items ======================
    _open("misc_1", "overfitting", ML, "graph_interpretation",
          "A training curve shows training loss decreasing steadily to near zero, while the "
          "validation loss decreases for the first 15 epochs, reaches a minimum, and then rises "
          "steadily for the remaining 40 epochs. Interpret the curve: name the phenomenon, "
          "identify the correct stopping point, and explain what the rising validation curve "
          "means about what the model is learning.",
          "The curve shows classic overfitting with the optimal early-stopping point at the "
          "validation minimum around epoch 15. Up to that point the model is learning "
          "generalisable structure, so both curves fall. After it, the continued fall in training "
          "loss with a rising validation loss means the extra capacity is being spent fitting "
          "noise and idiosyncrasies specific to the training sample, which do not transfer - "
          "variance is increasing while bias no longer decreases usefully. The correct action is "
          "early stopping with the checkpoint from the validation minimum (not the final epoch), "
          "optionally combined with regularisation or more data. Note that the training loss alone "
          "gives no signal here: it decreases monotonically in both the useful and harmful phases.",
          ["overfitting", "validation minimum is the stopping point", "fitting noise",
           "increasing variance", "checkpoint selection", "training loss is uninformative alone"],
          difficulty=4, subtopic="Evaluation"),

    _open("misc_2", "bagging_vs_boosting", ML, "comparison",
          "Compare bagging and boosting: how each ensemble is built, which error component each "
          "primarily reduces, sensitivity to noise, and parallelisability.",
          "Bagging trains many base learners **independently in parallel** on bootstrap resamples "
          "and averages (or votes over) them. Averaging decorrelated predictors primarily reduces "
          "**variance**, leaving bias roughly at the level of a single learner - hence it is paired "
          "with deep, low-bias, high-variance trees, as in random forests, which further decorrelate "
          "by subsampling features at each split. Boosting trains learners **sequentially**, each "
          "one fitted to the errors (residuals or reweighted samples) of the current ensemble, so "
          "it primarily reduces **bias** and is paired with shallow, high-bias weak learners such "
          "as stumps. Consequences: boosting can reach lower error but is more sensitive to noisy "
          "data and label noise, since it keeps concentrating capacity on persistently misclassified "
          "points - which may be outliers - and it can overfit if run for too many rounds without "
          "shrinkage. Bagging is naturally parallel; boosting is inherently sequential.",
          ["parallel vs sequential", "bootstrap vs reweighting/residuals",
           "variance reduction vs bias reduction", "deep vs shallow base learners",
           "boosting sensitive to noise", "parallelisability"],
          difficulty=5, subtopic="Ensembles"),

    _open("misc_3", "dbscan", ML, "comparison",
          "Compare DBSCAN and K-Means: what each assumes about cluster shape, how the number of "
          "clusters is determined, how each handles outliers, and the parameters each requires.",
          "K-Means assumes clusters are roughly spherical and of comparable size, because it "
          "assigns every point to the nearest centroid by Euclidean distance and minimises "
          "within-cluster variance; k must be chosen in advance (elbow/silhouette) and every point "
          "is forced into some cluster, so outliers distort the centroids. DBSCAN makes no shape "
          "assumption: it grows clusters from **core points** that have at least MinPts neighbours "
          "within radius eps, so it recovers arbitrarily shaped, non-convex clusters; the number of "
          "clusters emerges from the data, and points in no dense region are explicitly labelled "
          "**noise** rather than forced into a cluster - which makes it robust to outliers and "
          "usable for outlier detection. The trade-off: DBSCAN needs eps and MinPts, which are "
          "hard to set, and it struggles when clusters have very different densities, whereas "
          "K-Means only needs k and scales better to large datasets.",
          ["spherical assumption vs arbitrary shape", "k chosen in advance vs emergent",
           "outliers forced in vs labelled noise", "core/border/noise points",
           "eps and MinPts", "varying density weakness"],
          difficulty=5, subtopic="Clustering"),

    _open("misc_4", "transfer_learning", DL, "scenario",
          "You must train an image classifier for a medical task with only 800 labelled images. "
          "Describe the approach you would take and justify each decision. What would you do "
          "differently with 800,000 images?",
          "With 800 images, training a deep CNN from scratch would overfit badly - the parameter "
          "count vastly exceeds the information in the data. Use transfer learning: take a network "
          "pretrained on a large corpus (e.g. ImageNet), freeze the early layers, which encode "
          "generic edge/texture/shape features that transfer across domains, and replace and train "
          "the classification head. If performance plateaus, unfreeze the later blocks and "
          "fine-tune with a small learning rate so the pretrained weights are not destroyed. "
          "Combine with heavy data augmentation, strong regularisation (weight decay, dropout), "
          "early stopping on a validation split, and cross-validation given how small the dataset "
          "is - and check that augmentations are clinically valid. With 800,000 images the "
          "calculus changes: there is enough data to train from scratch, pretraining gives less "
          "advantage (mainly faster convergence), full fine-tuning of all layers is safe, a larger "
          "architecture becomes justified, and the emphasis shifts from fighting overfitting to "
          "optimisation and throughput.",
          ["small data cannot support a deep net", "pretrained features are generic early on",
           "freeze early layers, replace the head", "fine-tune later layers with a small LR",
           "augmentation and regularisation", "with large data train from scratch",
           "capacity should scale with data"],
          difficulty=4, subtopic="CNN"),

    _open("misc_5", "gmm", ML, "comparison",
          "Compare K-Means with a Gaussian Mixture Model fitted by EM. Explain in what precise "
          "sense K-Means is a limiting case of GMM.",
          "K-Means makes **hard** assignments: each point belongs entirely to its nearest centroid. "
          "A GMM makes **soft** assignments: the E-step computes each point's responsibility - the "
          "posterior probability that component j generated it - and the M-step updates each "
          "component's mean, covariance and mixing weight as responsibility-weighted statistics. "
          "Because a GMM fits a full covariance per component it can model elongated, rotated and "
          "differently sized clusters, and it gives a generative density model with a likelihood, "
          "whereas K-Means only minimises within-cluster squared distance and implicitly assumes "
          "isotropic, equally sized clusters. K-Means is the limiting case of a GMM with covariances "
          "fixed to sigma^2 I and sigma -> 0: the responsibilities then become one-hot (the nearest "
          "centroid takes all the probability mass), the E-step reduces to nearest-centroid "
          "assignment and the M-step to averaging the assigned points, which is exactly Lloyd's "
          "algorithm. Both are EM-style alternating optimisations, and both converge only to a "
          "local optimum.",
          ["hard vs soft assignment", "responsibilities", "E-step and M-step",
           "full covariance models shape", "generative likelihood",
           "limiting case sigma -> 0 gives one-hot responsibilities", "local optimum"],
          difficulty=6, subtopic="Clustering"),
]


def seed_question_count() -> dict[str, int]:
    out: dict[str, int] = {}
    for q in SEED_QUESTIONS:
        out[q["question_type"]] = out.get(q["question_type"], 0) + 1
    return out
