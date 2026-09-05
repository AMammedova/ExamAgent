"""Assertion-Reason engine.

Every item carries three internal flags - assertion_truth, reason_truth and
reason_explains_assertion - from which the correct A-E option is *derived*, never
hand-typed. The student sees only the assertion, the reason and the options.

The offline bank below deliberately covers all five answer patterns, including
the two traps the examiner favours:
  B  both true but causally unrelated
  D  false assertion supported by a true reason
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ..config import get_logger
from ..models.schemas import AnswerOption, Category, Priority, Question, QuestionType
from ..data.seed_questions import AR_OPTIONS, ar_key
from .llm import get_llm, system_with_language

log = get_logger(__name__)

OPTIONS = [AnswerOption(key=o["key"], text=o["text"]) for o in AR_OPTIONS]


@dataclass
class ARItem:
    topic_id: str
    category: str
    assertion: str
    reason: str
    a_true: bool
    r_true: bool
    explains: bool
    explanation: str
    difficulty: int = 5

    @property
    def key(self) -> str:
        return ar_key(self.a_true, self.r_true, self.explains)


ML = "Machine Learning"
DL = "Deep Learning"

#: Curated offline bank. Non-trivial by construction: every reason is plausibly
#: related to its assertion, so the student cannot answer by topic-matching.
AR_BANK: list[ARItem] = [
    # ---------------- ML ----------------
    ARItem("overfitting", ML,
           "A model with a large gap between training and validation accuracy is overfitting.",
           "Overfitting occurs when a model has insufficient capacity to represent the underlying "
           "function.",
           True, False, False,
           "The assertion is true - the train/validation gap is the defining symptom. The reason "
           "describes UNDERfitting: overfitting comes from too much capacity relative to the data, "
           "not too little. Answer: C."),

    ARItem("overfitting", ML,
           "Adding more training data typically reduces overfitting.",
           "More data makes it harder for a fixed-capacity model to memorise the training set, so "
           "it must rely on structure that generalises.",
           True, True, True,
           "Both true and causally connected: with a fixed hypothesis class, more samples "
           "constrain the set of hypotheses consistent with the data, shrinking the variance term. "
           "Answer: A."),

    ARItem("model_evaluation", ML,
           "Accuracy can be a misleading metric on an imbalanced dataset.",
           "A classifier that always predicts the majority class can achieve high accuracy while "
           "having zero recall on the minority class.",
           True, True, True,
           "Both true and the reason is exactly the failure mode: with 99% negatives, the trivial "
           "all-negative classifier scores 99% accuracy and is useless. Answer: A."),

    ARItem("model_evaluation", ML,
           "Increasing the classification threshold generally increases precision.",
           "A higher threshold means the classifier only predicts positive when it is more "
           "confident, so fewer false positives are produced.",
           True, True, True,
           "Both true and causally linked. Note the cost the examiner wants named: recall falls "
           "at the same time, since genuine positives below the threshold are now missed. Answer: A."),

    ARItem("confusion_matrix", ML,
           "In medical screening, recall is usually prioritised over precision.",
           "Recall is defined as TP/(TP+FN).",
           True, True, False,
           "Both statements are true, but the reason is only a definition - it does not explain "
           "the priority. The actual explanation is asymmetric cost: a false negative (a missed "
           "disease) is far more harmful than a false positive (an unnecessary follow-up test). "
           "Answer: B. This is the classic 'true definition that explains nothing' trap."),

    ARItem("feature_scaling", ML,
           "Decision trees do not require feature scaling.",
           "Tree splits are chosen by thresholding a single feature at a time, and any monotone "
           "rescaling preserves the ordering of values.",
           True, True, True,
           "Both true and the reason is the mechanism: since the split criterion only depends on "
           "the order of values within a feature, a monotone transform yields identical splits. "
           "Answer: A."),

    ARItem("cross_validation", ML,
           "Leave-one-out cross-validation gives a nearly unbiased estimate of generalisation "
           "error.",
           "Leave-one-out cross-validation is computationally cheaper than 5-fold "
           "cross-validation.",
           True, False, False,
           "The assertion is true - each model is trained on n-1 samples, almost the full dataset, "
           "so the bias is very small (though the variance is high). The reason is false: LOOCV "
           "fits n models rather than 5, so it is far more expensive. Answer: C."),

    ARItem("regularization_ml", ML,
           "L1 regularization can perform feature selection.",
           "The L1 penalty is non-differentiable at zero, and its constant-magnitude gradient can "
           "drive coefficients exactly to zero.",
           True, True, True,
           "Both true and the reason is the mechanism - geometrically, the diamond-shaped L1 "
           "constraint region has corners on the axes, so the optimum frequently lands on one, "
           "giving exact zeros. Answer: A."),

    ARItem("svm", ML,
           "An SVM's decision boundary is determined only by the support vectors.",
           "The SVM optimisation maximises the margin, and only points on or inside the margin "
           "have non-zero dual coefficients.",
           True, True, True,
           "Both true, and the reason is exactly why: points outside the margin have alpha = 0 and "
           "could be deleted from the training set without changing the solution. Answer: A."),

    ARItem("kernel_svm", ML,
           "The kernel trick lets an SVM find a nonlinear decision boundary without explicitly "
           "computing high-dimensional feature vectors.",
           "A kernel function computes the inner product of two points in the feature space "
           "directly from their original coordinates.",
           True, True, True,
           "Both true and causally linked: because the SVM dual depends on the data only through "
           "inner products, replacing them with K(x, x') gives the effect of the mapping without "
           "ever forming phi(x). Answer: A."),

    ARItem("random_forests", ML,
           "Random forests reduce the variance of decision trees.",
           "Random forests grow each tree to full depth on the entire training set without any "
           "randomisation.",
           True, False, False,
           "The assertion is true. The reason is false and contradicts the method: random forests "
           "rely on two sources of randomisation - bootstrap resampling and random feature subsets "
           "at each split - to decorrelate the trees, which is precisely what makes averaging "
           "reduce variance. Answer: C."),

    ARItem("knn", ML,
           "KNN performance degrades in very high-dimensional spaces.",
           "In high dimensions, distances between points concentrate, so the nearest and farthest "
           "neighbours become nearly equidistant.",
           True, True, True,
           "Both true and the reason is the mechanism of the curse of dimensionality: when the "
           "contrast between distances vanishes, 'nearest' carries almost no information. Answer: A."),

    ARItem("logistic_regression", ML,
           "Logistic regression is a linear model.",
           "Logistic regression applies a nonlinear sigmoid function to its output.",
           True, True, False,
           "Both statements are true, but the reason does not explain the assertion - if anything "
           "it appears to contradict it. Logistic regression is called linear because its decision "
           "boundary is the set where w.x + b = 0, a hyperplane; the sigmoid is a monotone "
           "transformation of that linear score into a probability and does not bend the boundary. "
           "Answer: B."),

    ARItem("pca", ML,
           "The principal components produced by PCA are mutually orthogonal.",
           "They are eigenvectors of a real symmetric covariance matrix, whose eigenvectors for "
           "distinct eigenvalues are orthogonal.",
           True, True, True,
           "Both true and the reason is the linear-algebraic guarantee (spectral theorem). Answer: A."),

    ARItem("kmeans", ML,
           "The result of K-Means can change if you rerun it with a different random seed.",
           "K-Means minimises a non-convex objective and converges to a local optimum determined "
           "by the initialisation.",
           True, True, True,
           "Both true and causally linked - the standard mitigations are k-means++ initialisation "
           "and multiple restarts keeping the lowest WCSS. Answer: A."),

    ARItem("dbscan", ML,
           "DBSCAN can find non-convex clusters that K-Means cannot.",
           "DBSCAN grows clusters by connecting density-reachable points rather than assigning "
           "points to the nearest centroid.",
           True, True, True,
           "Both true and the reason is the mechanism: chaining through dense neighbourhoods can "
           "trace an arbitrary shape, whereas nearest-centroid assignment always yields convex "
           "(Voronoi) regions. Answer: A."),

    ARItem("hierarchical_clustering", ML,
           "Agglomerative hierarchical clustering requires the number of clusters to be specified "
           "before the algorithm runs.",
           "A dendrogram records the full merge history, so any number of clusters can be obtained "
           "afterwards by cutting it at a chosen height.",
           False, True, False,
           "The reason is true and is precisely why the assertion is false: the algorithm runs to "
           "completion independently of k, and k is chosen after the fact by cutting the "
           "dendrogram. Answer: D."),

    ARItem("expectation_maximization", ML,
           "The EM algorithm is guaranteed to find the global maximum of the likelihood.",
           "Each EM iteration is guaranteed not to decrease the observed-data log-likelihood.",
           False, True, False,
           "The reason is a true and important property (monotone ascent). But monotone "
           "improvement only guarantees convergence to a stationary point - typically a local "
           "maximum that depends on initialisation. Answer: D. Note the parallel with the K-Means "
           "trap."),

    ARItem("bias_variance", ML,
           "A very flexible model always generalises better than a simpler one.",
           "Increasing model complexity reduces bias.",
           False, True, False,
           "The reason is true in isolation. The assertion is false: reducing bias raises variance, "
           "and total expected error is the sum of bias^2, variance and irreducible noise, so past "
           "the optimum extra flexibility increases test error. Answer: D."),

    ARItem("boosting", ML,
           "Boosting can be more sensitive to label noise than bagging.",
           "Boosting repeatedly increases the weight of misclassified samples, so persistently "
           "mislabelled points attract disproportionate attention.",
           True, True, True,
           "Both true and the reason is the mechanism. Answer: A."),

    ARItem("kde", ML,
           "The bandwidth of a kernel density estimator controls a bias-variance tradeoff.",
           "A small bandwidth produces a spiky estimate that follows individual samples, while a "
           "large bandwidth oversmooths and blurs genuine structure.",
           True, True, True,
           "Both true and the reason is exactly the tradeoff: small bandwidth = low bias, high "
           "variance; large bandwidth = high bias, low variance. Answer: A."),

    ARItem("data_preprocessing", ML,
           "One-hot encoding should be preferred over integer label encoding for nominal "
           "categorical features in a linear model.",
           "Integer encoding imposes an artificial ordering and spacing between categories that "
           "the model will interpret as meaningful.",
           True, True, True,
           "Both true and the reason is the mechanism - a linear model would treat 'category 3' as "
           "three times 'category 1'. Answer: A."),

    # ---------------- DL ----------------
    ARItem("backpropagation", DL,
           "Backpropagation computes gradients more efficiently than evaluating each partial "
           "derivative numerically.",
           "Backpropagation applies the chain rule in reverse, reusing intermediate results so all "
           "parameter gradients are obtained in roughly one backward pass.",
           True, True, True,
           "Both true and the reason is the mechanism: reverse-mode differentiation costs about "
           "the same as one forward pass regardless of the parameter count, whereas numerical "
           "differentiation would need two evaluations per parameter. Answer: A."),

    ARItem("backpropagation", DL,
           "Backpropagation is a learning algorithm that decides how the weights should change.",
           "Backpropagation efficiently computes the gradient of the loss with respect to every "
           "parameter.",
           False, True, False,
           "The reason is true and is exactly what backpropagation does - and that is why the "
           "assertion is false. Backpropagation only *computes gradients*; the optimiser (SGD, "
           "Adam, ...) decides how to use them to update the weights. Answer: D. Examiners like "
           "this distinction."),

    ARItem("gradient_descent", DL,
           "Stochastic gradient descent can escape some poor local minima that full-batch "
           "gradient descent would settle into.",
           "The gradient computed on a minibatch is a noisy estimate of the full gradient.",
           True, True, True,
           "Both true and the reason is the mechanism: gradient noise perturbs the iterate out of "
           "sharp, narrow minima. Answer: A."),

    ARItem("learning_rate", DL,
           "A learning rate that is too large can cause the loss to increase.",
           "With a large step size the update can overshoot the minimum and land at a point of "
           "higher loss, and the iterates may oscillate or diverge.",
           True, True, True,
           "Both true and the reason is the mechanism - stability requires eta < 2/L for an "
           "L-smooth objective. Answer: A."),

    ARItem("weight_initialization", DL,
           "He initialisation is preferred over Xavier initialisation for ReLU networks.",
           "ReLU sets roughly half of its inputs to zero, halving the variance of the activations, "
           "which He initialisation compensates for with a factor of 2 in the variance.",
           True, True, True,
           "Both true and the reason is exactly the derivation: Var(W) = 2/fan_in for ReLU versus "
           "the symmetric-activation assumption behind Xavier. Answer: A."),

    ARItem("optimizers", DL,
           "Adam applies bias correction to its moment estimates.",
           "The moving averages are initialised at zero, which biases them toward zero during the "
           "first few steps.",
           True, True, True,
           "Both true and the reason is the justification: dividing by (1 - beta^t) removes the "
           "initialisation bias, which matters most in the earliest iterations. Answer: A."),

    ARItem("optimizers", DL,
           "Adam usually converges in fewer iterations than plain SGD.",
           "Adam adapts a per-parameter step size using estimates of the first and second moments "
           "of the gradient.",
           True, True, True,
           "Both true and causally linked. Worth adding in an exam answer: SGD with momentum and a "
           "well-tuned schedule often generalises slightly better despite converging more slowly. "
           "Answer: A."),

    ARItem("activation_functions", DL,
           "The softmax function is typically used in the output layer of a multi-class "
           "classifier.",
           "Softmax outputs are non-negative and sum to one, so they can be read as a probability "
           "distribution over the classes.",
           True, True, True,
           "Both true and the reason is the justification, which is also why softmax pairs with "
           "cross-entropy. Answer: A."),

    ARItem("activation_functions", DL,
           "ReLU units can permanently stop learning during training.",
           "If a unit's pre-activation becomes negative for all inputs, its gradient is zero and "
           "its weights receive no further updates.",
           True, True, True,
           "Both true and the reason is the dying-ReLU mechanism. Answer: A."),

    ARItem("dropout", DL,
           "Dropout increases training time to convergence.",
           "Dropout makes the effective network different at every step, so the gradient signal is "
           "noisier and more epochs are needed.",
           True, True, True,
           "Both true and causally linked. The examiner wants the trade-off stated: slower "
           "convergence in exchange for better generalisation. Answer: A."),

    ARItem("weight_decay", DL,
           "L2 weight decay changes the objective function being optimised.",
           "It adds a penalty term lambda||w||^2 to the loss, so the optimiser minimises a "
           "different function than the original loss.",
           True, True, True,
           "Both true and the reason is the mechanism, which yields the extra gradient term "
           "2*lambda*w and hence multiplicative shrinkage at each step. Answer: A."),

    ARItem("batch_normalization", DL,
           "Batch normalization behaves differently at training and inference time.",
           "At inference the batch statistics are replaced by running averages accumulated during "
           "training.",
           True, True, True,
           "Both true and the reason is exactly the difference - necessary so that a prediction "
           "does not depend on which other samples share its batch. Answer: A."),

    ARItem("cnn_basics", DL,
           "A convolutional layer has far fewer parameters than a fully connected layer with the "
           "same input and output sizes.",
           "The same kernel is reused at every spatial position instead of learning an independent "
           "weight for each pair of positions.",
           True, True, True,
           "Both true and the reason is weight sharing, the mechanism. Answer: A."),

    ARItem("pooling", DL,
           "Max pooling adds no trainable parameters to a network.",
           "Max pooling reduces the spatial dimensions of the feature map.",
           True, True, False,
           "Both statements are true, but the reason does not explain the assertion - "
           "downsampling and having parameters are independent properties (a strided convolution "
           "also downsamples yet does have parameters). The correct explanation is that max "
           "pooling applies a fixed selection function with nothing to learn. Answer: B."),

    ARItem("stride_padding", DL,
           "Padding is used to prevent the spatial dimensions from shrinking after a convolution.",
           "Without padding, a KxK kernel can only be centred on positions at least (K-1)/2 away "
           "from the border, so the output is smaller than the input.",
           True, True, True,
           "Both true and the reason is the mechanism. Answer: A."),

    ARItem("transfer_learning", DL,
           "Transfer learning is particularly valuable when the target dataset is small.",
           "Early layers of a pretrained network encode generic features such as edges and "
           "textures that transfer across visual domains.",
           True, True, True,
           "Both true and causally linked: the pretrained features supply information the small "
           "dataset cannot, so far fewer parameters need to be estimated from it. Answer: A."),

    ARItem("rnn", DL,
           "A recurrent neural network can process input sequences of variable length.",
           "The same weight matrices are applied at every timestep, so the number of parameters "
           "does not depend on the sequence length.",
           True, True, True,
           "Both true and the reason is the mechanism (parameter sharing across time). Answer: A."),

    ARItem("lstm", DL,
           "The forget gate is the component that allows an LSTM to retain information over many "
           "timesteps.",
           "When the forget gate is close to 1 the cell state is carried forward almost unchanged, "
           "and the gradient along that path is multiplied by a value near 1 at each step.",
           True, True, True,
           "Both true and the reason is the mechanism (the constant error carousel). Answer: A."),

    ARItem("gru", DL,
           "A GRU has fewer parameters than an LSTM with the same hidden size.",
           "A GRU has no output gate and merges the forget and input gates into a single update "
           "gate, leaving three weight blocks instead of four.",
           True, True, True,
           "Both true and the reason is the arithmetic: 3/4 of the LSTM's parameters. Answer: A."),

    ARItem("attention", DL,
           "Attention allows a decoder to access information from any position of the input "
           "sequence.",
           "Attention computes a weighted sum over all encoder hidden states, with weights derived "
           "from the compatibility between the decoder query and each encoder key.",
           True, True, True,
           "Both true and the reason is the mechanism, which is what removes the fixed-context-"
           "vector bottleneck. Answer: A."),

    ARItem("transformer", DL,
           "Transformers can be parallelised across sequence positions during training more "
           "effectively than RNNs.",
           "Self-attention computes all pairwise interactions in a single matrix operation, with "
           "no dependence on the previous timestep's output.",
           True, True, True,
           "Both true and the reason is the mechanism, and the main practical reason Transformers "
           "displaced RNNs. Answer: A."),

    ARItem("transformer", DL,
           "The self-attention operation has computational cost that grows quadratically with "
           "sequence length.",
           "Every token must compute a compatibility score with every other token, giving n^2 "
           "scores for a sequence of length n.",
           True, True, True,
           "Both true and the reason is the mechanism - the motivation for efficient-attention "
           "variants. Answer: A."),

    ARItem("bert", DL,
           "BERT can be used directly to generate text autoregressively.",
           "BERT is pretrained with a masked language modelling objective using bidirectional "
           "context.",
           False, True, False,
           "The reason is true and is exactly why the assertion is false: bidirectional masked "
           "prediction is not a left-to-right factorisation of the sequence probability, so BERT "
           "has no native autoregressive generation procedure. Answer: D."),

    ARItem("gpt", DL,
           "GPT models use masked (causal) self-attention in every layer.",
           "Without masking, a position could attend to future tokens, which would leak the answer "
           "during next-token-prediction training.",
           True, True, True,
           "Both true and the reason is the justification. Answer: A."),

    ARItem("embeddings", DL,
           "Word embeddings can capture semantic relationships between words.",
           "Embeddings are trained so that words appearing in similar contexts obtain nearby "
           "vectors, following the distributional hypothesis.",
           True, True, True,
           "Both true and the reason is the mechanism. Answer: A."),

    ARItem("vision_transformers", DL,
           "Vision Transformers typically require more training data than CNNs to reach "
           "comparable accuracy.",
           "A ViT lacks the built-in locality and translation-equivariance priors of a "
           "convolution, so it must learn those regularities from data.",
           True, True, True,
           "Both true and the reason is the mechanism - which is why ViTs are usually pretrained "
           "on very large corpora before fine-tuning. Answer: A."),

    ARItem("seq2seq", DL,
           "A classical encoder-decoder without attention struggles with long input sequences.",
           "The encoder compresses the entire input into a single fixed-size context vector.",
           True, True, True,
           "Both true and the reason is the bottleneck mechanism. Answer: A."),

    ARItem("residual_connections", DL,
           "Concatenation-based skip connections increase the channel count of a feature map "
           "while additive skip connections do not.",
           "Concatenation stacks the two tensors along the channel dimension, whereas addition "
           "requires matching shapes and returns the same shape.",
           True, True, True,
           "Both true and the reason is the mechanism - the U-Net versus ResNet distinction. "
           "Answer: A."),

    ARItem("object_detection", DL,
           "Detecting small objects is harder on deep, heavily downsampled feature maps.",
           "Deep feature maps have larger receptive fields.",
           True, True, False,
           "Both statements are true, but the reason is not the explanation - a large receptive "
           "field is generally beneficial. The actual cause is the loss of spatial resolution: "
           "after a total stride of 32, an 8-pixel object occupies less than one cell, so it "
           "cannot be localised. That is why detectors fuse higher-resolution features (FPN). "
           "Answer: B."),

    ARItem("scaled_dot_product", DL,
           "Attention logits are divided by sqrt(d_k) before the softmax.",
           "Dividing by sqrt(d_k) makes the attention weights sum to one.",
           True, False, False,
           "The assertion is true. The reason is false - the softmax is what makes the weights sum "
           "to one, and it does so with or without scaling. The scaling controls the variance of "
           "the logits to avoid softmax saturation. Answer: C."),

    ARItem("early_stopping", DL,
           "Early stopping acts as a form of regularisation.",
           "Stopping before convergence limits how far the weights can move from their "
           "initialisation, restricting the effective capacity of the model.",
           True, True, True,
           "Both true and the reason is the mechanism - for linear models early stopping can be "
           "shown to be approximately equivalent to L2 regularisation. Answer: A."),

    ARItem("mlp", DL,
           "A single hidden layer network with enough units can approximate any continuous "
           "function on a compact domain.",
           "This guarantees that a single hidden layer is the most practical architecture for "
           "real problems.",
           True, False, False,
           "The assertion is the universal approximation theorem and is true. The reason is false: "
           "the theorem is an existence result and says nothing about the *number of units* "
           "required (which may be exponential) or about whether gradient descent can find such a "
           "solution. Depth is usually far more parameter-efficient. Answer: C."),
]


#: Deliberately trap-heavy supplement. The examiner rarely makes every item an
#: "A", so these keep the B/C/D/E patterns well represented.
AR_BANK += [
    ARItem("data_preprocessing", ML,
           "Missing values should always be replaced with the column mean.",
           "Mean imputation preserves the mean of the observed feature distribution.",
           False, True, False,
           "The reason is true - that is precisely what mean imputation guarantees. The assertion "
           "is false: mean imputation shrinks the variance, destroys correlations, and is wrong "
           "when data are not missing at random (a missing value may itself be informative). "
           "Median, model-based imputation or an explicit missingness indicator are often better. "
           "Answer: D."),

    ARItem("model_validation", ML,
           "The test set may be used to choose the model's hyperparameters as long as it is only "
           "used once at the end.",
           "Hyperparameters are not learned from the training data by the optimiser.",
           False, True, False,
           "The reason is a true definition of a hyperparameter. The assertion is false and "
           "self-contradictory: selecting on the test set *is* using it, and the reported score "
           "then becomes optimistically biased. Hyperparameters belong to a validation set or "
           "inner cross-validation loop. Answer: D."),

    ARItem("linear_regression", ML,
           "A high R^2 means the model will generalise well to new data.",
           "R^2 measures the proportion of the variance in the target explained by the model.",
           False, True, False,
           "The reason is the correct definition. The assertion is false: R^2 computed on the "
           "training data always increases as predictors are added and says nothing about "
           "out-of-sample performance - a perfectly overfitted model can reach R^2 = 1. Answer: D."),

    ARItem("polynomial_regression", ML,
           "Polynomial regression is a nonlinear model.",
           "Its decision surface / fitted curve is not a straight line.",
           False, True, False,
           "The reason is true visually, but the assertion is false in the sense that matters: "
           "polynomial regression is linear *in its parameters*, which is why it is still fitted "
           "by ordinary least squares. The nonlinearity lives in the basis expansion of the "
           "features, not in the model class. Answer: D."),

    ARItem("naive_bayes", ML,
           "Laplace (add-one) smoothing is applied in Naive Bayes to improve computational "
           "efficiency.",
           "Without smoothing, a single unseen feature-class combination gives a zero likelihood "
           "that annihilates the entire product.",
           False, True, False,
           "The reason is true and is the actual motivation. The assertion is false: smoothing "
           "exists for numerical/statistical robustness, not speed - if anything it adds work. "
           "Answer: D."),

    ARItem("decision_trees", ML,
           "Pruning a decision tree usually increases its training accuracy.",
           "Pruning removes branches that provide little generalisation benefit.",
           False, True, False,
           "The reason is true. The assertion is false: pruning almost always *decreases* training "
           "accuracy - an unpruned tree can reach 100% on training data - while improving "
           "validation accuracy. Confusing the two is a common exam error. Answer: D."),

    ARItem("svm", ML,
           "A larger C parameter in a soft-margin SVM produces a wider margin.",
           "C controls the penalty applied to margin violations.",
           False, True, False,
           "The reason is true. The assertion is false and inverted: a large C penalises "
           "violations heavily, so the optimiser prefers a *narrower* margin that misclassifies "
           "fewer training points (lower bias, higher variance). Small C gives a wider, more "
           "tolerant margin. Answer: D."),

    ARItem("cross_validation", ML,
           "Stratified k-fold cross-validation is preferable for imbalanced classification.",
           "Stratification guarantees each fold has the same number of samples.",
           True, False, False,
           "The assertion is true. The reason is false: stratification preserves the *class "
           "proportions* in every fold, not the fold sizes (plain k-fold already makes those "
           "nearly equal). Without it a rare class may be absent from some fold entirely. "
           "Answer: C."),

    ARItem("apriori", ML,
           "A rule with high confidence is always an interesting rule.",
           "Confidence measures the conditional probability of the consequent given the "
           "antecedent.",
           False, True, False,
           "The reason is the correct definition. The assertion is false: if the consequent is "
           "very frequent on its own, confidence is high for trivial reasons. Lift, which "
           "compares the confidence to the consequent's base rate, is what identifies an "
           "interesting rule. Answer: D."),

    ARItem("rl_basics", ML,
           "A reinforcement learning agent should always take the action with the highest current "
           "estimated value.",
           "Choosing the highest-value action maximises the immediate expected reward under the "
           "current estimates.",
           False, True, False,
           "The reason is true for the *current* estimates. The assertion is false: pure "
           "exploitation may lock onto a suboptimal action whose true value was never discovered, "
           "because the estimates themselves depend on how often each action was tried. This is "
           "the exploration-exploitation tradeoff (epsilon-greedy, UCB, Thompson sampling). "
           "Answer: D."),

    ARItem("kmeans", ML,
           "The elbow method identifies the mathematically optimal number of clusters by "
           "minimising WCSS.",
           "WCSS decreases monotonically as k increases.",
           False, True, False,
           "The reason is true and is exactly why the assertion is false: since WCSS is minimised "
           "by k = n (every point its own cluster), you cannot select k by minimising it. The "
           "elbow method is a heuristic that looks for a diminishing-returns kink, not an "
           "optimisation. Answer: D."),

    ARItem("forward_propagation", DL,
           "The bias term in a neural network layer can be omitted without loss of expressive "
           "power.",
           "The bias shifts the pre-activation, allowing the activation boundary to move away "
           "from the origin.",
           False, True, False,
           "The reason is true and shows why the assertion is false: without a bias every "
           "hyperplane is forced through the origin, which strictly reduces the set of functions "
           "the layer can represent. Answer: D."),

    ARItem("loss_functions", DL,
           "Mean squared error is the appropriate loss for binary classification with a sigmoid "
           "output.",
           "MSE is differentiable and penalises large errors more heavily than small ones.",
           False, True, False,
           "The reason is true. The assertion is false: combining MSE with a sigmoid yields a "
           "gradient containing the factor sigma'(z), which vanishes when the unit is saturated "
           "and confidently wrong - exactly when a large update is needed. Cross-entropy cancels "
           "that factor, giving the clean (p - y) gradient. Answer: D."),

    ARItem("chain_rule", DL,
           "The vanishing gradient problem can occur in deep feedforward networks, not only in "
           "recurrent ones.",
           "Backpropagation multiplies local derivatives along the path from the loss to the "
           "parameter, so many sub-unit factors compound.",
           True, True, True,
           "Both true and the reason is the mechanism - depth in space and depth in time cause the "
           "same compounding. Answer: A."),

    ARItem("convolution", DL,
           "Increasing the number of filters in a convolutional layer increases the spatial "
           "resolution of its output.",
           "Each filter produces one output feature map.",
           False, True, False,
           "The reason is true. The assertion is false and confuses two dimensions: the filter "
           "count sets the output *depth* (channels), whereas the spatial resolution is fixed by "
           "the input size, kernel, stride and padding. Answer: D. Dimension confusion is heavily "
           "penalised in this exam."),

    ARItem("cnn_parameter_count", DL,
           "A convolutional layer's parameter count depends on the input image resolution.",
           "The same kernel weights are reused at every spatial location.",
           False, True, False,
           "The reason is true and is exactly why the assertion is false: weight sharing makes the "
           "parameter count (K*K*C_in + 1)*C_out, entirely independent of H and W. Only the "
           "*compute* scales with resolution. Answer: D."),

    ARItem("receptive_field", DL,
           "Stacking two 3x3 convolutions gives the same receptive field as one 5x5 convolution.",
           "Two stacked 3x3 layers use fewer parameters than a single 5x5 layer with the same "
           "channel counts.",
           True, True, False,
           "Both statements are true (RF = 5 in both cases; 2*9 = 18 versus 25 weights per "
           "channel pair), but the reason does not explain the assertion - the parameter saving is "
           "a separate consequence, not the cause of the equal receptive field. The VGG design "
           "argument uses both facts, plus the extra nonlinearity between the two layers. Answer: B."),

    ARItem("transfer_learning", DL,
           "When fine-tuning a pretrained network you should use a larger learning rate than when "
           "training from scratch.",
           "The pretrained weights are already close to a good solution.",
           False, True, False,
           "The reason is true and implies the opposite of the assertion: precisely because the "
           "weights are already good, a large learning rate would destroy the pretrained features "
           "(catastrophic forgetting). Fine-tuning uses a *smaller* learning rate, often with "
           "layer-wise decay. Answer: D."),

    ARItem("rnn", DL,
           "Increasing the hidden state size of an RNN solves the vanishing gradient problem.",
           "A larger hidden state can store more information about the sequence.",
           False, True, False,
           "The reason is true in terms of capacity. The assertion is false: vanishing gradients "
           "arise from the repeated multiplication of Jacobians through time, which is a property "
           "of the recurrence, not of the state width. The remedy is architectural (gating, "
           "additive paths) or algorithmic. Answer: D."),

    ARItem("attention", DL,
           "Attention weights provide a faithful explanation of which inputs caused the model's "
           "prediction.",
           "Attention weights are non-negative and sum to one over the input positions.",
           False, True, False,
           "The reason is a true property of the softmax. The assertion is false, or at least not "
           "established: attention weights are one component of a deep computation, alternative "
           "weight configurations can produce the same output, and the literature treats "
           "'attention is not explanation' as an open critique. Answer: D."),

    ARItem("self_vs_cross_attention", DL,
           "In cross-attention the queries, keys and values all come from the decoder.",
           "Cross-attention lets the decoder incorporate information from the encoder.",
           False, True, False,
           "The reason is true and directly contradicts the assertion: in cross-attention the "
           "queries come from the decoder while the keys and values come from the *encoder* "
           "output - that is the only way encoder information can enter. Answer: D."),

    ARItem("gpt", DL,
           "In-context learning updates the model's weights based on the examples in the prompt.",
           "Providing examples in the prompt can substantially improve a large language model's "
           "accuracy on a task.",
           False, True, False,
           "The reason is true and is the observed phenomenon. The assertion is false: in-context "
           "learning performs no gradient update at all - the weights are frozen and the examples "
           "merely condition the forward pass. Answer: D."),

    ARItem("model_compression", DL,
           "Knowledge distillation trains a smaller student model to match a larger teacher's "
           "outputs.",
           "The teacher's soft probability distribution carries more information per sample than a "
           "hard one-hot label.",
           True, True, True,
           "Both true and the reason is the mechanism - the 'dark knowledge' in the relative "
           "probabilities of the wrong classes. Answer: A."),

    ARItem("underfitting", ML,
           "A model with high training error and similarly high validation error is overfitting.",
           "Overfitting is characterised by a large gap between training and validation "
           "performance.",
           False, True, False,
           "The reason is a correct characterisation of overfitting and rules the assertion out: "
           "high error on *both* sets with no gap is underfitting (high bias / insufficient "
           "capacity or training). Answer: D."),

    ARItem("lr_schedules", DL,
           "Learning rate warmup is used because large initial learning rates can destabilise "
           "training in the first iterations.",
           "At initialisation the gradient estimates and adaptive-optimiser moment estimates are "
           "poorly conditioned, so large steps can push the model into a bad region.",
           True, True, True,
           "Both true and the reason is the mechanism, particularly relevant to Adam with "
           "Transformers. Answer: A."),

    ARItem("gmm", ML,
           "A Gaussian Mixture Model assigns each point to exactly one component.",
           "A GMM models the data as a weighted sum of Gaussian densities.",
           False, True, False,
           "The reason is a true definition. The assertion is false: a GMM produces *soft* "
           "assignments - each point receives a responsibility for every component, and the hard "
           "assignment is only obtained afterwards by taking the argmax. Answer: D."),

    ARItem("hierarchical_clustering", ML,
           "Single-linkage clustering is prone to the chaining effect.",
           "Single linkage defines the distance between two clusters as the minimum distance "
           "between any pair of their members.",
           True, True, True,
           "Both true and the reason is the mechanism: one nearby bridging point is enough to "
           "merge two otherwise distant clusters into an elongated chain. Answer: A."),

    ARItem("ucb", ML,
           "UCB selects the arm with the highest observed average reward.",
           "UCB adds an exploration bonus that decreases as an arm is sampled more often.",
           False, True, False,
           "The reason is true and contradicts the assertion: UCB maximises the *sum* of the "
           "empirical mean and a confidence bonus, so a rarely tried arm can be selected despite a "
           "lower average. That is exactly its exploration mechanism. Answer: D."),

    ARItem("bag_of_words", ML,
           "The bag-of-words representation discards word order.",
           "It represents a document by the counts of its vocabulary terms, with no positional "
           "information.",
           True, True, True,
           "Both true and the reason is the mechanism - which is why 'dog bites man' and 'man "
           "bites dog' are identical under bag-of-words. Answer: A."),

    ARItem("neural_networks", DL,
           "A perceptron with a step activation can learn the XOR function.",
           "The XOR function is linearly separable in two dimensions.",
           False, False, False,
           "Both statements are false. XOR is not linearly separable, and a single perceptron can "
           "only realise a linear decision boundary, so it cannot represent XOR - the historical "
           "result that motivated hidden layers. Answer: E."),

    ARItem("dropout", DL,
           "Dropout should be applied to the output layer of a classifier.",
           "Dropout reduces overfitting wherever it is applied.",
           False, False, False,
           "Both are false. Dropping output units would randomly destroy class scores and corrupt "
           "the predicted distribution; dropout belongs in hidden layers. And it is not a "
           "universal improvement - applied badly (or together with batch normalisation in the "
           "wrong place) it can hurt both optimisation and accuracy. Answer: E."),

    ARItem("pca", ML,
           "PCA is a supervised technique that maximises class separability.",
           "PCA requires class labels in order to compute the projection directions.",
           False, False, False,
           "Both false. PCA is unsupervised and maximises projected *variance*, using only the "
           "covariance of the features; no labels are involved. The description given belongs to "
           "LDA. Answer: E."),

    ARItem("batch_normalization", DL,
           "Batch normalization makes a network's prediction for a single input independent of "
           "the other inputs in its batch during training.",
           "Batch normalization normalises each feature using statistics computed over the current "
           "minibatch.",
           False, True, False,
           "The reason is true and is precisely why the assertion is false: during training the "
           "normalisation statistics depend on the whole minibatch, so an example's output does "
           "depend on its batch-mates. Independence is only restored at inference, where running "
           "averages are used. Answer: D."),
]


def bank_for_topic(topic_id: str) -> list[ARItem]:
    return [i for i in AR_BANK if i.topic_id == topic_id]


def balanced_pick(pool: list[ARItem], rng: random.Random,
                  recent_keys: list[str] | None = None) -> ARItem:
    """Pick from a pool while steering the A-E answer distribution.

    Real papers are not 80% 'A'. This down-weights whichever pattern the student
    has just seen so the option letter itself carries no information.
    """
    recent = recent_keys or []
    weights = []
    for item in pool:
        w = 1.0
        if item.key == "A":
            w *= 0.55  # the bank is A-rich; balance it at draw time
        w *= 0.4 ** recent.count(item.key)
        weights.append(w)
    total = sum(weights) or 1.0
    r = rng.random() * total
    acc = 0.0
    for item, w in zip(pool, weights):
        acc += w
        if r <= acc:
            return item
    return pool[-1]


def _to_question(item: ARItem, qid: str, source_basis: str = "bank") -> Question:
    return Question(
        id=qid,
        topic=item.topic_id,
        category=Category.DL if item.category == DL else Category.ML,
        question_type=QuestionType.ASSERTION_REASON,
        difficulty=item.difficulty,
        priority=Priority.CRITICAL,
        prompt=(f"**Assertion (A):** {item.assertion}\n\n"
                f"**Reason (R):** {item.reason}\n\n"
                "Select the correct option."),
        options=OPTIONS,
        correct_option=item.key,
        model_answer=item.explanation,
        assertion=item.assertion,
        reason=item.reason,
        assertion_truth=item.a_true,
        reason_truth=item.r_true,
        reason_explains_assertion=item.explains,
        estimated_time=120,
        source_basis=source_basis,
        expected_concepts=[],
    )


def generate_assertion_reason(
    topic_id: str,
    topic_name: str = "",
    category: str = ML,
    context: str = "",
    exclude: set[str] | None = None,
    use_llm: bool = True,
    seed: int | None = None,
    recent_keys: list[str] | None = None,
    min_difficulty: int = 1,
) -> Question | None:
    """Produce an assertion-reason question, preferring the LLM when configured."""
    exclude = exclude or set()
    rng = random.Random(seed)

    if use_llm:
        q = _llm_assertion_reason(topic_id, topic_name or topic_id, category, context)
        if q is not None:
            return q

    def _qid(i: ARItem) -> str:
        return f"ar:{topic_id}:{abs(hash(i.assertion)) & 0xffffff}"

    pool = [i for i in bank_for_topic(topic_id) if _qid(i) not in exclude]
    if not pool:
        pool = bank_for_topic(topic_id)
    if min_difficulty > 1:
        harder = [i for i in pool if i.difficulty >= min_difficulty]
        if harder:
            pool = harder
    if not pool:
        return None
    item = balanced_pick(pool, rng, recent_keys)
    return _to_question(item, _qid(item))


_AR_PROMPT = """Write ONE Assertion-Reason exam question on the topic: {topic}.

{context_block}

Requirements (this mirrors a real university exam):
- The Reason must be plausibly RELATED to the Assertion. Never trivial, never unrelated.
- Do NOT always make both statements true. Pick deliberately from these patterns:
  A: both true, reason explains assertion
  B: both true, reason does NOT explain assertion (e.g. reason is a mere definition)
  C: assertion true, reason false (e.g. reason states a common misconception)
  D: assertion false, reason true (a true fact that does not support the false claim)
  E: both false
- Target pattern for THIS question: {pattern}
- The statements must be technically precise and examinable at university level.
- The explanation must justify each truth value and, if applicable, why the reason does or does
  not explain the assertion.

Return JSON exactly:
{{"assertion": "...", "reason": "...", "assertion_truth": true/false,
  "reason_truth": true/false, "reason_explains_assertion": true/false,
  "explanation": "...", "difficulty": 4-6}}"""


def _llm_assertion_reason(topic_id: str, topic_name: str, category: str,
                          context: str = "") -> Question | None:
    llm = get_llm()
    if not llm.available:
        return None
    pattern = random.choice([
        "A (both true, reason explains)",
        "B (both true, reason does not explain)",
        "C (assertion true, reason false)",
        "D (assertion false, reason true)",
        "B (both true, reason does not explain)",
        "C (assertion true, reason false)",
        "D (assertion false, reason true)",
    ])
    context_block = (f"Ground the statements in this course material:\n{context}\n"
                     if context.strip() else
                     "Use standard university-level ML/DL knowledge.\n")
    data, resp = llm.complete_json(
        _AR_PROMPT.format(topic=topic_name, context_block=context_block, pattern=pattern),
        system=system_with_language(
            "You are a university examiner writing assertion-reason questions for a combined "
            "Machine Learning and Deep Learning final exam."
        ),
        temperature=0.8,
        max_tokens=900,
    )
    if not isinstance(data, dict):
        log.info("AR generation fell back to bank: %s", resp.error)
        return None
    try:
        item = ARItem(
            topic_id=topic_id,
            category=category,
            assertion=str(data["assertion"]).strip(),
            reason=str(data["reason"]).strip(),
            a_true=bool(data["assertion_truth"]),
            r_true=bool(data["reason_truth"]),
            explains=bool(data["reason_explains_assertion"]),
            explanation=str(data.get("explanation", "")).strip(),
            difficulty=int(data.get("difficulty", 5)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        log.info("malformed AR payload (%s); using bank", exc)
        return None
    if not item.assertion or not item.reason:
        return None
    # a reason that cannot explain a false assertion must not be flagged as explaining it
    if not item.a_true and item.explains:
        item.explains = False
    return _to_question(item, f"ar:{topic_id}:llm:{abs(hash(item.assertion)) & 0xffffff}",
                        source_basis="llm")


def evaluate_assertion_reason(question: Question, chosen: str) -> dict[str, Any]:
    """Grade an A-E choice and explain each truth flag."""
    correct = (chosen or "").strip().upper()[:1]
    expected = (question.correct_option or "").upper()
    is_right = correct == expected
    breakdown = [
        f"Assertion is **{'TRUE' if question.assertion_truth else 'FALSE'}**",
        f"Reason is **{'TRUE' if question.reason_truth else 'FALSE'}**",
    ]
    if question.assertion_truth and question.reason_truth:
        breakdown.append(
            "The Reason **does** correctly explain the Assertion"
            if question.reason_explains_assertion
            else "The Reason does **not** explain the Assertion (both are true but unrelated "
                 "as cause and effect)"
        )
    return {
        "correct": is_right,
        "chosen": correct,
        "expected": expected,
        "breakdown": breakdown,
        "explanation": question.model_answer,
    }


def bank_stats() -> dict[str, Any]:
    by_key: dict[str, int] = {}
    for item in AR_BANK:
        by_key[item.key] = by_key.get(item.key, 0) + 1
    return {"total": len(AR_BANK), "by_answer": by_key,
            "topics": len({i.topic_id for i in AR_BANK})}
