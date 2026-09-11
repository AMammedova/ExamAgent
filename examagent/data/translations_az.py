"""Azerbaijani wording for the assertion-reason bank, written by hand.

The bank's statements are fixed English, which is what makes the offline
generators exact - but it also meant a student working in Azerbaijani met
English questions whenever the LLM was unavailable. Translating them at
runtime needed the LLM too, so with no API credit there was no way through
at all.

These translations are therefore checked in as data: the app now serves
Azerbaijani closed-form questions with no API call and no API budget.

Rules followed throughout, because the answer key depends on the exact claim:

* Negations, quantifiers ("always", "never", "only", "each") and hedges are
  preserved exactly. A statement that is false in English must stay false.
* Technical names stay in English (overfitting, gradient descent, ReLU); the
  sentence around them is Azerbaijani.
* Nothing is rephrased into something easier - a bank statement is written to
  be decidable, and smoothing it would blunt that.
"""
from __future__ import annotations

#: english statement -> Azerbaijani
STATEMENTS: dict[str, str] = {
    "A model with a large gap between training and validation accuracy is overfitting.":
        "Training və validation accuracy arasında böyük fərq olan model overfitting edir.",
    "Overfitting occurs when a model has insufficient capacity to represent the "
    "underlying function.":
        "Overfitting modelin əsas funksiyanı təmsil etmək üçün kifayət qədər "
        "capacity-si olmadıqda baş verir.",
    "Adding more training data typically reduces overfitting.":
        "Daha çox training data əlavə etmək adətən overfitting-i azaldır.",
    "More data makes it harder for a fixed-capacity model to memorise the training "
    "set, so it must rely on structure that generalises.":
        "Daha çox data sabit capacity-li modelin training set-i əzbərləməsini "
        "çətinləşdirir, ona görə model generalize edən strukturlara söykənməli olur.",
    "Accuracy can be a misleading metric on an imbalanced dataset.":
        "Accuracy imbalanced dataset üzərində yanıldıcı metrika ola bilər.",
    "A classifier that always predicts the majority class can achieve high accuracy "
    "while having zero recall on the minority class.":
        "Həmişə majority class-ı proqnozlaşdıran classifier yüksək accuracy əldə edə "
        "bilər, lakin minority class üzrə recall-u sıfır olur.",
    "Increasing the classification threshold generally increases precision.":
        "Classification threshold-unu artırmaq ümumiyyətlə precision-u artırır.",
    "A higher threshold means the classifier only predicts positive when it is more "
    "confident, so fewer false positives are produced.":
        "Daha yüksək threshold o deməkdir ki, classifier yalnız daha əmin olduqda "
        "positive proqnoz verir, beləliklə daha az false positive yaranır.",
    "In medical screening, recall is usually prioritised over precision.":
        "Tibbi screening-də adətən recall precision-dan üstün tutulur.",
    "Recall is defined as TP/(TP+FN).":
        "Recall TP/(TP+FN) kimi təyin olunur.",
    "Decision trees do not require feature scaling.":
        "Decision trees feature scaling tələb etmir.",
    "Tree splits are chosen by thresholding a single feature at a time, and any "
    "monotone rescaling preserves the ordering of values.":
        "Tree split-ləri hər dəfə tək bir feature-a threshold tətbiq etməklə seçilir "
        "və istənilən monoton rescaling dəyərlərin sıralanmasını qoruyur.",
    "Leave-one-out cross-validation gives a nearly unbiased estimate of "
    "generalisation error.":
        "Leave-one-out cross-validation generalisation error-un demək olar ki, "
        "unbiased qiymətləndirməsini verir.",
    "Leave-one-out cross-validation is computationally cheaper than 5-fold "
    "cross-validation.":
        "Leave-one-out cross-validation hesablama baxımından 5-fold "
        "cross-validation-dan ucuzdur.",
    "L1 regularization can perform feature selection.":
        "L1 regularization feature selection həyata keçirə bilər.",
    "The L1 penalty is non-differentiable at zero, and its constant-magnitude "
    "gradient can drive coefficients exactly to zero.":
        "L1 penalty sıfırda differensiallanmır və sabit ölçülü gradient-i "
        "coefficient-ləri dəqiq sıfıra apara bilər.",
    "An SVM's decision boundary is determined only by the support vectors.":
        "SVM-in decision boundary-si yalnız support vector-lar tərəfindən müəyyən "
        "olunur.",
    "The SVM optimisation maximises the margin, and only points on or inside the "
    "margin have non-zero dual coefficients.":
        "SVM optimizasiyası margin-i maksimallaşdırır və yalnız margin üzərində və "
        "ya onun içində olan nöqtələrin dual coefficient-i sıfırdan fərqlidir.",
    "The kernel trick lets an SVM find a nonlinear decision boundary without "
    "explicitly computing high-dimensional feature vectors.":
        "Kernel trick SVM-ə yüksək ölçülü feature vector-ları açıq şəkildə "
        "hesablamadan qeyri-xətti decision boundary tapmağa imkan verir.",
    "A kernel function computes the inner product of two points in the feature space "
    "directly from their original coordinates.":
        "Kernel funksiyası iki nöqtənin feature space-dəki skalyar hasilini onların "
        "ilkin koordinatlarından birbaşa hesablayır.",
    "Random forests reduce the variance of decision trees.":
        "Random forests decision trees-in variance-ını azaldır.",
    "Random forests grow each tree to full depth on the entire training set without "
    "any randomisation.":
        "Random forests hər ağacı heç bir randomizasiya olmadan bütün training set "
        "üzərində tam dərinliyə qədər böyüdür.",
    "KNN performance degrades in very high-dimensional spaces.":
        "KNN-in performansı çox yüksək ölçülü fəzalarda pisləşir.",
    "In high dimensions, distances between points concentrate, so the nearest and "
    "farthest neighbours become nearly equidistant.":
        "Yüksək ölçülərdə nöqtələr arasındakı məsafələr cəmləşir, ona görə ən yaxın "
        "və ən uzaq neighbour demək olar ki, eyni məsafədə olur.",
    "Logistic regression is a linear model.":
        "Logistic regression xətti modeldir.",
    "Logistic regression applies a nonlinear sigmoid function to its output.":
        "Logistic regression öz çıxışına qeyri-xətti sigmoid funksiyası tətbiq edir.",
    "The principal components produced by PCA are mutually orthogonal.":
        "PCA-nın yaratdığı principal component-lər qarşılıqlı ortoqonaldır.",
    "They are eigenvectors of a real symmetric covariance matrix, whose eigenvectors "
    "for distinct eigenvalues are orthogonal.":
        "Onlar real simmetrik covariance matrisinin eigenvector-larıdır və fərqli "
        "eigenvalue-lara uyğun eigenvector-lar ortoqonaldır.",
    "The result of K-Means can change if you rerun it with a different random seed.":
        "K-Means-in nəticəsi fərqli random seed ilə yenidən işlədildikdə dəyişə bilər.",
    "K-Means minimises a non-convex objective and converges to a local optimum "
    "determined by the initialisation.":
        "K-Means qeyri-konveks məqsəd funksiyasını minimallaşdırır və initialisation "
        "ilə müəyyən olunan lokal optimuma yığılır.",
    "DBSCAN can find non-convex clusters that K-Means cannot.":
        "DBSCAN K-Means-in tapa bilmədiyi qeyri-konveks cluster-ləri tapa bilir.",
    "DBSCAN grows clusters by connecting density-reachable points rather than "
    "assigning points to the nearest centroid.":
        "DBSCAN cluster-ləri nöqtələri ən yaxın centroid-ə təyin etmək əvəzinə "
        "density-reachable nöqtələri birləşdirməklə böyüdür.",
    "Agglomerative hierarchical clustering requires the number of clusters to be "
    "specified before the algorithm runs.":
        "Agglomerative hierarchical clustering alqoritm işə düşməzdən əvvəl cluster "
        "sayının göstərilməsini tələb edir.",
    "A dendrogram records the full merge history, so any number of clusters can be "
    "obtained afterwards by cutting it at a chosen height.":
        "Dendrogram bütün birləşmə tarixçəsini saxlayır, ona görə sonradan onu "
        "seçilmiş hündürlükdə kəsməklə istənilən sayda cluster almaq olar.",
    "The EM algorithm is guaranteed to find the global maximum of the likelihood.":
        "EM alqoritmi likelihood-un qlobal maksimumunu tapmağa zəmanət verir.",
    "Each EM iteration is guaranteed not to decrease the observed-data "
    "log-likelihood.":
        "Hər EM iterasiyası observed-data log-likelihood-un azalmamasına zəmanət "
        "verir.",
    "A very flexible model always generalises better than a simpler one.":
        "Çox çevik model həmişə daha sadə modeldən yaxşı generalize edir.",
    "Increasing model complexity reduces bias.":
        "Model complexity-sini artırmaq bias-ı azaldır.",
    "Boosting can be more sensitive to label noise than bagging.":
        "Boosting label noise-a bagging-dən daha həssas ola bilər.",
    "Boosting repeatedly increases the weight of misclassified samples, so "
    "persistently mislabelled points attract disproportionate attention.":
        "Boosting səhv təsnif edilmiş nümunələrin çəkisini təkrar-təkrar artırır, "
        "ona görə davamlı səhv etiketlənmiş nöqtələr qeyri-mütənasib diqqət cəlb edir.",
    "The bandwidth of a kernel density estimator controls a bias-variance tradeoff.":
        "Kernel density estimator-un bandwidth-i bias-variance tradeoff-unu idarə edir.",
    "A small bandwidth produces a spiky estimate that follows individual samples, "
    "while a large bandwidth oversmooths and blurs genuine structure.":
        "Kiçik bandwidth ayrı-ayrı nümunələri izləyən iti qiymətləndirmə verir, böyük "
        "bandwidth isə həddindən artıq hamarlayıb əsl strukturu bulanıqlaşdırır.",
    "One-hot encoding should be preferred over integer label encoding for nominal "
    "categorical features in a linear model.":
        "Xətti modeldə nominal categorical feature-lər üçün one-hot encoding integer "
        "label encoding-dən üstün tutulmalıdır.",
    "Integer encoding imposes an artificial ordering and spacing between categories "
    "that the model will interpret as meaningful.":
        "Integer encoding kateqoriyalar arasında süni sıralanma və məsafə yaradır və "
        "model bunu mənalı kimi şərh edir.",
    "Backpropagation computes gradients more efficiently than evaluating each partial "
    "derivative numerically.":
        "Backpropagation gradient-ləri hər xüsusi törəməni ədədi qiymətləndirməkdən "
        "daha səmərəli hesablayır.",
    "Backpropagation applies the chain rule in reverse, reusing intermediate results "
    "so all parameter gradients are obtained in roughly one backward pass.":
        "Backpropagation chain rule-u tərsinə tətbiq edir və aralıq nəticələri təkrar "
        "istifadə edir, beləliklə bütün parametr gradient-ləri təxminən bir backward "
        "pass-da alınır.",
    "Backpropagation is a learning algorithm that decides how the weights should "
    "change.":
        "Backpropagation weight-lərin necə dəyişməli olduğuna qərar verən öyrənmə "
        "alqoritmidir.",
    "Backpropagation efficiently computes the gradient of the loss with respect to "
    "every parameter.":
        "Backpropagation loss-un hər parametrə görə gradient-ini səmərəli hesablayır.",
    "Stochastic gradient descent can escape some poor local minima that full-batch "
    "gradient descent would settle into.":
        "Stochastic gradient descent full-batch gradient descent-in ilişib qalacağı "
        "bəzi pis lokal minimumlardan çıxa bilir.",
    "The gradient computed on a minibatch is a noisy estimate of the full gradient.":
        "Minibatch üzərində hesablanan gradient tam gradient-in küylü "
        "qiymətləndirməsidir.",
    "A learning rate that is too large can cause the loss to increase.":
        "Həddindən artıq böyük learning rate loss-un artmasına səbəb ola bilər.",
    "With a large step size the update can overshoot the minimum and land at a point "
    "of higher loss, and the iterates may oscillate or diverge.":
        "Böyük addım ölçüsü ilə yeniləmə minimumu ötüb daha yüksək loss-lu nöqtəyə "
        "düşə bilər və iterasiyalar rəqs edə və ya dağıla bilər.",
    "He initialisation is preferred over Xavier initialisation for ReLU networks.":
        "ReLU şəbəkələri üçün He initialisation Xavier initialisation-dan üstün "
        "tutulur.",
    "ReLU sets roughly half of its inputs to zero, halving the variance of the "
    "activations, which He initialisation compensates for with a factor of 2 in the "
    "variance.":
        "ReLU girişlərinin təxminən yarısını sıfırlayır və activation-ların "
        "variance-ını yarıya endirir; He initialisation bunu variance-da 2 əmsalı ilə "
        "kompensasiya edir.",
    "Adam applies bias correction to its moment estimates.":
        "Adam öz moment qiymətləndirmələrinə bias correction tətbiq edir.",
    "The moving averages are initialised at zero, which biases them toward zero "
    "during the first few steps.":
        "Moving average-lər sıfırdan başladılır, bu da ilk bir neçə addımda onları "
        "sıfıra doğru meyilləndirir.",
    "Adam usually converges in fewer iterations than plain SGD.":
        "Adam adətən sadə SGD-dən daha az iterasiyada yığılır.",
    "Adam adapts a per-parameter step size using estimates of the first and second "
    "moments of the gradient.":
        "Adam gradient-in birinci və ikinci momentlərinin qiymətləndirmələrindən "
        "istifadə edərək hər parametr üçün addım ölçüsünü uyğunlaşdırır.",
    "The softmax function is typically used in the output layer of a multi-class "
    "classifier.":
        "Softmax funksiyası adətən multi-class classifier-in output layer-ində "
        "istifadə olunur.",
    "Softmax outputs are non-negative and sum to one, so they can be read as a "
    "probability distribution over the classes.":
        "Softmax çıxışları qeyri-mənfidir və cəmi birə bərabərdir, ona görə onları "
        "siniflər üzrə ehtimal paylanması kimi oxumaq olar.",
    "ReLU units can permanently stop learning during training.":
        "ReLU vahidləri training zamanı həmişəlik öyrənməyi dayandıra bilər.",
    "If a unit's pre-activation becomes negative for all inputs, its gradient is zero "
    "and its weights receive no further updates.":
        "Əgər bir vahidin pre-activation-ı bütün girişlər üçün mənfi olarsa, onun "
        "gradient-i sıfır olur və weight-ləri daha yenilənmir.",
    "Dropout increases training time to convergence.":
        "Dropout yığılmaya qədər training müddətini artırır.",
    "Dropout makes the effective network different at every step, so the gradient "
    "signal is noisier and more epochs are needed.":
        "Dropout hər addımda effektiv şəbəkəni fərqli edir, ona görə gradient siqnalı "
        "daha küylü olur və daha çox epoch tələb olunur.",
    "L2 weight decay changes the objective function being optimised.":
        "L2 weight decay optimallaşdırılan məqsəd funksiyasını dəyişir.",
    "It adds a penalty term lambda||w||^2 to the loss, so the optimiser minimises a "
    "different function than the original loss.":
        "O, loss-a lambda||w||^2 penalty həddi əlavə edir, beləliklə optimizator ilkin "
        "loss-dan fərqli funksiyanı minimallaşdırır.",
    "Batch normalization behaves differently at training and inference time.":
        "Batch normalization training və inference zamanı fərqli davranır.",
    "At inference the batch statistics are replaced by running averages accumulated "
    "during training.":
        "Inference zamanı batch statistikaları training boyunca toplanmış running "
        "average-lərlə əvəz olunur.",
    "A convolutional layer has far fewer parameters than a fully connected layer with "
    "the same input and output sizes.":
        "Convolutional layer eyni giriş və çıxış ölçülü fully connected layer-dən "
        "xeyli az parametrə malikdir.",
    "The same kernel is reused at every spatial position instead of learning an "
    "independent weight for each pair of positions.":
        "Hər mövqe cütü üçün müstəqil weight öyrənmək əvəzinə eyni kernel bütün fəza "
        "mövqelərində təkrar istifadə olunur.",
    "Max pooling adds no trainable parameters to a network.":
        "Max pooling şəbəkəyə heç bir öyrənilən parametr əlavə etmir.",
    "Max pooling reduces the spatial dimensions of the feature map.":
        "Max pooling feature map-in fəza ölçülərini azaldır.",
    "Padding is used to prevent the spatial dimensions from shrinking after a "
    "convolution.":
        "Padding convolution-dan sonra fəza ölçülərinin kiçilməsinin qarşısını almaq "
        "üçün istifadə olunur.",
    "Without padding, a KxK kernel can only be centred on positions at least (K-1)/2 "
    "away from the border, so the output is smaller than the input.":
        "Padding olmadan KxK kernel yalnız sərhəddən ən azı (K-1)/2 uzaqlıqdakı "
        "mövqelərdə mərkəzləşə bilir, ona görə çıxış girişdən kiçik olur.",
    "Transfer learning is particularly valuable when the target dataset is small.":
        "Transfer learning hədəf dataset kiçik olduqda xüsusilə dəyərlidir.",
    "Early layers of a pretrained network encode generic features such as edges and "
    "textures that transfer across visual domains.":
        "Pretrained şəbəkənin ilkin layer-ləri kənarlar və teksturalar kimi ümumi "
        "feature-ləri kodlaşdırır və bunlar vizual domenlər arasında keçir.",
    "A recurrent neural network can process input sequences of variable length.":
        "Recurrent neural network dəyişən uzunluqlu giriş ardıcıllıqlarını emal edə "
        "bilər.",
    "The same weight matrices are applied at every timestep, so the number of "
    "parameters does not depend on the sequence length.":
        "Hər timestep-də eyni weight matrisləri tətbiq olunur, ona görə parametrlərin "
        "sayı ardıcıllığın uzunluğundan asılı deyil.",
    "The forget gate is the component that allows an LSTM to retain information over "
    "many timesteps.":
        "Forget gate LSTM-ə məlumatı çoxlu timestep boyunca saxlamağa imkan verən "
        "komponentdir.",
    "When the forget gate is close to 1 the cell state is carried forward almost "
    "unchanged, and the gradient along that path is multiplied by a value near 1 at "
    "each step.":
        "Forget gate 1-ə yaxın olduqda cell state demək olar ki, dəyişmədən ötürülür "
        "və həmin yol boyunca gradient hər addımda 1-ə yaxın dəyərə vurulur.",
    "A GRU has fewer parameters than an LSTM with the same hidden size.":
        "GRU eyni hidden size-lı LSTM-dən az parametrə malikdir.",
    "A GRU has no output gate and merges the forget and input gates into a single "
    "update gate, leaving three weight blocks instead of four.":
        "GRU-nun output gate-i yoxdur və forget ilə input gate-lərini tək bir update "
        "gate-də birləşdirir, beləliklə dörd əvəzinə üç weight bloku qalır.",
    "Attention allows a decoder to access information from any position of the input "
    "sequence.":
        "Attention decoder-ə giriş ardıcıllığının istənilən mövqeyindəki məlumata "
        "müraciət etməyə imkan verir.",
    "Attention computes a weighted sum over all encoder hidden states, with weights "
    "derived from the compatibility between the decoder query and each encoder key.":
        "Attention bütün encoder hidden state-ləri üzrə çəkili cəm hesablayır və "
        "çəkilər decoder query ilə hər encoder key arasındakı uyğunluqdan alınır.",
    "Transformers can be parallelised across sequence positions during training more "
    "effectively than RNNs.":
        "Transformer-lər training zamanı ardıcıllıq mövqeləri üzrə RNN-lərdən daha "
        "effektiv paralelləşdirilə bilər.",
    "Self-attention computes all pairwise interactions in a single matrix operation, "
    "with no dependence on the previous timestep's output.":
        "Self-attention bütün cüt-cüt qarşılıqlı təsirləri tək bir matris "
        "əməliyyatında hesablayır və əvvəlki timestep-in çıxışından asılı deyil.",
    "The self-attention operation has computational cost that grows quadratically "
    "with sequence length.":
        "Self-attention əməliyyatının hesablama xərci ardıcıllıq uzunluğu ilə "
        "kvadratik artır.",
    "Every token must compute a compatibility score with every other token, giving "
    "n^2 scores for a sequence of length n.":
        "Hər token hər digər token ilə uyğunluq balı hesablamalıdır, bu da n "
        "uzunluqlu ardıcıllıq üçün n^2 bal verir.",
    "BERT can be used directly to generate text autoregressively.":
        "BERT birbaşa mətni autoregressive şəkildə generasiya etmək üçün istifadə "
        "oluna bilər.",
    "BERT is pretrained with a masked language modelling objective using "
    "bidirectional context.":
        "BERT ikitərəfli kontekstdən istifadə edən masked language modelling məqsədi "
        "ilə pretrain olunur.",
    "GPT models use masked (causal) self-attention in every layer.":
        "GPT modelləri hər layer-də masked (causal) self-attention istifadə edir.",
    "Without masking, a position could attend to future tokens, which would leak the "
    "answer during next-token-prediction training.":
        "Masking olmadan bir mövqe gələcək token-lərə baxa bilərdi və bu, "
        "next-token-prediction training zamanı cavabı sızdırardı.",
    "Word embeddings can capture semantic relationships between words.":
        "Word embedding-lər sözlər arasındakı semantik əlaqələri tuta bilir.",
    "Embeddings are trained so that words appearing in similar contexts obtain nearby "
    "vectors, following the distributional hypothesis.":
        "Embedding-lər distributional hipotezə uyğun olaraq, oxşar kontekstlərdə "
        "görünən sözlərin yaxın vektorlar alması üçün öyrədilir.",
    "Vision Transformers typically require more training data than CNNs to reach "
    "comparable accuracy.":
        "Vision Transformer-lər müqayisə oluna bilən accuracy-yə çatmaq üçün adətən "
        "CNN-lərdən daha çox training data tələb edir.",
    "A ViT lacks the built-in locality and translation-equivariance priors of a "
    "convolution, so it must learn those regularities from data.":
        "ViT-də convolution-un daxili lokallıq və translation-equivariance ilkin "
        "fərziyyələri yoxdur, ona görə bu qanunauyğunluqları datadan öyrənməlidir.",
    "A classical encoder-decoder without attention struggles with long input "
    "sequences.":
        "Attention-suz klassik encoder-decoder uzun giriş ardıcıllıqları ilə çətinlik "
        "çəkir.",
    "The encoder compresses the entire input into a single fixed-size context vector.":
        "Encoder bütün girişi tək bir sabit ölçülü context vector-a sıxışdırır.",
    "Concatenation-based skip connections increase the channel count of a feature map "
    "while additive skip connections do not.":
        "Concatenation əsaslı skip connection-lar feature map-in channel sayını "
        "artırır, toplama əsaslı skip connection-lar isə artırmır.",
    "Concatenation stacks the two tensors along the channel dimension, whereas "
    "addition requires matching shapes and returns the same shape.":
        "Concatenation iki tensoru channel ölçüsü boyunca yığır, toplama isə uyğun "
        "ölçülər tələb edir və eyni ölçünü qaytarır.",
    "Detecting small objects is harder on deep, heavily downsampled feature maps.":
        "Kiçik obyektləri aşkarlamaq dərin, güclü downsample olunmuş feature map-lərdə "
        "daha çətindir.",
    "Deep feature maps have larger receptive fields.":
        "Dərin feature map-lərin receptive field-ləri daha böyükdür.",
    "Attention logits are divided by sqrt(d_k) before the softmax.":
        "Attention logit-ləri softmax-dan əvvəl sqrt(d_k)-ya bölünür.",
    "Dividing by sqrt(d_k) makes the attention weights sum to one.":
        "sqrt(d_k)-ya bölmək attention çəkilərinin cəmini birə bərabər edir.",
    "Early stopping acts as a form of regularisation.":
        "Early stopping regularisation-ın bir forması kimi çıxış edir.",
    "Stopping before convergence limits how far the weights can move from their "
    "initialisation, restricting the effective capacity of the model.":
        "Yığılmadan əvvəl dayandırmaq weight-lərin initialisation-dan nə qədər "
        "uzaqlaşa biləcəyini məhdudlaşdırır və modelin effektiv capacity-sini "
        "məhdudlaşdırır.",
    "A single hidden layer network with enough units can approximate any continuous "
    "function on a compact domain.":
        "Kifayət qədər vahidi olan tək hidden layer-li şəbəkə kompakt oblastda "
        "istənilən kəsilməz funksiyanı təxmin edə bilər.",
    "This guarantees that a single hidden layer is the most practical architecture "
    "for real problems.":
        "Bu, tək hidden layer-in real məsələlər üçün ən praktik arxitektura olduğuna "
        "zəmanət verir.",
    "Missing values should always be replaced with the column mean.":
        "Çatışmayan dəyərlər həmişə sütunun ortalaması ilə əvəz olunmalıdır.",
    "Mean imputation preserves the mean of the observed feature distribution.":
        "Mean imputation müşahidə olunan feature paylanmasının ortalamasını qoruyur.",
    "The test set may be used to choose the model's hyperparameters as long as it is "
    "only used once at the end.":
        "Test set yalnız sonda bir dəfə istifadə olunduğu müddətcə modelin "
        "hyperparameter-lərini seçmək üçün istifadə oluna bilər.",
    "Hyperparameters are not learned from the training data by the optimiser.":
        "Hyperparameter-lər optimizator tərəfindən training data-dan öyrənilmir.",
    "A high R^2 means the model will generalise well to new data.":
        "Yüksək R^2 modelin yeni dataya yaxşı generalize edəcəyi deməkdir.",
    "R^2 measures the proportion of the variance in the target explained by the "
    "model.":
        "R^2 hədəfdəki variance-ın model tərəfindən izah olunan payını ölçür.",
    "Polynomial regression is a nonlinear model.":
        "Polynomial regression qeyri-xətti modeldir.",
    "Its decision surface / fitted curve is not a straight line.":
        "Onun decision surface-i / uyğunlaşdırılmış əyrisi düz xətt deyil.",
    "Laplace (add-one) smoothing is applied in Naive Bayes to improve computational "
    "efficiency.":
        "Laplace (add-one) smoothing Naive Bayes-də hesablama səmərəliliyini artırmaq "
        "üçün tətbiq olunur.",
    "Without smoothing, a single unseen feature-class combination gives a zero "
    "likelihood that annihilates the entire product.":
        "Smoothing olmadan görünməmiş tək bir feature-class kombinasiyası sıfır "
        "likelihood verir və bu, bütün hasili sıfırlayır.",
    "Pruning a decision tree usually increases its training accuracy.":
        "Decision tree-ni pruning etmək adətən onun training accuracy-sini artırır.",
    "Pruning removes branches that provide little generalisation benefit.":
        "Pruning generalisation baxımından az fayda verən budaqları silir.",
    "A larger C parameter in a soft-margin SVM produces a wider margin.":
        "Soft-margin SVM-də daha böyük C parametri daha geniş margin yaradır.",
    "C controls the penalty applied to margin violations.":
        "C margin pozuntularına tətbiq olunan penalty-ni idarə edir.",
    "Stratified k-fold cross-validation is preferable for imbalanced classification.":
        "Stratified k-fold cross-validation imbalanced classification üçün daha "
        "məqsədəuyğundur.",
    "Stratification guarantees each fold has the same number of samples.":
        "Stratification hər fold-un eyni sayda nümunəyə malik olmasına zəmanət verir.",
    "A rule with high confidence is always an interesting rule.":
        "Yüksək confidence-li qayda həmişə maraqlı qaydadır.",
    "Confidence measures the conditional probability of the consequent given the "
    "antecedent.":
        "Confidence nəticənin şərt verildikdə şərti ehtimalını ölçür.",
    "A reinforcement learning agent should always take the action with the highest "
    "current estimated value.":
        "Reinforcement learning agenti həmişə cari qiymətləndirilmiş dəyəri ən yüksək "
        "olan hərəkəti seçməlidir.",
    "Choosing the highest-value action maximises the immediate expected reward under "
    "the current estimates.":
        "Ən yüksək dəyərli hərəkəti seçmək cari qiymətləndirmələr altında ani "
        "gözlənilən reward-u maksimallaşdırır.",
    "The elbow method identifies the mathematically optimal number of clusters by "
    "minimising WCSS.":
        "Elbow metodu WCSS-i minimallaşdırmaqla riyazi olaraq optimal cluster sayını "
        "müəyyən edir.",
    "WCSS decreases monotonically as k increases.":
        "k artdıqca WCSS monoton azalır.",
    "The bias term in a neural network layer can be omitted without loss of "
    "expressive power.":
        "Neural network layer-indəki bias həddi ifadə gücü itirilmədən buraxıla bilər.",
    "The bias shifts the pre-activation, allowing the activation boundary to move "
    "away from the origin.":
        "Bias pre-activation-ı sürüşdürür və activation sərhədinin başlanğıc "
        "nöqtəsindən uzaqlaşmasına imkan verir.",
    "Mean squared error is the appropriate loss for binary classification with a "
    "sigmoid output.":
        "Mean squared error sigmoid çıxışlı binary classification üçün uyğun loss-dur.",
    "MSE is differentiable and penalises large errors more heavily than small ones.":
        "MSE differensiallanandır və böyük səhvləri kiçiklərdən daha ağır "
        "cəzalandırır.",
    "The vanishing gradient problem can occur in deep feedforward networks, not only "
    "in recurrent ones.":
        "Vanishing gradient problemi yalnız recurrent şəbəkələrdə deyil, dərin "
        "feedforward şəbəkələrdə də baş verə bilər.",
    "Backpropagation multiplies local derivatives along the path from the loss to the "
    "parameter, so many sub-unit factors compound.":
        "Backpropagation loss-dan parametrə gedən yol boyunca lokal törəmələri vurur, "
        "ona görə vahiddən kiçik çoxsaylı əmsallar bir-birinə hasil olur.",
    "Increasing the number of filters in a convolutional layer increases the spatial "
    "resolution of its output.":
        "Convolutional layer-də filter sayını artırmaq onun çıxışının fəza "
        "resolution-unu artırır.",
    "Each filter produces one output feature map.":
        "Hər filter bir çıxış feature map-i yaradır.",
    "A convolutional layer's parameter count depends on the input image resolution.":
        "Convolutional layer-in parametr sayı giriş şəklinin resolution-undan asılıdır.",
    "The same kernel weights are reused at every spatial location.":
        "Eyni kernel weight-ləri bütün fəza mövqelərində təkrar istifadə olunur.",
    "Stacking two 3x3 convolutions gives the same receptive field as one 5x5 "
    "convolution.":
        "İki 3x3 convolution-u üst-üstə yığmaq bir 5x5 convolution ilə eyni receptive "
        "field verir.",
    "Two stacked 3x3 layers use fewer parameters than a single 5x5 layer with the "
    "same channel counts.":
        "Eyni channel saylı iki 3x3 layer tək bir 5x5 layer-dən az parametr istifadə "
        "edir.",
    "When fine-tuning a pretrained network you should use a larger learning rate than "
    "when training from scratch.":
        "Pretrained şəbəkəni fine-tune edərkən sıfırdan öyrətməyə nisbətən daha böyük "
        "learning rate istifadə etmək lazımdır.",
    "The pretrained weights are already close to a good solution.":
        "Pretrained weight-lər artıq yaxşı həllə yaxındır.",
    "Increasing the hidden state size of an RNN solves the vanishing gradient "
    "problem.":
        "RNN-in hidden state ölçüsünü artırmaq vanishing gradient problemini həll edir.",
    "A larger hidden state can store more information about the sequence.":
        "Daha böyük hidden state ardıcıllıq haqqında daha çox məlumat saxlaya bilər.",
    "Attention weights provide a faithful explanation of which inputs caused the "
    "model's prediction.":
        "Attention çəkiləri modelin proqnozuna hansı girişlərin səbəb olduğunun "
        "sədaqətli izahını verir.",
    "Attention weights are non-negative and sum to one over the input positions.":
        "Attention çəkiləri qeyri-mənfidir və giriş mövqeləri üzrə cəmi birə "
        "bərabərdir.",
    "In cross-attention the queries, keys and values all come from the decoder.":
        "Cross-attention-da query, key və value-ların hamısı decoder-dən gəlir.",
    "Cross-attention lets the decoder incorporate information from the encoder.":
        "Cross-attention decoder-ə encoder-dən gələn məlumatı daxil etməyə imkan verir.",
    "In-context learning updates the model's weights based on the examples in the "
    "prompt.":
        "In-context learning prompt-dakı nümunələrə əsasən modelin weight-lərini "
        "yeniləyir.",
    "Providing examples in the prompt can substantially improve a large language "
    "model's accuracy on a task.":
        "Prompt-da nümunələr vermək böyük dil modelinin tapşırıq üzrə accuracy-sini "
        "əhəmiyyətli dərəcədə yaxşılaşdıra bilər.",
    "Knowledge distillation trains a smaller student model to match a larger "
    "teacher's outputs.":
        "Knowledge distillation daha kiçik student modelini daha böyük teacher-in "
        "çıxışlarına uyğunlaşmaq üçün öyrədir.",
    "The teacher's soft probability distribution carries more information per sample "
    "than a hard one-hot label.":
        "Teacher-in yumşaq ehtimal paylanması hər nümunə üçün sərt one-hot etiketdən "
        "daha çox məlumat daşıyır.",
    "A model with high training error and similarly high validation error is "
    "overfitting.":
        "Yüksək training error və ona bənzər yüksək validation error-a malik model "
        "overfitting edir.",
    "Overfitting is characterised by a large gap between training and validation "
    "performance.":
        "Overfitting training və validation performansı arasındakı böyük fərqlə "
        "xarakterizə olunur.",
    "Learning rate warmup is used because large initial learning rates can "
    "destabilise training in the first iterations.":
        "Learning rate warmup ona görə istifadə olunur ki, böyük ilkin learning rate "
        "ilk iterasiyalarda training-i qeyri-sabit edə bilər.",
    "At initialisation the gradient estimates and adaptive-optimiser moment estimates "
    "are poorly conditioned, so large steps can push the model into a bad region.":
        "Initialisation zamanı gradient qiymətləndirmələri və adaptive optimizatorun "
        "moment qiymətləndirmələri zəif şərtlənmiş olur, ona görə böyük addımlar "
        "modeli pis oblasta ata bilər.",
    "A Gaussian Mixture Model assigns each point to exactly one component.":
        "Gaussian Mixture Model hər nöqtəni dəqiq bir komponentə təyin edir.",
    "A GMM models the data as a weighted sum of Gaussian densities.":
        "GMM datanı Gaussian sıxlıqlarının çəkili cəmi kimi modelləşdirir.",
    "Single-linkage clustering is prone to the chaining effect.":
        "Single-linkage clustering chaining effect-ə meyillidir.",
    "Single linkage defines the distance between two clusters as the minimum distance "
    "between any pair of their members.":
        "Single linkage iki cluster arasındakı məsafəni onların üzvlərinin istənilən "
        "cütü arasındakı minimum məsafə kimi təyin edir.",
    "UCB selects the arm with the highest observed average reward.":
        "UCB müşahidə olunan orta reward-u ən yüksək olan qolu seçir.",
    "UCB adds an exploration bonus that decreases as an arm is sampled more often.":
        "UCB qol daha tez-tez seçildikcə azalan exploration bonusu əlavə edir.",
    "The bag-of-words representation discards word order.":
        "Bag-of-words təsviri söz sırasını atır.",
    "It represents a document by the counts of its vocabulary terms, with no "
    "positional information.":
        "O, sənədi lüğət terminlərinin sayları ilə təsvir edir və heç bir mövqe "
        "məlumatı saxlamır.",
    "A perceptron with a step activation can learn the XOR function.":
        "Step activation-lı perceptron XOR funksiyasını öyrənə bilər.",
    "The XOR function is linearly separable in two dimensions.":
        "XOR funksiyası iki ölçüdə xətti ayrılandır.",
    "Dropout should be applied to the output layer of a classifier.":
        "Dropout classifier-in output layer-inə tətbiq olunmalıdır.",
    "Dropout reduces overfitting wherever it is applied.":
        "Dropout tətbiq olunduğu hər yerdə overfitting-i azaldır.",
    "PCA is a supervised technique that maximises class separability.":
        "PCA sinif ayrılabilirliyini maksimallaşdıran supervised texnikadır.",
    "PCA requires class labels in order to compute the projection directions.":
        "PCA proyeksiya istiqamətlərini hesablamaq üçün sinif etiketləri tələb edir.",
    "Batch normalization makes a network's prediction for a single input independent "
    "of the other inputs in its batch during training.":
        "Batch normalization training zamanı şəbəkənin tək bir giriş üçün proqnozunu "
        "həmin batch-dəki digər girişlərdən asılı olmayan edir.",
    "Batch normalization normalises each feature using statistics computed over the "
    "current minibatch.":
        "Batch normalization hər feature-u cari minibatch üzərində hesablanmış "
        "statistikalardan istifadə edərək normallaşdırır.",
}

#: The fixed wording the offline generators wrap around those statements.
PHRASES: dict[str, str] = {
    "Which of the following statements are correct?":
        "Aşağıdakı ifadələrdən hansılar doğrudur?",
    "The statement holds as written.":
        "İfadə yazıldığı kimi doğrudur.",
    "The statement is false as written - check the clause that overstates it.":
        "İfadə yazıldığı kimi yanlışdır — onu şişirdən hissəyə diqqət yetir.",
    "All of them":
        "Hamısı",
    "True":
        "True",
    "False":
        "False",
}


def statement(text: str) -> str:
    """Azerbaijani for a bank statement, or the English back if untranslated."""
    return STATEMENTS.get(text) or PHRASES.get(text) or text


def coverage() -> dict[str, int]:
    """How much of the bank is translated - used by the tests to keep this file
    honest as the bank grows."""
    from ..services.exam_formats import all_facts

    unique = {f.text for f in all_facts()}
    return {
        "bank_statements": len(unique),
        "translated": sum(1 for t in unique if t in STATEMENTS),
    }
