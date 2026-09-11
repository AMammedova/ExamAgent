"""Yaddaş kitabçası - a written revision booklet in Azerbaijani.

Checked in as data rather than generated. Two reasons: it needs no API credit,
and a revision sheet is something to read and re-read, so it should be stable -
a page that says something slightly different every time it loads is not
something you can memorise from.

Each point is written to be examinable on its own: a single claim, precise
enough to answer a True/False or a multiple-choice item on. Technical names
stay in English, as they will on the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    title: str
    points: list[str] = field(default_factory=list)
    formulas: list[str] = field(default_factory=list)


ML_SECTIONS: list[Section] = [
    Section(
        "Əsaslar — öyrənmə növləri",
        [
            "Supervised learning etiketli data üzərində öyrənir: hər nümunənin giriş və "
            "gözlənilən çıxışı var.",
            "Unsupervised learning etiket olmadan datanın daxili strukturunu tapır "
            "(clustering, dimensionality reduction).",
            "Reinforcement learning agentin mühitlə qarşılıqlı təsirindən, reward "
            "siqnalı vasitəsilə öyrənməsidir — əvvəlcədən verilmiş etiketli cütlərdən yox.",
            "Semi-supervised learning az sayda etiketli və çox sayda etiketsiz "
            "nümunədən birlikdə istifadə edir.",
            "Modelin əsl məqsədi training data-nı əzbərləmək yox, görünməmiş data "
            "üzərində yaxşı işləməkdir (generalization).",
            "Inductive bias modelin datadan kənar gətirdiyi fərziyyələrdir; onsuz "
            "öyrənmə mümkün deyil.",
        ],
    ),
    Section(
        "Data hazırlığı",
        [
            "Feature scaling məsafə əsaslı (KNN, SVM, K-Means) və gradient əsaslı "
            "modellər üçün vacibdir; Decision trees və onların ansamblları üçün lazım deyil.",
            "Standardization ortalamanı 0, standart kənarlaşmanı 1 edir; "
            "normalization (min-max) dəyərləri [0,1] aralığına salır.",
            "Scaler YALNIZ training set üzərində fit edilməlidir — bütün data üzərində "
            "fit etmək data leakage-dir.",
            "Data leakage test/validation haqqında məlumatın training prosesinə "
            "sızmasıdır; nəticə real olmayan yüksək bal verir.",
            "Nominal categorical feature-lər üçün one-hot encoding istifadə olunur; "
            "integer encoding kateqoriyalar arasında süni sıra yaradır.",
            "Ordinal feature-lərdə (məsələn: aşağı/orta/yüksək) sıra mənalıdır, ona görə "
            "integer encoding uyğundur.",
            "Çatışmayan dəyərlər üçün mean/median imputation sadə üsuldur, lakin "
            "feature-in variance-ını süni azaldır.",
            "Imbalanced data-da stratified split sinif nisbətlərini hər hissədə qoruyur.",
            "Outlier-ləri split-dən ƏVVƏL silmək də leakage yarada bilər, çünki qərar "
            "bütün datanı görməklə verilir.",
        ],
    ),
    Section(
        "Regression",
        [
            "Linear regression çıxışı giriş feature-lərin xətti kombinasiyası kimi "
            "modelləşdirir və adətən MSE-ni minimallaşdırır.",
            "Ordinary least squares-in closed-form həlli var (normal equation), lakin "
            "böyük ölçülərdə gradient descent daha praktikdir.",
            "Polynomial regression parametrlərə görə hələ də xəttidir, amma giriş "
            "feature-lərinə görə qeyri-xətti əyri verir.",
            "Ridge (L2) əmsalları kiçildir, lakin dəqiq sıfıra endirmir.",
            "Lasso (L1) bəzi əmsalları dəqiq sıfıra endirir, ona görə feature selection "
            "effekti yaradır.",
            "Elastic Net L1 və L2-ni birləşdirir — korrelyasiyalı feature-lər olduqda "
            "faydalıdır.",
            "Regularization gücü (lambda / alpha) artdıqca model sadələşir: variance "
            "azalır, bias artır.",
            "R² hədəfdəki variance-ın model tərəfindən izah olunan hissəsidir; yüksək R² "
            "TƏK BAŞINA yaxşı generalization demək deyil.",
        ],
    ),
    Section(
        "Classification",
        [
            "Logistic regression xətti modeldir: xətti kombinasiyanın üzərinə sigmoid "
            "tətbiq edib ehtimal verir, decision boundary isə xətti qalır.",
            "Logistic regression-un loss funksiyası cross-entropy-dir, MSE deyil.",
            "KNN-in training mərhələsi yoxdur (lazy learner) — bütün hesablama proqnoz "
            "zamanı aparılır.",
            "KNN-də k kiçik olduqda model həssas olur (yüksək variance), k böyük olduqda "
            "hamarlanır (yüksək bias).",
            "KNN yüksək ölçülərdə pisləşir, çünki məsafələr bir-birinə yaxınlaşır "
            "(curse of dimensionality).",
            "Naive Bayes feature-lərin sinif verildikdə şərti müstəqil olduğunu fərz edir.",
            "Laplace (add-one) smoothing görünməmiş feature-class kombinasiyasının sıfır "
            "ehtimal verib bütün hasili sıfırlamasının qarşısını alır.",
            "Decision tree split-ləri Gini impurity və ya entropy/information gain ilə "
            "seçir.",
            "Pruning ağacı sadələşdirir: training accuracy AZALIR, generalization isə "
            "adətən yaxşılaşır.",
            "SVM margin-i maksimallaşdırır və decision boundary yalnız support "
            "vector-lardan asılıdır.",
            "Kernel trick yüksək ölçülü feature vektorlarını açıq hesablamadan "
            "qeyri-xətti sərhəd qurmağa imkan verir.",
            "Soft-margin SVM-də C böyüdükcə margin pozuntularına cəza artır və margin "
            "DARALIR (overfitting riski).",
        ],
    ),
    Section(
        "Model qiymətləndirmə",
        [
            "Data üç yerə bölünür: training (öyrətmə), validation (hyperparameter seçimi), "
            "test (yalnız sonda, bir dəfə).",
            "Test set hyperparameter seçmək üçün istifadə olunarsa, o artıq test set "
            "deyil — nəticə optimist olur.",
            "K-fold cross-validation datanı k hissəyə bölür, hər hissə bir dəfə "
            "validation rolunu oynayır.",
            "Stratified k-fold hər fold-da sinif nisbətlərini qoruyur — imbalanced "
            "data üçün vacibdir.",
            "Leave-one-out cross-validation demək olar ki, unbiased qiymət verir, lakin "
            "hesablama baxımından bahalıdır.",
            "Accuracy imbalanced data-da yanıldıcıdır: 95% majority sinifdə həmişə "
            "majority proqnozu 95% accuracy verir, recall isə sıfır olur.",
            "Precision: proqnoz etdiyin positive-lərin neçəsi doğrudan positive idi.",
            "Recall (sensitivity): əsl positive-lərin neçəsini tuta bildin.",
            "F1 precision və recall-un harmonik ortasıdır — biri çox aşağı olduqda F1 də "
            "aşağı olur.",
            "Tibbi screening kimi hallarda recall precision-dan üstün tutulur (xəstəni "
            "buraxmamaq daha vacibdir).",
            "Threshold-u artırmaq precision-u artırır, recall-u azaldır; azaltmaq əksini "
            "edir.",
            "ROC əyrisi TPR-i FPR-ə qarşı çəkir; AUC = 1.0 mükəmməl sıralama, 0.5 "
            "təsadüfi deməkdir.",
            "AUC threshold-dan asılı deyil — bütün threshold-lar üzrə performansı ölçür.",
            "Regression üçün əsas metrikalar MSE, RMSE, MAE və R²-dir.",
        ],
        [
            "Precision = TP / (TP + FP)",
            "Recall = TP / (TP + FN)",
            "F1 = 2 · (Precision · Recall) / (Precision + Recall)",
            "Accuracy = (TP + TN) / (TP + TN + FP + FN)",
        ],
    ),
    Section(
        "Overfitting, underfitting və bias-variance",
        [
            "Overfitting: training error aşağı, validation error yüksək — aralarında "
            "böyük fərq var.",
            "Underfitting: həm training, həm validation error yüksək və bir-birinə "
            "yaxındır.",
            "Underfitting yüksək bias və aşağı variance ilə əlaqəlidir.",
            "Overfitting yüksək variance ilə əlaqəlidir.",
            "Model complexity artdıqca bias azalır, variance artır — bu, bias-variance "
            "tradeoff-dur.",
            "Daha çox training data overfitting-i azaldır, underfitting-i yox.",
            "Overfitting ilə mübarizə: regularization, sadə model, daha çox data, "
            "early stopping, dropout, data augmentation.",
            "Underfitting ilə mübarizə: daha güclü model, daha çox feature, "
            "regularization-u azaltmaq, daha uzun training.",
            "Learning curve-də hər iki əyri yüksək və yaxındırsa — underfitting; "
            "aralarında böyük fərq varsa — overfitting.",
        ],
    ),
    Section(
        "Ensemble metodları",
        [
            "Bagging modelləri PARALEL öyrədir və bootstrap nümunələrindən istifadə edir "
            "— əsasən variance-ı azaldır.",
            "Boosting modelləri ARDICIL öyrədir, hər yeni model əvvəlkinin səhvlərini "
            "düzəldir — əsasən bias-ı azaldır.",
            "Random forest bagging + hər split-də təsadüfi feature alt çoxluğu "
            "istifadə edir, beləliklə ağaclar bir-birindən az asılı olur.",
            "Boosting label noise-a bagging-dən daha həssasdır, çünki səhv etiketli "
            "nöqtələrin çəkisi durmadan artır.",
            "Gradient boosting hər yeni ağacı qalıqların (residual) gradientinə uyğun "
            "qurur.",
            "XGBoost gradient boosting-in regularization və optimizasiya ilə "
            "gücləndirilmiş versiyasıdır.",
        ],
    ),
    Section(
        "Ölçü azaltma (Dimensionality reduction)",
        [
            "PCA unsupervised-dır — sinif etiketlərindən istifadə ETMİR.",
            "PCA variance-ı maksimallaşdıran ortoqonal istiqamətlər (principal "
            "components) tapır.",
            "Principal component-lər covariance matrisinin eigenvector-larıdır; "
            "eigenvalue həmin istiqamətdəki variance-ı göstərir.",
            "PCA-dan əvvəl data mütləq standartlaşdırılmalıdır, əks halda böyük miqyaslı "
            "feature dominant olur.",
            "LDA supervised-dır — siniflər arası ayrılabilirliyi maksimallaşdırır.",
            "t-SNE və UMAP əsasən vizuallaşdırma üçündür və lokal strukturu qoruyur; "
            "onlar məsafələri qlobal miqyasda qorumur.",
        ],
    ),
    Section(
        "Clustering",
        [
            "K-Means-də k əvvəlcədən verilməlidir — datadan öyrənilmir.",
            "K-Means qeyri-konveks məqsəd funksiyasını minimallaşdırır, ona görə fərqli "
            "random seed fərqli nəticə verə bilər.",
            "K-Means kürəvi (sferik), oxşar ölçülü cluster-ləri yaxşı tapır; uzanmış və "
            "ya qeyri-konveks formalarda zəifdir.",
            "Elbow metodu k seçmək üçün EVRİSTİKADIR — riyazi optimal cavab vermir "
            "(WCSS k artdıqca həmişə azalır).",
            "Silhouette score cluster-lərin nə qədər yaxşı ayrıldığını ölçür (-1-dən 1-ə).",
            "DBSCAN sıxlıq əsaslıdır: cluster sayı verilmir, outlier-ləri ayrıca "
            "işarələyir və qeyri-konveks formaları tapa bilir.",
            "Hierarchical clustering dendrogram qurur — cluster sayı sonradan onu kəsməklə "
            "seçilir, əvvəlcədən verilmir.",
            "Single linkage chaining effect-ə meyillidir (uzun zəncirvari cluster-lər).",
            "GMM hər nöqtəyə cluster-lərə aid olma EHTİMALI verir (soft assignment), "
            "K-Means isə sərt təyinat edir.",
            "EM alqoritmi hər iterasiyada log-likelihood-u azaltmır, lakin qlobal "
            "maksimuma zəmanət vermir.",
        ],
    ),
    Section(
        "Digər ML mövzuları",
        [
            "Association rules-da support qaydanın nə qədər tez-tez rast gəldiyini, "
            "confidence isə şərti ehtimalı ölçür.",
            "Yüksək confidence tək başına qaydanı maraqlı etmir — lift də nəzərə "
            "alınmalıdır.",
            "Reinforcement learning-də exploration/exploitation balansı vacibdir: "
            "həmişə ən yüksək dəyərli hərəkəti seçmək optimal deyil.",
            "Epsilon-greedy ehtimalla təsadüfi hərəkət seçərək exploration edir.",
            "UCB az sınanmış hərəkətlərə exploration bonusu əlavə edir.",
            "Bag-of-words sənədi söz saylarıyla təmsil edir və söz sırasını itirir.",
            "TF-IDF tez-tez rast gələn ümumi sözlərin çəkisini azaldır.",
        ],
    ),
]

DL_SECTIONS: list[Section] = [
    Section(
        "Neural network əsasları",
        [
            "Neuron girişlərin çəkili cəmini alır, bias əlavə edir və activation "
            "funksiyasından keçirir.",
            "Activation funksiyası olmadan neçə layer yığsan da, şəbəkə tək bir xətti "
            "çevrilməyə bərabər olur.",
            "Universal approximation teoremi: kifayət qədər neyronu olan tək hidden "
            "layer kompakt oblastda istənilən kəsilməz funksiyanı təxmin edə bilər — "
            "lakin bu, praktik olaraq ən yaxşı arxitektura olduğu anlamına gəlmir.",
            "Dərin şəbəkələr eyni funksiyanı daha az neyronla, iyerarxik feature-lərlə "
            "ifadə edə bilir.",
            "Bias həddi activation sərhədini başlanğıc nöqtəsindən sürüşdürür — onu "
            "silmək ifadə gücünü azaldır.",
        ],
    ),
    Section(
        "Activation funksiyaları",
        [
            "ReLU: max(0, x) — sadə, hesablaması ucuz, müsbət tərəfdə gradient sabitdir.",
            "ReLU neyronu bütün girişlər üçün mənfi pre-activation alırsa, gradient sıfır "
            "olur və neyron həmişəlik ölür (dying ReLU).",
            "Leaky ReLU mənfi tərəfdə kiçik mail verərək dying ReLU problemini azaldır.",
            "Sigmoid çıxışı (0,1) aralığındadır və böyük müsbət/mənfi girişlərdə doyur "
            "(saturation) — gradient sıfıra yaxınlaşır.",
            "Tanh çıxışı (-1,1) aralığındadır və sıfır mərkəzlidir, ona görə sigmoid-dən "
            "adətən yaxşıdır, lakin o da doyur.",
            "Softmax vektoru ehtimal paylanmasına çevirir: bütün çıxışlar qeyri-mənfidir "
            "və cəmi 1-dir — multi-class çıxış layer-i üçün.",
            "Binary classification-da çıxış layer-ində sigmoid, multi-class-da softmax "
            "istifadə olunur.",
        ],
    ),
    Section(
        "Loss funksiyaları",
        [
            "Regression üçün standart loss Mean Squared Error-dur.",
            "Binary classification üçün binary cross-entropy, multi-class üçün "
            "categorical cross-entropy istifadə olunur.",
            "Sigmoid çıxışlı classification-da MSE istifadə etmək zəif gradient verir — "
            "cross-entropy düzgün seçimdir.",
            "Sigmoid + cross-entropy birləşməsində çıxış layer-inin gradienti sadəcə "
            "(ŷ - y) olur — bu, imtahanda tez-tez soruşulur.",
            "MAE outlier-lərə MSE-dən daha davamlıdır, çünki səhvi kvadrata yüksəltmir.",
        ],
        [
            "Binary cross-entropy: L = -[y·ln(ŷ) + (1-y)·ln(1-ŷ)]",
            "MSE: L = (1/n)·Σ(y - ŷ)²",
            "Sigmoid + BCE üçün: ∂L/∂z = ŷ - y",
        ],
    ),
    Section(
        "Backpropagation və optimizasiya",
        [
            "Backpropagation chain rule-u tərsinə tətbiq edərək bütün parametrlərin "
            "gradientini təxminən bir backward pass-da hesablayır.",
            "Backpropagation gradient HESABLAYAN alqoritmdir; çəkiləri yeniləyən isə "
            "optimizator-dur (məsələn SGD).",
            "Forward pass-da aralıq nəticələr saxlanılır, çünki backward pass-da onlar "
            "lazım olur.",
            "Batch gradient descent bütün datanı, SGD tək nümunəni, mini-batch isə kiçik "
            "qrupu istifadə edir.",
            "Mini-batch gradienti tam gradientin küylü qiymətləndirməsidir; bu küy bəzi "
            "pis lokal minimumlardan çıxmağa kömək edə bilər.",
            "Learning rate çox böyük olarsa loss arta və ya dağıla bilər; çox kiçik "
            "olarsa training çox yavaş gedər.",
            "Momentum keçmiş gradientləri toplayaraq yeniləmələri hamarlayır və "
            "sürətləndirir.",
            "Adam momentum ilə per-parametr adaptiv addım ölçüsünü birləşdirir və "
            "moment qiymətləndirmələrinə bias correction tətbiq edir.",
            "Learning rate schedule training gedişində learning rate-i azaldır; warmup "
            "isə training-in ƏVVƏLİNDƏ kiçik dəyərdən başlayır.",
            "Epoch bütün training set üzərində bir tam keçiddir; iteration isə bir "
            "batch üzərində bir yeniləmədir.",
        ],
    ),
    Section(
        "Initialization və normalization",
        [
            "Bütün çəkiləri sıfırla başlatmaq olmaz — bütün neyronlar eyni gradienti "
            "alıb eyni qalır (symmetry problemi).",
            "Xavier/Glorot initialization sigmoid və tanh üçün, He initialization isə "
            "ReLU üçün uyğundur.",
            "He initialization variance-da 2 əmsalı saxlayır, çünki ReLU girişlərin "
            "təxminən yarısını sıfırlayır.",
            "Batch normalization hər feature-u cari mini-batch statistikaları ilə "
            "normallaşdırır və öyrənilən scale/shift parametrləri əlavə edir.",
            "Batch normalization training və inference zamanı FƏRQLİ davranır: "
            "inference-də batch statistikaları əvəzinə running average istifadə olunur.",
            "Batch normalization diqqətli initialization ehtiyacını azaldır, lakin "
            "tamamilə aradan qaldırmır — və o, çəkiləri yox, activation-ları normallaşdırır.",
            "Layer normalization batch ölçüsündən asılı deyil, ona görə Transformer və "
            "RNN-lərdə üstünlük təşkil edir.",
        ],
    ),
    Section(
        "Regularization",
        [
            "Dropout training zamanı neyronları təsadüfi söndürür, inference zamanı isə "
            "bütün neyronlar aktivdir.",
            "Dropout co-adaptation-ı azaldır və overfitting-in qarşısını alır, lakin "
            "yığılma üçün lazım olan epoch sayını artırır.",
            "Dropout adətən çıxış layer-inə TƏTBİQ OLUNMUR.",
            "L2 weight decay loss-a λ‖w‖² həddi əlavə edir — yəni optimallaşdırılan "
            "məqsəd funksiyasını dəyişir.",
            "Early stopping validation loss yaxşılaşmayanda dayandırır və regularization "
            "kimi işləyir.",
            "Data augmentation girişi etiketi dəyişməyəcək şəkildə çevirməklə effektiv "
            "data həcmini artırır.",
            "Augmentation seçimi tapşırıqdan asılıdır: rəqəm tanımada 180° fırlatma "
            "etiketi poza bilər (6 və 9).",
        ],
    ),
    Section(
        "Vanishing / exploding gradient",
        [
            "Vanishing gradient dərin şəbəkələrdə gradient-lərin geriyə getdikcə "
            "kiçilməsidir — yalnız RNN-lərdə deyil, dərin feedforward şəbəkələrdə də olur.",
            "Sigmoid və tanh doyduqda törəmələri kiçik olur və bu kiçik əmsallar "
            "bir-birinə vurularaq gradient-i söndürür.",
            "ReLU, residual connection və normalization vanishing gradient-i azaldır.",
            "Exploding gradient əks haldır: gradient-lər böyüyüb training-i dağıdır.",
            "Gradient clipping EXPLODING gradient üçündür — vanishing üçün deyil.",
            "LSTM-in cell state-i və forget gate-i uzun məsafəli gradienti qoruyur.",
        ],
    ),
    Section(
        "CNN — konvolyusiya şəbəkələri",
        [
            "Convolution eyni kernel çəkilərini bütün fəza mövqelərində tətbiq edir "
            "(weight sharing) — ona görə parametr sayı giriş ölçüsündən ASILI DEYİL.",
            "Convolutional layer eyni ölçülü fully connected layer-dən qat-qat az "
            "parametrə malikdir.",
            "Stride addım ölçüsüdür: stride artdıqca çıxışın fəza ölçüsü azalır.",
            "Padding çıxış ölçüsünün kiçilməsinin qarşısını alır; stride 1-də "
            "(k-1)/2 padding ölçünü dəyişməz saxlayır ('same' padding).",
            "Pooling öyrənilən parametr əlavə etmir və fəza ölçüsünü azaldır.",
            "Max pooling kiçik sürüşmələrə qarşı müəyyən invariantlıq verir.",
            "Filter sayını artırmaq çıxışın CHANNEL sayını artırır, fəza "
            "resolution-unu yox.",
            "Hər filter bir çıxış feature map-i yaradır.",
            "Dərin layer-lərin receptive field-i böyükdür və daha abstrakt feature-lər "
            "tutur; ilk layer-lər kənar və tekstura kimi ümumi feature-lər öyrənir.",
            "İki 3×3 convolution bir 5×5 ilə eyni receptive field verir, lakin daha az "
            "parametr istifadə edir.",
            "Dilated (atrous) convolution receptive field-i resolution-u azaltmadan "
            "genişləndirir.",
            "1×1 convolution channel sayını dəyişə bilir (fəza ölçüsünə toxunmadan).",
            "Residual (skip) connection gradientə qısa yol verir və çox dərin şəbəkələrin "
            "öyrədilməsini mümkün edir — parametr sayını azaltmaqla yox.",
            "Transfer learning hədəf dataset kiçik olduqda xüsusilə dəyərlidir: ilkin "
            "layer-lər dondurulur, son layer-lər yenidən öyrədilir.",
            "Fine-tuning zamanı adətən sıfırdan öyrətməyə nisbətən DAHA KİÇİK learning "
            "rate istifadə olunur.",
        ],
        [
            "Çıxış ölçüsü: O = ⌊(W - K + 2P) / S⌋ + 1",
            "Conv layer parametr sayı: (K × K × C_in + 1) × C_out",
            "Receptive field (stride 1, L layer, K kernel): 1 + L·(K-1)",
        ],
    ),
    Section(
        "Ardıcıllıqlar — RNN, LSTM, GRU",
        [
            "RNN hər timestep-də EYNİ çəki matrislərini tətbiq edir, ona görə parametr "
            "sayı ardıcıllığın uzunluğundan asılı deyil.",
            "Bu səbəbdən öyrədilmiş RNN training zamanı gördüyündən daha uzun "
            "ardıcıllıqla da işlədilə bilər.",
            "BPTT (backpropagation through time) ardıcıllığı açaraq gradient hesablayır.",
            "LSTM-in üç gate-i var: forget, input, output — və ayrıca cell state.",
            "Forget gate 1-ə yaxın olduqda cell state demək olar ki, dəyişmədən ötürülür "
            "və gradient qorunur.",
            "GRU-nun iki gate-i var (update və reset), ayrıca cell state yoxdur — "
            "ona görə LSTM-dən az parametrə malikdir.",
            "BiLSTM ardıcıllığı hər iki istiqamətdə oxuyur, ona görə mətn generasiya edən "
            "decoder kimi istifadə OLUNA BİLMƏZ (gələcəyi görür).",
            "Word embedding-lər sözləri sıx vektorlara çevirir və oxşar kontekstli sözlər "
            "yaxın vektorlar alır.",
            "word2vec kontekstdən asılı olmayan (static) embedding verir: 'bank' sözü hər "
            "yerdə eyni vektor alır.",
            "Seq2seq-də attention olmadan bütün mənbə ardıcıllığı tək bir sabit ölçülü "
            "context vector-a sıxılır — uzun cümlələrdə bu, əsas darboğazdır.",
        ],
    ),
    Section(
        "Attention və Transformer",
        [
            "Attention decoder-ə giriş ardıcıllığının istənilən mövqeyinə birbaşa "
            "müraciət etməyə imkan verir — seq2seq darboğazını aradan qaldırır.",
            "Scaled dot-product attention: Q və K-nın skalyar hasili √d_k-ya bölünür, "
            "sonra softmax tətbiq olunur.",
            "√d_k-ya bölmə böyük ölçülərdə skalyar hasillərin böyüyüb softmax-ı doyurmasının "
            "qarşısını alır — attention çəkilərinin cəmini 1 edən isə SOFTMAX-dır.",
            "Self-attention-da Q, K, V eyni mənbədən gəlir.",
            "Cross-attention-da Q decoder-dən, K və V isə encoder çıxışından gəlir — "
            "bu, mənbə məlumatının decoder-ə yeganə keçid yoludur.",
            "Multi-head attention fərqli alt-fəzalarda paralel attention hesablayır və "
            "nəticələri birləşdirir.",
            "Transformer-də recurrence yoxdur, ona görə mövqe məlumatı positional "
            "encoding ilə əlavə edilməlidir.",
            "Decoder-də masked (causal) self-attention istifadə olunur ki, model gələcək "
            "token-lərə baxa bilməsin — əks halda training zamanı cavab sızır.",
            "Self-attention-ın hesablama xərci ardıcıllıq uzunluğu ilə KVADRATİK artır "
            "(n² cüt-cüt müqayisə).",
            "Transformer blokunda hər alt-layer-dən sonra residual connection və layer "
            "normalization gəlir.",
            "Attention çəkiləri modelin nəyə baxdığını göstərir, lakin səbəb-nəticə "
            "izahı kimi etibarlı sayıla bilməz.",
        ],
        [
            "Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V",
        ],
    ),
    Section(
        "Pretraining və müasir modellər",
        [
            "BERT encoder-only-dir və masked language modelling ilə ikitərəfli kontekstdə "
            "pretrain olunur — birbaşa mətn generasiyası üçün nəzərdə tutulmayıb.",
            "GPT decoder-only-dir və növbəti token-i proqnozlaşdırmaqla öyrədilir, hər "
            "layer-də causal masking istifadə edir.",
            "In-context learning prompt-dakı nümunələrdən istifadə edir, lakin modelin "
            "ÇƏKİLƏRİNİ YENİLƏMİR.",
            "Fine-tuning çəkiləri yeniləyir; prompting isə yeniləmir.",
            "Vision Transformer şəkli patch-lərə bölür və onları token kimi emal edir; "
            "CNN-in lokallıq fərziyyəsi olmadığı üçün adətən daha çox data tələb edir.",
            "CLIP şəkil və mətni ortaq embedding fəzasında uyğunlaşdırır (contrastive "
            "öyrənmə).",
            "Knowledge distillation kiçik student modelini böyük teacher-in yumşaq "
            "ehtimal çıxışlarına uyğunlaşdırır.",
            "RAG modelin cavabını xarici mənbədən çəkilən sənədlərlə əsaslandırır.",
        ],
    ),
    Section(
        "Training gedişini oxumaq (imtahanda tez-tez çıxır)",
        [
            "Training loss azalır, validation loss artırsa — overfitting başlayıb; "
            "dönmə nöqtəsi early stopping üçün doğru andır.",
            "Hər iki loss yüksək və yaxındırsa — underfitting (model zəifdir).",
            "Validation loss dalğalanırsa — learning rate çox böyük ola bilər.",
            "Training loss ümumiyyətlə azalmırsa — learning rate, initialization və ya "
            "data hazırlığında problem var.",
            "Modelin performansı kimi HƏMİŞƏ ən yaxşı validation nöqtəsi yox, seçim "
            "qaydasına uyğun nəticə bildirilir: early stopping tətbiq edilibsə, ən yaxşı "
            "nöqtə etibarlıdır.",
            "Yekun nəticə test set üzərində, yalnız bir dəfə hesablanmalıdır.",
        ],
    ),
]


def all_sections() -> list[tuple[str, list[Section]]]:
    return [("Machine Learning", ML_SECTIONS), ("Deep Learning", DL_SECTIONS)]


def stats() -> dict[str, int]:
    points = sum(len(s.points) for _, group in all_sections() for s in group)
    formulas = sum(len(s.formulas) for _, group in all_sections() for s in group)
    sections = sum(len(group) for _, group in all_sections())
    return {"sections": sections, "points": points, "formulas": formulas}
