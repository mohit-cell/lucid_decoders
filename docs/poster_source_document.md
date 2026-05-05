# Five-Page Technical Poster Source Document
## Attention-Based Hallucination Detection in Neural Machine Translation

---

## Page 1: Problem Definition

### Project Title
**Attention-Based Hallucination Detection in Neural Machine Translation**

### Core Problem
Neural Machine Translation (NMT) systems can generate fluent translations that contain information not supported by the source sentence. This phenomenon is called **hallucination**. A hallucinated translation may introduce new entities, facts, numbers, events, or claims that are absent from the original source text.

The goal of this project is to detect hallucinations in English-to-German machine translation outputs by using internal attention behavior from a multilingual sequence-to-sequence translation model.

### Main Task
Given:

```text
Input:
  Source sentence: English sentence
  Generated translation: German machine translation

Output:
  Sentence-level hallucination probability: score in [0, 1]
  Token-level hallucination prediction: whether each generated token is hallucinated
  Head-level interpretability score: which decoder attention heads are most predictive
```

The project studies three related prediction tasks:

| Task | Input | Output | Purpose |
|---|---|---|---|
| Sentence-level hallucination detection | Source + generated translation | Probability that the translation is hallucinated | Detect whether the entire translation is unreliable |
| Token-level hallucination localization | Source + generated translation | Label for each target token | Identify where the hallucination occurs |
| Attention-head-level sentence detection | Features from each decoder layer/head | Probability that the sentence is hallucinated | Find which attention heads encode hallucination signals |

### Example

```text
Source:
  The committee approved the budget on Monday.

Generated German translation:
  Der Ausschuss genehmigte am Montag den Haushalt von 5 Millionen Euro.

Literal English meaning:
  The committee approved the budget of 5 million euros on Monday.
```

The phrase:

```text
5 Millionen Euro
```

is not grounded in the source sentence. The source says a budget was approved, but does not mention an amount.

Expected outputs:

```text
Sentence label:
  hallucinated = 1

Token-level hallucination:
  Der: 0
  Ausschuss: 0
  genehmigte: 0
  am: 0
  Montag: 0
  den: 0
  Haushalt: 0
  von: 1
  5: 1
  Millionen: 1
  Euro: 1

Sentence hallucination probability:
  0.82
```

### Why This Is Important
Machine translation errors are often evaluated using surface similarity metrics or human quality judgments. However, hallucination is especially dangerous because the output can be fluent and plausible while being factually unsupported. This matters in high-stakes settings such as legal, medical, academic, financial, or government translation.

The central hypothesis of this project is:

```text
Hallucinated translations should show weaker or more diffuse grounding behavior in cross-attention patterns between target tokens and source tokens.
```

---

## Page 2: Motivation And Research Gap

### How This Problem Is Typically Solved
Hallucination detection and translation quality estimation are usually approached through one of the following methods:

| Approach | Description | Limitation |
|---|---|---|
| Human evaluation | Human annotators identify adequacy errors or hallucinations | Expensive, slow, not scalable |
| Reference-based metrics | BLEU, chrF, COMET, BLEURT compare output to reference translation | Requires reference translations |
| Quality estimation models | Predict MT quality from source and hypothesis without reference | Often black-box and not directly interpretable |
| Token classification models | Learn token-level error labels from annotated data | Requires expensive token-level supervision |
| LLM-based judges | Ask an LLM whether a translation is faithful | Can be inconsistent, expensive, and hard to interpret |

### What Is Missing
Most methods can tell us whether a translation is bad, but they often do not explain **why** the model failed internally.

For NMT hallucination detection, the missing piece is:

```text
Can internal model attention reveal whether generated translation tokens are grounded in the source sentence?
```

This project investigates whether cross-attention and decoder self-attention statistics can act as interpretable signals for hallucination detection.

### Why Attention Is Relevant
In encoder-decoder NMT models, cross-attention connects the generated target-side token to source-side tokens. If a generated token is well grounded, we expect the decoder to attend meaningfully to source tokens. If a generated token is hallucinated, the model may rely more on decoder-side language modeling behavior and less on source grounding.

This gives a technical intuition:

```text
Grounded token:
  strong, focused cross-attention to relevant source tokens

Hallucinated token:
  weak, diffuse, or abnormal cross-attention
  stronger reliance on decoder self-attention
```

### Project Gap Statement
The project addresses the gap between **performance-based hallucination detection** and **interpretable hallucination detection**.

Instead of only asking:

```text
Can we classify hallucinations?
```

we also ask:

```text
Which attention heads are informative?
Are hallucination signals visible in cross-attention distributions?
Do token-level and sentence-level signals behave differently?
```

### Main Research Questions
1. Can mBART attention-derived features detect sentence-level hallucinations?
2. Can the same features localize hallucinated target tokens?
3. Are some decoder attention heads more predictive of hallucinations than others?
4. Is attention-based detection more useful as an interpretability signal or as a high-performance classifier?
5. How does performance change when scaling from 1k to 10k and beyond?

---

## Page 3: Main Ideas And Method Details

### Main Ideas Investigated

#### Idea 1: Use Teacher-Forced mBART Attention
Instead of generating new translations, the project feeds the source sentence and the already generated translation into mBART using teacher forcing. This allows extraction of attention matrices for the exact translation being evaluated.

Model used:

```text
facebook/mbart-large-50-many-to-many-mmt
Source language: en_XX
Target language: de_DE
```

mBART-large has:

```text
12 decoder layers
16 attention heads per layer
192 total decoder cross-attention heads
```

For each example, the model returns:

```text
Cross-attention:
  target token -> source tokens

Decoder self-attention:
  target token -> previous target tokens
```

#### Idea 2: Convert Attention Matrices Into Compact Features
Raw attention tensors are too large to train directly on. The project converts each attention distribution into interpretable summary statistics.

For each attention distribution, the project extracts:

| Feature | Meaning |
|---|---|
| Entropy | How diffuse or uncertain the attention distribution is |
| Max attention | Strength of the strongest attended token |
| Variance | Spread of attention mass |
| Top-k mass | Total attention assigned to the top-k attended tokens |

For token-level features, these statistics are pooled across layers and heads using:

```text
mean
standard deviation
minimum
maximum
last-layer mean
```

Additional ratio features compare decoder self-attention to source cross-attention:

```text
self_to_cross_max_ratio
self_to_cross_entropy_ratio
```

These ratios are important because hallucination may occur when the model relies more on target-side fluency than source-side grounding.

#### Idea 3: Train Three Classifier Types
The project trains three machine learning classifiers on the extracted features:

```text
1. Logistic regression
2. Random forest
3. Multi-layer perceptron
```

Logistic regression is included as the main interpretable baseline. Random forest tests whether non-linear decision boundaries improve performance. MLP tests whether a small neural classifier can learn stronger interactions between features.

#### Idea 4: Train Classifiers At Three Granularities
The project does not only train one classifier. It evaluates hallucination detection at three levels:

| Level | Feature rows | Label |
|---|---|---|
| Token level | One row per generated target token | token_label |
| Sentence level | One row per translation | sentence_label |
| Sentence-head level | One row per sentence per attention head | sentence_label |

The sentence-head classifier is especially important for interpretability. It trains one classifier per decoder layer/head pair. With 12 layers and 16 heads, this gives:

```text
12 x 16 = 192 head-specific classifiers
```

Each head classifier answers:

```text
If we only use features from this one attention head, how predictive is it of hallucination?
```

### Method Block Diagram

```text
Source sentence + Generated translation
              |
              v
Teacher-forced mBART
              |
              v
Cross-attention + decoder self-attention tensors
              |
              v
Attention feature extraction
              |
              +-------------------+----------------------+----------------------+
              |                   |                      |
              v                   v                      v
       Token features       Sentence features      Sentence-head features
              |                   |                      |
              v                   v                      v
       Token classifier     Sentence classifier    192 head classifiers
              |                   |                      |
              v                   v                      v
   Token hallucination     Sentence hallucination  Best predictive heads
   localization            probability             and interpretability
```

### Training Pipeline
The implemented pipeline contains these stages:

```text
prepare
extract-chunked
train-token
train-sentence
train-heads
```

The extraction stage is chunked and resumable. This was necessary because mBART attention extraction is GPU-intensive and long-running.

For 20k examples with chunk size 250:

```text
20,000 examples / 250 examples per chunk = 80 chunks
```

Each chunk writes:

```text
chunk_XXXXX.token.parquet
chunk_XXXXX.sentence.parquet
chunk_XXXXX.sentence_head.parquet
chunk_XXXXX.report.json
```

If extraction stops, completed chunks can be skipped when rerunning the same command.

---

## Page 4: Evaluation Setup And Experimental Results

### Dataset
The project uses WMT22 and WMT23 English-to-German quality estimation and hallucination-related data.

Processed sources include:

| Source | Use |
|---|---|
| WMT22 sentence-level MQM | Sentence-level quality/hallucination-style supervision |
| WMT22 word-level MQM | Token-level BAD/OK supervision |
| WMT23 Task 2 span annotations | Hallucination/error span supervision |
| WMT23 hallucination gold labels | Exported for inspection but excluded from trainable attention set if source is missing |

Important preprocessing rule:

```text
Rows without source text are excluded from attention training because cross-attention requires a source sentence.
```

Current normalized data sizes observed in the project:

| File | Rows |
|---|---:|
| WMT22 sentence MQM | 29,811 |
| WMT22 word MQM | 60,860 |
| WMT23 Task 2 | 35,717 |
| WMT23 hallucination gold | 1,897 |
| Total normalized files | 128,285 |

### 10k Scaling Experiment
A 10k balanced subset was extracted and evaluated end-to-end.

Subset composition:

| Split | Positive | Negative |
|---|---:|---:|
| Train | 4,037 | 4,037 |
| Validation | 758 | 758 |
| Test | 205 | 205 |

Additional preprocessing facts:

```text
Positive examples required token/span supervision.
Duplicate sampled example_ids rewritten: 474
Positive rows skipped for missing token supervision: 9,895
```

Feature extraction used:

```text
Model: facebook/mbart-large-50-many-to-many-mmt
GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB VRAM
Chunks: 40 / 40
Chunk size: 250
Processed examples: 10,000
Skipped examples: 0
Token rows: 450,461
Sentence rows: 10,000
Sentence-head rows: 1,920,000
Approximate extraction runtime: 5h16m
```

The sentence-head row count follows directly from:

```text
10,000 sentences x 192 heads = 1,920,000 rows
```

### Evaluation Metrics
The classifiers are evaluated with:

| Metric | Meaning |
|---|---|
| Precision | Of predicted hallucinations, how many are correct |
| Recall | Of true hallucinations, how many are found |
| F1 | Harmonic mean of precision and recall |
| ROC-AUC | Ranking quality over all thresholds |
| Confusion matrix | TN / FP / FN / TP error breakdown |

### Sentence-Level Results On 10k
Sentence-level hallucination detection was the strongest task.

| Model | Validation P/R/F1/AUC | Test P/R/F1/AUC |
|---|---|---|
| Logistic regression | .570 / .863 / .686 / .674 | .584 / .800 / .675 / .658 |
| Random forest | .542 / .914 / .681 / .632 | .548 / .863 / .670 / .648 |
| MLP | .546 / .704 / .615 / .589 | .504 / .624 / .558 / .544 |

Key takeaway:

```text
Logistic regression performed best overall for sentence-level detection.
```

It had the strongest test F1/AUC balance and is also the simplest and most interpretable model.

### Token-Level Results On 10k
Token localization remained difficult.

| Model | Validation P/R/F1/AUC | Test P/R/F1/AUC |
|---|---|---|
| Logistic regression | .076 / .484 / .131 / .565 | .080 / .434 / .135 / .555 |
| Random forest | .070 / .676 / .128 / .552 | .075 / .669 / .135 / .553 |
| MLP | .075 / .547 / .131 / .569 | .078 / .543 / .137 / .559 |

Key takeaway:

```text
Token-level precision is very low across all models.
```

This suggests that attention summary features alone are not sufficient for accurate token-level hallucination localization.

### Head-Level Results On 10k
The sentence-head classifier trained all 192 head classifiers.

| Model | Best Head | Validation F1/AUC | Test F1/AUC |
|---|---|---|---|
| Logistic regression | Layer 6, Head 13 | .702 / .713 | .711 / .727 |
| Random forest | Layer 9, Head 13 | .697 / .686 | .669 / .652 |
| MLP | Layer 6, Head 13 | .673 / .659 | .663 / .627 |

Top logistic-regression heads:

```text
L6H13
L9H13
L7H3
L6H9
L4H9
```

Key takeaway:

```text
Some individual attention heads are more predictive than the aggregate sentence classifier.
```

The best head-level logistic regression model achieved:

```text
Test F1: 0.711
Test AUC: 0.727
```

This is the strongest result observed so far.

---

## Page 5: Analysis, Conclusions, And Future Work

### Main Findings

#### Finding 1: Sentence-Level Detection Works Better Than Token Localization
The project found that attention-derived features are more useful for deciding whether a full translation is hallucinated than for identifying the exact hallucinated token.

Reason:

```text
Sentence-level labels aggregate weak signals across many tokens.
Token-level labels require precise alignment between character spans, subword tokens, and attention behavior.
```

The token-level task is also heavily imbalanced. Most target tokens are not hallucinated, so the classifiers produce many false positives when attempting to increase recall.

#### Finding 2: Logistic Regression Is Surprisingly Strong
Logistic regression performed best or near-best across the most important tasks.

This suggests that hallucination signals in attention features may be approximately linearly separable at sentence/head level. This is useful because logistic regression is:

```text
interpretable
fast to train
stable on larger datasets
less prone to massive artifacts than random forest
```

Random forest was competitive for sentence-level detection but created very large artifacts for head classifiers. In the 10k experiment, the random-forest head model pickle reached approximately 10.7 GB. This makes full-scale RF head training impractical without sharded or optional model saving.

#### Finding 3: Head-Level Analysis Is The Most Promising Result
The best sentence-head classifier outperformed the aggregate sentence classifier.

This supports the interpretability hypothesis:

```text
Hallucination-related information is not uniformly distributed across all attention heads.
Some decoder heads encode stronger source-grounding or hallucination-sensitive behavior.
```

The repeated appearance of heads such as L6H13 and L9H13 across model rankings suggests that mid-to-late decoder layers may contain useful faithfulness signals.

#### Finding 4: Cross-Attention Alone Is Not A Complete Explanation
Although attention helps, it does not fully solve hallucination detection. Token-level results show that low-level localization remains noisy.

Possible reasons:

```text
Attention is not guaranteed to be a faithful explanation.
Subword tokenization complicates word-level labels.
Hallucination spans may not align cleanly with mBART tokens.
Some hallucinations are semantic and not visible from attention concentration alone.
The model used for attention extraction is not necessarily the model that generated the translation.
```

### Where The System Works Well
The system works best when hallucination affects the global adequacy of the translation. Examples include:

```text
extra named entities
unsupported numbers
inserted facts
major semantic additions
translation segments with weak source grounding
```

In these cases, sentence-level and head-level attention statistics can capture abnormal grounding behavior.

### Where The System Fails
The system struggles when hallucinations are:

```text
small function-word changes
subtle semantic shifts
morphological errors
paraphrases that do not strongly alter attention
token spans that do not align with subword tokens
```

It also struggles when token-level supervision is sparse or noisy. The 10k experiment showed that many positive sentence-level examples lacked token/span supervision, which forced a tradeoff between using more data and preserving token-level labels.

### Practical Engineering Lessons
The project also produced several engineering lessons:

```text
mBART attention extraction is the computational bottleneck.
Chunked extraction is necessary for Colab/Kaggle reliability.
Sentence-head features are large: 192 rows per sentence.
Full random-forest head training is storage-heavy and should be avoided without sharding.
Free GPU platforms are unreliable for full extraction, so resumability is essential.
```

Observed runtime:

```text
10k examples on RTX 4070 Laptop 8GB: ~5h16m
20k examples estimated: ~10-12h on similar hardware
Full ~126k trainable examples estimated: multi-day extraction
```

### Conclusions
This project shows that attention-derived features from mBART contain measurable information about hallucination in NMT outputs.

The strongest conclusion is:

```text
Attention-head-level sentence classifiers are the most promising direction.
```

The best 10k result came from a single-head logistic regression classifier:

```text
Layer 6, Head 13
Test F1: 0.711
Test AUC: 0.727
```

This suggests that specific decoder attention heads may act as useful indicators of whether a translation is grounded in the source.

However, token-level hallucination localization remains unresolved. The low token-level precision shows that attention summaries alone are insufficient for fine-grained localization.

### Future Work
The next improvements should focus on four directions.

#### 1. Better Token-Level Supervision
Improve token alignment between WMT labels and mBART subword tokens. Use better span-to-token mapping and possibly evaluate at word level rather than subword level.

#### 2. Calibration And Threshold Analysis
Sentence-level models currently have high recall but many false positives. Calibration curves, precision-recall curves, and threshold selection should be added to tune the classifier for specific use cases.

#### 3. Sharded Head Model Saving
Random-forest head models are too large when saved as one pickle. Future implementation should save one model per head or make model saving optional.

#### 4. Compare Against Text-Based Baselines
Add baselines that use source/hypothesis embeddings or sentence-pair encoders. This would clarify whether attention features add value beyond standard text representations.

### Final Poster Takeaway
The project demonstrates a practical and interpretable hallucination detection pipeline:

```text
mBART attention -> compact attention statistics -> sentence/token/head classifiers
```

The major scientific takeaway is:

```text
Hallucination detection is more successful at sentence and attention-head level than at token level.
```

The major interpretability takeaway is:

```text
Specific mBART decoder heads appear to encode stronger hallucination-sensitive grounding signals than the average attention pattern.
```

The major engineering takeaway is:

```text
Scalable hallucination analysis requires chunked, resumable feature extraction because attention tensors are expensive to compute and store.
```
