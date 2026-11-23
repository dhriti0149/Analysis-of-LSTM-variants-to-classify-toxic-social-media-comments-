# Analysis-of-LSTM-variants-to-classify-toxic-social-media-comments-

Here, I have applied Long Short Term Memory variants to perform classification of comments on social media based on their toxicities at a tabular csv dataset.


#  Multi-Label Toxic Comment Classification
### A Statistical Stability & Production Readiness Benchmark Across LSTM Architectures



*Beyond single-split accuracy: diagnosing model collapse, proving cross-fold reliability via hypothesis testing, and benchmarking sub-millisecond production latency.*


</div>

---

##  Overview

In real-world content moderation, **single-split validation accuracy is a trap**. Because toxic comments represent less than **10%** of online traffic—and severe categories like `threat` appear in under **0.3%**—a model can look like a winner on one lucky split while collapsing unpredictably in production.

This project benchmarks **three pure unidirectional LSTM architectures** on **159,571 real-world comments** from the [Jigsaw Toxic Comment Classification Challenge](https://kaggle.com) across six multi-label classes (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).

> ** Core Engineering Takeaway:**  
> On an initial 90/10 split, **`3_Regularized_Norm_LSTM`** ranked #1 (`0.9742` ROC-AUC). However, **K-Fold Cross-Validation** and **Variance F-Tests** exposed a catastrophic fold collapse (`27.21%` Coefficient of Variation, `p < 0.001`), disqualifying it and proving **`2_Stacked_Deep_LSTM`** and **`1_Baseline_LSTM`** as the true production-ready models.

---

##  Pipeline Architecture

Every experiment enforces strict anti-leakage hygiene (`TextVectorization` adapted inside each training split), `tf.data` prefetching, and gradient clipping (`clipnorm=1.0`).

```text
Raw Comment Text (159,571 rows)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Custom Regex Standardization (IPs, Newlines, Punct.)    │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Leak-Free TextVectorization (30k Vocab, 200 Seq Len)    │
└──────────────────────────────────────────────────────────┘
       │
       ├──► [1] Baseline LSTM       ──► Embedding(128) ─► LSTM(128) ─► Dense(64) ─► Sigmoid(6)
       ├──► [2] Stacked Deep LSTM   ──► Embedding(128) ─► LSTM(128) ─► LSTM(64)  ─► Sigmoid(6)
       └──► [3] Regularized LSTM    ──► Embedding(128) ─► SpatialDropout ─► LSTM(128) ─► LayerNorm ─► Sigmoid(6)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  4-Stage Evaluation Suite:                               │
│  1. Single-Split Volatility & Overfit Gap                │
│  2. K-Fold Cross-Validation Stability                    │
│  3. Welch's t-test & Variance Ratio F-Test               │
│  4. Production Stress Test (PR-AUC, F1, Calibration, ms) │
└──────────────────────────────────────────────────────────┘
```

---

##  Benchmark Results

### 1️ Single-Split Illusion vs. K-Fold Reality

| Model Variant | Single-Split AUC | K-Fold Mean AUC | 95% Confidence Interval | AUC Volatility (CV) | K-Fold Mean Loss | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1_Baseline_LSTM`** | 0.9565 | 0.9622 | `[0.9610, 0.9634]` | **0.052%** | **0.0504** | ✅ Rock-Solid |
| **`2_Stacked_Deep_LSTM`** | 0.9557 | **0.9645** | `[0.9553, 0.9737]` | 0.386% | 0.0511 | ✅ Reliable |
| **`3_Regularized_Norm_LSTM`** | **0.9597** | 0.8217 | `[0.2662, 1.0000]` | 27.212% | 0.0810 | ❌ Fold Collapse |

---

### 2️ Statistical Hypothesis Testing (`SciPy`)

To verify whether performance and stability differences were statistically real or random noise across splits:

| Comparison Pair | Mean AUC Diff | Welch's t-test (`p-value`) | Accuracy Verdict | Variance F-Test (`p-value`) | Stability Verdict |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **Stacked Deep vs. Baseline** | `+0.0023` | `0.3968` | Statistically Tied | **`0.0357`** | Baseline Significantly More Stable |
| **Baseline vs. Regularized** | `+0.1405` | `0.3901` | High Variance Noise | **`0.0000`** | Regularized Statistically Unstable |
| **Stacked Deep vs. Regularized** | `+0.1428` | `0.3839` | High Variance Noise | **`0.0006`** | Regularized Statistically Unstable |

---

### 3️ Per-Class ROC-AUC Breakdown

All architectures excel on explicit toxicity (`>0.978`), while rare context-dependent categories (`threat` and `identity_hate`) separate model generalization:

| Model Variant | Macro AUC | `toxic` | `severe_toxic` | `obscene` | `threat` | `insult` | `identity_hate` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1_Baseline_LSTM`** | 0.9701 | 0.9676 | 0.9862 | 0.9853 | **0.9547** | 0.9780 | 0.9487 |
| **`2_Stacked_Deep_LSTM`** | 0.9708 | 0.9697 | **0.9870** | **0.9877** | 0.9506 | **0.9814** | 0.9480 |
| **`3_Regularized_Norm_LSTM`** | **0.9742** | **0.9767** | 0.9857 | 0.9903 | 0.9519 | 0.9829 | **0.9573** |

---

### 4️ Real-World Production Stress Test

Evaluated on imbalanced validation data using **Macro PR-AUC** (stricter than ROC-AUC on skewed data), **Precision-Recall Threshold Tuning**, **Brier Calibration**, and **Single-Message Latency**:

| Model Variant | Macro PR-AUC | Optimal Macro F1 | Brier Calibration | False Alarm Rate | Toxic Catch Rate | Throughput / Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1_Baseline_LSTM`** | 0.5061 | 0.5133 | 0.01486 | **1.33%** | 72.19% | **0.26 ms** (~3,830 msg/s) |
| **`2_Stacked_Deep_LSTM`** | **0.5107** | **0.5167** | **0.01430** | 1.61% | **74.73%** | 0.31 ms (~3,270 msg/s) |
| **`3_Regularized_Norm_LSTM`** | 0.5092 | 0.5158 | 0.01322 | 1.57% | 78.67% | 0.20 ms (Disqualified) |

---

## Final Production Verdict

| Deployment Goal | Recommended Model | Why It Wins |
| :--- | :--- | :--- |
| ** Best Overall Production Classifier** | **`2_Stacked_Deep_LSTM`** | Highest **Macro PR-AUC (`0.5107`)**, highest **Optimal F1 (`0.5167`)**, catches **74.73%** of abuse at **3270+ msgs/sec** & stays stable across K-Folds (`0.386%` CV). |
| ** Strictest Stability & Low False Bans** | **`1_Baseline_LSTM`** | Statistically tied in ROC-AUC (`p = 0.3968`), cuts innocent false alarms down to **1.33%**, and delivers near-zero split variance (`0.052%` CV). |
| ** Rejected for Production** | **`3_Regularized_Norm_LSTM`** | Heavy dropout + `LayerNormalization` at `lr=1e-3` caused gradient collapse on unseen splits (`F-test p < 0.001`). |

---

## Repository Visuals

After running the benchmark pipeline, the following diagnostic dashboards are generated automatically:
* `lstm_stability_comparison.png` — 4-panel comparison of K-Fold error bars, epoch volatility, and split-to-split variance.
* `lstm_roc_auc_comparison.png` — Overlaid multi-label Macro ROC-AUC curves across all three variants.

---


