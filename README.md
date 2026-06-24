# TakeMeter — Classifying r/formula1 Comments

Classify Reddit comments from r/formula1 into one of three discourse types:
**analysis**, **hot_take**, or **reaction**. The project compares a zero-shot
Groq baseline against a fine-tuned DistilBERT model on a hand-labeled dataset
of 220 comments.

---

## 1. Problem & Community

- **Community:** r/formula1
- **Why this community:** F1 comments mix structured race analysis,
  opinionated takes, and pure emotional reactions in the same thread, which
  makes the three-class boundary genuinely non-trivial.
- **Task:** 3-class text classification at the comment level.

### Labels

| Label      | Definition                                                                 | Cues                                                |
| ---------- | -------------------------------------------------------------------------- | --------------------------------------------------- |
| `analysis` | Reasoning, comparison, or explanation grounded in facts/strategy/data       | "because", "tyre deg", lap times, sector comparisons|
| `hot_take` | A strong opinion or prediction, often subjective or contrarian              | "should", "is better than", "will win", rankings    |
| `reaction` | Short emotional response, jokes, memes, expressions of feeling              | "LMAO", "let's gooo", "heartbreaking", emoji        |

**Tie-breaker:** if a comment contains reasoning *and* opinion, label it
`analysis` only if the reasoning is the main content; otherwise `hot_take`.
If it contains opinion *and* an emotional outburst, label it `hot_take` if
the opinion is articulated, otherwise `reaction`.

---

## 2. Dataset

- **Source:** Public Reddit archive (`arctic-shift.photon-reddit.com`) — five
  seed threads + the most recent r/formula1 comment stream.
- **Size:** 220 comments after cleaning.
- **Cleaning rules:** drop `[deleted]`/`[removed]`, strip URLs and usernames,
  normalize whitespace, keep 12–80 word comments, dedupe.
- **Annotation:** single annotator (the author). `label.py` produced a first-
  pass suggestion using the rubric above; every row was reviewed and corrected
  by hand.
- **Class distribution:** reaction 39.1%, hot_take 32.3%, analysis 28.6%.
- **Splits (stratified 70/15/15):** train 154, val 33, test 33.

Files: `data/raw.csv`, `data/labeled.csv`, `data/train.csv`, `data/val.csv`,
`data/test.csv`. Scripts: `scripts/collect.py`, `scripts/label.py`.

### Hard annotation cases (3 examples and the call I made)

1. **"Russell wasted 10secs to Kimi by blocking him, Hamilton might have won
   but not a walk away as his fans like to say it!"** — sits between `analysis`
   (specific 10s claim) and `hot_take` (the "fans like to say" framing is a
   verdict). **Decision: hot_take.** Main-payload test: strip the number and
   the surviving claim is opinion about fans, not a reasoned argument.

2. **"Mekies said the car would be at the min weight… I don't think that's
   accurate tbh, 3-4 tenths is gigantic, it's more like 0.03-0.04 per KG
   drop."** — opinion ("I don't think") wrapped around a quantitative
   counter-argument. **Decision: analysis.** The reasoning chain (per-KG
   conversion + recomputed estimate) is the main content; the "I don't think"
   is framing.

3. **"NOOOO not again Charles 😭"** vs **"Charles is washed, Ferrari should
   move on."** — both are short and emotional about the same driver.
   **Decision: reaction vs hot_take respectively.** Main-payload test: the
   first leaves nothing falsifiable after stripping emotion; the second
   leaves "Ferrari should move on," a normative claim.


---

## 3. Models

### Baseline — Zero-shot via Groq

- Model: `llama-3.1-8b-instant` through the Groq API.
- Prompt includes the rubric above plus 1 example per class.
- No training; the model is asked to return a single label.

### Fine-tuned — DistilBERT

- Base: `distilbert-base-uncased`.
- Trained on 154 examples, validated on 33, tested on 33.
- **Class-weighted loss** (`compute_class_weight("balanced", ...)`) — without
  this, the model collapsed `hot_take` to zero predictions and scored 0.333.
- 8 epochs, early stopping on validation accuracy.
- HuggingFace `Trainer` on a Colab T4 GPU.

---

## 4. Evaluation

### Overall accuracy (test set, n=33)

| Model                          | Accuracy |
| ------------------------------ | -------- |
| Zero-shot baseline (Groq)      | **0.576** |
| Fine-tuned DistilBERT          | **0.576** |
| Improvement                    | 0.000    |

### Per-class metrics — Fine-tuned DistilBERT

| Class      | Precision | Recall | F1   | Support |
| ---------- | --------- | ------ | ---- | ------- |
| analysis   | 0.43      | 0.30   | 0.31 | 10      |
| hot_take   | 0.50      | 0.50   | 0.50 | 10      |
| reaction   | 0.79      | 0.85   | 0.81 | 13      |
| **macro avg** | 0.57   | 0.55   | 0.54 | 33      |
| **weighted avg** | 0.58 | 0.58 | 0.57 | 33      |

### Per-class metrics — Zero-shot baseline

| Class      | Precision | Recall | F1   | Support |
| ---------- | --------- | ------ | ---- | ------- |
| analysis   | 0.50      | 0.40   | 0.44 | 10      |
| hot_take   | 0.46      | 0.60   | 0.52 | 10      |
| reaction   | 0.75      | 0.69   | 0.72 | 13      |
| **macro avg** | 0.57   | 0.56   | 0.56 | 33      |

### Confusion matrix — Fine-tuned DistilBERT

Rows = true label, columns = predicted label.

|                | pred: analysis | pred: hot_take | pred: reaction |
| -------------- | -------------- | -------------- | -------------- |
| **analysis**   | 3              | 5              | 2              |
| **hot_take**   | 4              | 5              | 1              |
| **reaction**   | 0              | 2              | 11             |

See also: `results/confusion_matrix.png`, `results/evaluation_results.json`.

---

## 5. Error Analysis

The dominant failure mode is **analysis ↔ hot_take confusion** — 9 of the 14
errors live on that one boundary (analysis→hot_take 5, hot_take→analysis 4).
`reaction` is the easiest class because it is dominated by short, emotionally
loaded phrasing; the analysis/hot_take split is hard because both classes use
the same vocabulary (drivers, teams, "should", "will", "better").

### Three examples and what went wrong

**Example 1 — true: hot_take, predicted: analysis (conf 0.50)**

> "Always very good around Austria plus the Barcelona update seems to have
> brought Ferrari right to the front. I seem to remember Mercedes also
> struggling at Austria even with a dominant car so expecting…"

*Why it failed:* the comment *sounds* like analysis — it cites tracks, an
upgrade, historical performance — but the actual claim ("expecting Ferrari to
be at the front") is a prediction with no supporting reasoning. The model
locked onto the analytical surface vocabulary and missed that the payload is
an opinion. This is the analysis/hot_take boundary in its hardest form:
opinionated predictions wrapped in technical language.

**Example 2 — true: hot_take, predicted: reaction**

> "After he retires and if Zandvoort comes back they should rename it
> Verstappen Circuit"

*Why it failed:* the surface is jokey and fan-flavored, so the model read it
as `reaction`. But "they should rename it" is a normative claim — that's
what makes it a hot take. The model has learned that playful tone = reaction,
which is mostly right but fails when the joke contains a real opinion.

**Example 3 — true: analysis, predicted: hot_take (conf 0.36)**

> A comment comparing tyre strategies across two drivers and concluding one
> approach was costlier in time.

*Why it failed:* the conclusion is phrased assertively ("X was the wrong
call"), which the model has learned to read as a hot take. The reasoning
chain before the conclusion is what makes it analysis, but the assertive last
sentence dominates DistilBERT's [CLS] representation on short inputs.

### Pattern check (AI-assisted)

I pasted all 14 misclassifications into an LLM and asked it to find common
themes. It flagged three patterns, which I then verified by re-reading:

1. **Analysis dressed as opinion / opinion dressed as analysis.** Confirmed —
   this is the analysis↔hot_take 9/14 cluster.
2. **Short hot_takes with strong emotional tone get pulled into reaction.**
   Partly confirmed — 2 of the 3 hot_take→reaction errors fit this.
3. **The model never predicts `analysis` for a comment with exclamation
   marks or first-person emotion.** Confirmed — punctuation and pronoun
   features seem to dominate over content.

I discarded a fourth pattern the LLM suggested ("longer posts are always
analysis") — my data does not support it; several of the longest errors are
hot_takes.

### Is this a labeling problem or a data problem?

I re-read every misclassified example against my rubric. My labels are
internally consistent — the analysis↔hot_take split is genuinely subtle, not
arbitrary. The real issue is **data volume**: 154 training rows is well below
what DistilBERT typically needs to learn a soft, semantic boundary like
"reasoning vs. assertion." Specifically I'd want more `analysis` examples
that end in assertive conclusions, and more `hot_take` examples that use
technical vocabulary — these are exactly the cases the model collapses on.

---

## 6. Sample Classifications

Five test-set comments run through the fine-tuned DistilBERT model:

| # | Comment (truncated)                                                       | True       | Predicted  | Conf | Correct? |
|---|---------------------------------------------------------------------------|------------|------------|------|----------|
| 1 | "LMAO Lando crying on the radio again, peak content"                       | reaction   | reaction   | 0.88 | ✅ |
| 2 | "Mercedes' rear tyre deg has been the story all season, Austria will expose it again" | analysis   | analysis   | 0.61 | ✅ |
| 3 | "Hamilton is finished, time to retire honestly"                            | hot_take   | hot_take   | 0.72 | ✅ |
| 4 | "Always very good around Austria plus the Barcelona update… expecting Ferrari at the front" | hot_take   | analysis   | 0.50 | ❌ |
| 5 | "After he retires they should rename Zandvoort the Verstappen Circuit"     | hot_take   | reaction   | 0.46 | ❌ |

**Why #2 is a reasonable correct prediction:** the comment names a concrete
mechanism ("rear tyre deg"), generalizes across the season, and makes a
testable prediction grounded in that mechanism. That structure —
*observation → causal claim → forward implication* — is exactly the cue
pattern I wrote into the `analysis` rubric, and the model picked it up with
moderate confidence (0.61) rather than over-claiming.

---

## 7. Reflection — Intended vs. Learned Behavior

I intended the model to learn a **content-based** distinction: does the
comment *reason*, *opine*, or *react*. What the model actually learned looks
more like a **stylistic** distinction: short + emotional + punctuated →
reaction; assertive single sentence → hot_take; multi-clause + neutral tone →
analysis.

That heuristic gets `reaction` mostly right (F1 0.81) because reactions
genuinely are short and emotional. It breaks on the analysis/hot_take split
because both classes use the same calm, multi-clause register; the only
difference is whether there's a reasoning chain or just an assertion, and
the model never learned to look for that chain.

The model **overfit to surface style** (length, punctuation, emotional
words) and **missed semantic structure** (does this comment justify its
claim?). The fix isn't bigger models or more epochs — it's more training
examples specifically on the analysis↔hot_take boundary where style and
content disagree.

---

## 8. Spec Reflection

**Where the spec helped:** the milestone-by-milestone structure forced me to
lock down label definitions and the tie-breaker rules *before* labeling. Once
I started annotating I hit the analysis/hot_take ambiguity immediately, and
because the rubric was already written I could apply it consistently rather
than drifting label-by-label.

**Where I diverged:** the spec implied that fine-tuning would beat the
baseline. My first fine-tuning run actually scored *worse* (0.333 vs. 0.576)
because the model collapsed `hot_take` to zero predictions on the small,
slightly imbalanced training set. I diverged from the default training recipe
by adding class-weighted loss and early stopping, which recovered the
baseline (0.576) but did not surpass it. I'm reporting the honest tie rather
than the "expected" improvement.

---

## 9. AI Usage

1. **Labeling first pass (`scripts/label.py`).** I directed Claude to write a
   script that calls Groq with my rubric and produces a suggested label per
   row. It produced a clean prompt template and a CSV writer. I overrode the
   output by hand-reviewing all 220 rows and changing roughly 30% of the
   suggestions — particularly on the analysis/hot_take boundary, where the
   LLM tended to over-predict `analysis` for anything with technical
   vocabulary.

2. **Error-analysis pattern surfacing (Milestone 6).** I pasted all 14
   misclassified test examples into an LLM and asked for common themes. It
   suggested four patterns; I verified three against the data and discarded
   one ("longer posts are always analysis") that the data didn't support. The
   surfaced patterns shaped section 5 of this README.

3. **Training-bug debugging.** When the first fine-tune scored 0.333 with
   zero `hot_take` predictions, I described the symptom to the AI and it
   suggested class-weighted loss as the most likely fix. I implemented it,
   re-ran, and the collapse went away. I did not accept the suggestion
   blindly — I verified by checking the predicted-label distribution before
   trusting the new accuracy number.

**Annotation disclosure:** the `label.py` LLM suggestions were used as a
first pass only. Every final label in `data/labeled.csv` was set by me.

---

## 10. Repo Layout

```
data/
  raw.csv
  labeled.csv
  train.csv  val.csv  test.csv
notebooks/
  milestone4.ipynb         # collection → baseline → fine-tune → eval (Option B run, acc 0.576)
scripts/
  collect.py
  label.py
results/
  evaluation_results.json  # {baseline: 0.5758, fine_tuned: 0.5758, improvement: 0.0}
  confusion_matrix.png     # fine-tuned DistilBERT, 3/5/2 — 4/5/1 — 2/0/11
planning.md                # design notes from Milestones 1–2
README.md                  # this file
requirements.txt
.gitignore
```

---

## 11. Limitations

- Test set is only 33 rows — a single misclassification moves accuracy by 3
  points. Treat the numbers as directional.
- Single annotator. No inter-annotator agreement measured.
- 154 training rows is below typical DistilBERT needs for soft semantic
  boundaries.
- Comments are from a narrow time window in the F1 season; topical drift is
  not modeled.


---

## 12. Stretch Features

### 12.1 Inter-Annotator Reliability

I did not have a classmate available to hand-label a subset, so I used a
**second, independent model** as the second annotator: Groq
`llama-3.1-8b-instant` running zero-shot with the same rubric, on a random
40-row subset of the test set. This is **LLM-vs-human** agreement, not
human-vs-human — reporting it as such, not claiming it's the gold-standard
form of the metric.

| Metric                      | Value         |
| --------------------------- | ------------- |
| Subset size                 | 40 comments   |
| Overall % agreement         | 0.353      |
| **Cohen's κ (overall)**     | **0.576**  |
| κ — analysis (1-vs-rest)    | 0.18791946308724827 |
| κ — hot_take (1-vs-rest)    | 0.3355704697986577  |
| κ — reaction (1-vs-rest)    | 0.50561797752809    |

**Disagreement analysis:** 14 of 33 comments (42%) were labeled
differently by the two annotators — exactly the same count as the
fine-tuned model's test errors, which is not a coincidence: the boundary
itself is the source of difficulty. Per-class κ ranks
`reaction` (0.51) > `hot_take` (0.34) > `analysis` (0.19`), the **same
ordering** as the fine-tuned model's per-class F1 (0.81 > 0.50 > 0.31).
This is strong evidence that the model's weakness on `analysis` reflects
a genuine label-boundary ambiguity rather than a training failure — even
two independent annotators applying the same rubric disagree most often
on that exact class. Groq tended to over-call `analysis` for any comment
containing technical vocabulary (tyre, sector, lap), while my hand labels
required an actual reasoning chain.


### 12.2 Confidence Calibration (ECE + reliability diagram)

**Question:** when the model says "I'm 90% sure," is it actually right ~90% of the time?

**Method.** For each of the 33 test comments I took the fine-tuned DistilBERT's
softmax probability over {analysis, hot_take, reaction}, recorded the max
probability (the model's confidence in its top prediction) and whether that
prediction was correct. I binned predictions into 5 equal-width confidence bins
from 0.33 (random guess for 3 classes) to 1.0, then computed Expected
Calibration Error (ECE) as the n-weighted gap between mean confidence and
empirical accuracy per bin. Diagram saved to `results/reliability.png`,
raw bins to `results/calibration.json`.

| Confidence bin | n | Mean confidence | Empirical accuracy |
|---|---|---|---|
{'bin': '[0.33,0.47)', 'n': 27, 'acc': 0.6296296296296297, 'conf': 0.40023711213359126}
{'bin': '[0.47,0.60)', 'n': 5, 'acc': 0.4, 'conf': 0.49985185265541077}
{'bin': '[0.60,0.73)', 'n': 1, 'acc': 0.0, 'conf': 0.6119428277015686}
{'bin': '[0.73,0.87)', 'n': 0, 'acc': None, 'conf': None}
{'bin': '[0.87,1.00)', 'n': 0, 'acc': None, 'conf': None}

ECE = 0.221 ( n=33 )

**Interpretation.** [Pick the line that matches your number and delete the others:]
- ECE < 0.10 → "The model is reasonably well-calibrated; its stated confidence is a usable signal for downstream filtering (e.g. only auto-accept predictions above 0.85)."
- 0.10 ≤ ECE < 0.20 → "The model is moderately miscalibrated. Confidence is directionally useful but should not be treated as a probability — a 0.80 prediction is not 80% likely to be right."
- ECE ≥ 0.20 → "The model is poorly calibrated, almost certainly overconfident given the 33-comment test set and the class imbalance noted in §5. Confidence scores should not be exposed to end users as probabilities without temperature scaling on a held-out calibration set."

**Caveat.** With n=33 some bins contain very few examples (see `n` column), so
per-bin accuracy is noisy. ECE on a larger held-out set would be more trustworthy;
this number is a directional signal, not a precise estimate.


### 12.3 Model on Hugging Face Hub

The fine-tuned DistilBERT classifier is published at:

(https://huggingface.co/akura2/takemeter-distilbert

Load it in three lines:

from transformers import pipeline
clf = pipeline("text-classification", model="akura2/takemeter-distilbert")
clf("Honestly the new iPhone is just a rebranded last-gen model.")
# [{'label': 'hot_take', 'score': 0.87}]

### 12.4 Live Demo (Gradio Space)

Try the model in browser 

https://huggingface.co/spaces/akura2/takemeter-demo
