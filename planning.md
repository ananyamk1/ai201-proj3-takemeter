At minimum, your document must substantively address these six questions:

Community: What community did you choose and why? Why is this community a good fit for a classification task — what makes the discourse varied enough to be interesting?
Labels: What are your 2–4 labels? Define each in a complete sentence. Include 2 example posts per label.
Hard edge cases: What type of post will be genuinely ambiguous between two labels? How will you handle it when you encounter it during annotation?
Data collection plan: Where will you collect examples? How many per label? What will you do if a label is underrepresented after 200 examples?
Evaluation metrics: Which metrics will you use to evaluate your model and why are those the right ones for this specific task? (Accuracy alone is not enough — explain what else you need and why.)
Definition of success: What performance would make this classifier genuinely useful? What would you accept as "good enough" for deployment in a real community tool?




# Project 3 Planning: TakeMeter

> Write this document before you collect any data or write any training code.
> Your label rubric and evaluation plan are what you'll use to direct AI tools
> (Claude, Copilot, etc.) to generate the Colab cells and the deployed app —
> the more specific they are, the more useful the generated code will be.
> Update the Labeling Strategy and Modeling Approach sections if you change
> approach during implementation. Update this file before starting stretch features.

---

## Community

<!-- Which online community did you pick? Why is "what makes a good take here" specific
     to this community and hard to define from outside it? -->

Community: r/formula1.

F1 is an opinion factory. Every race weekend the subreddit produces thousands of
top-level comments that split, in practice, into three native modes: long
debrief posts that cite sector times and FIA regulations, bold verdicts about
drivers and teams stated without evidence, and in-the-moment venting during
live race threads. What counts as a "good take" here is genuinely community-
specific — a comment like "Norris was 0.3s up in S2 every lap" reads as
substantive analysis to F1 fans and as noise to anyone else, and a comment like
"NOOOO Charles why" is a recognizable native form of participation, not just
noise. I picked F1 over r/nba (where the hot-take/reaction boundary collapses
because almost every reaction is a take) because race-thread reactions are
tightly tied to a live on-track event, which keeps the three classes
separable.

---

## Labels

<!-- What are your 3 classes? Give a one-sentence definition and 2 examples each.
     Then name the hardest edge case — a type of post that genuinely sits between
     two labels — and explain how you'll handle it. -->

Three mutually exclusive classes. Every comment gets exactly one label.

**`analysis`** — a structured argument backed by verifiable evidence (lap times,
sector deltas, tyre compounds, regulation text, team radio, historical
comparisons).
- "Norris was 0.3s up in S2 every lap until Mercedes pitted Russell for the
  undercut on lap 34 — McLaren left the response stop one lap too late."
- "Per Article 28.2 of the 2024 Sporting Regs, mechanics can't touch the car
  while a 5s penalty is being served. The Aston stop was legal because the
  timer expired before the gun touched the wheel."

**`hot_take`** — a bold verdict about a driver, team, or decision, stated
without evidence, or with evidence so thin it functions as decoration.
- "Hamilton to Ferrari is going to be the biggest flop in F1 history."
- "Stroll is washed and only has the seat because of his dad."

**`reaction`** — an in-the-moment emotional response to a specific event.
Short, often caps/emoji/repeated letters, addressed *to* a driver or *at* an
event rather than *about* one.
- "NOOOOOOO not again Charles I can't keep doing this every Sunday."
- "LMAOOOO Stroll into the wall on lap 1 again."

**Hardest edge case — `hot_take` vs `reaction`:**
Both are emotional, both are short, both spike during races. The hard cases
are comments that vent a feeling *and* deliver a verdict in the same breath:
"Unreal from Russell, he's better than Hamilton now."

Rule I'll use: **the main-payload test**. Strip the emotional words. If a
falsifiable claim survives, it's a `hot_take`. If nothing survives, it's a
`reaction`. ("NOOOO Charles why" → nothing left → reaction. "Unreal from
Russell, he's better than Hamilton" → "he's better than Hamilton" survives →
hot_take.) Tiebreakers when that's still ambiguous: under ~20 words with
caps/emoji → lean reaction; present-tense and addressed *to* the driver →
reaction; falsifiable and addressed *about* the driver → hot_take.

Second-hardest case — `analysis` vs `hot_take` (stat-decorated takes): remove
the verdict sentence. If a self-contained, falsifiable argument remains, it's
`analysis`. If only an orphaned fact remains, it's `hot_take`.

Sarcasm: I label by literal surface meaning, not inferred sarcasm. "Yeah Stroll
is clearly the most talented driver on the grid 🙄" → `hot_take`. This keeps
annotation reproducible across annotators who don't share the same cultural
read.

---

## Data Sources

<!-- Where exactly are the 200 comments coming from? List the threads/source
     posts so the dataset is reproducible. -->

200 top-level comments sampled across the last 6 race weekends, drawn from a
mix of thread types so each label class has somewhere natural to come from.
Sampling targets: 40% race-day live threads (reaction-heavy), 40% post-race
debrief / technical threads (analysis-heavy), 20% weekday discussion threads
(hot_take-heavy). Inclusion: top-level only (replies bias the label), 10–500
chars, English, not deleted, not from AutoModerator or other bots. Exclusion:
pure questions, pure factual statements, off-topic memes.

| # | Thread type | Source thread | Target comments |
|---|-------------|---------------|----|
| 1 | race_live | r/formula1 — Las Vegas GP race-day live thread | 20 |
| 2 | race_live | r/formula1 — Qatar GP race-day live thread | 20 |
| 3 | race_live | r/formula1 — Abu Dhabi GP race-day live thread | 20 |
| 4 | race_live | r/formula1 — São Paulo GP sprint race live thread | 20 |
| 5 | post_race | r/formula1 — Post-race debrief: Las Vegas | 20 |
| 6 | post_race | r/formula1 — Post-race debrief: Qatar | 20 |
| 7 | post_race | r/formula1 — "Technical: why McLaren's floor is faster in S2" | 20 |
| 8 | weekday | r/formula1 — Daily Discussion (week before Abu Dhabi) | 15 |
| 9 | weekday | r/formula1 — "2025 silly season megathread" | 15 |
| 10 | weekday | r/formula1 — Weekly hot-take thread (sub's recurring "Take Tuesday" or equivalent) | 30 |

Target label distribution (≥20% per class, the spec's balance rule):
`analysis` ≈70, `hot_take` ≈65, `reaction` ≈65.

**Collection note (Milestone 3):** Reddit hard-blocks scripted requests at the
TLS-fingerprint level, so the public JSON endpoints (`.json` on a thread URL)
work but rate-limit aggressively. I'll use PRAW with my personal Reddit app
credentials; if PRAW gets rate-limited, fallback is to fetch the
`old.reddit.com/.../comments.json` endpoint by hand with sensible delays. If a
thread is removed between collection and submission, it's replaced with the
nearest equivalent thread from the same race weekend and the substitution is
logged in the dataset's `notes` column.

---

## Labeling Strategy

<!-- How will you label 200 comments consistently? State the rubric, the
     decision rules for edge cases, and how you'll measure agreement. -->

**Single annotator (me)** labels all 200 comments using the rubric above. I
label in passes by thread type rather than mixed, because mixing class-prior
contexts (race-thread vs. debrief) keeps biasing me toward whichever label
was most recent. Each comment gets stored with the rubric version number it
was labeled under so re-labels are tracked.

**Schema for `data/takemeter_dataset.csv`:**
id,subreddit,thread_id,thread_type,text,label,annotator,rubric_version,notes

`thread_type` ∈ {race_live, post_race, weekday}. `label` ∈ {analysis,
hot_take, reaction}.

**Agreement check (Milestone 5, also feeds stretch §1):** a second annotator
(classmate) independently labels a random 40-comment subset (20% of dataset)
using the same rubric. I compute Cohen's κ overall and per class. Target κ ≥
0.6 (substantial agreement). Disagreements are resolved by discussion and the
resolution rule is logged.

---

## Modeling Approach

<!-- Baseline (Groq zero-shot) and fine-tuned (DistilBERT). State the model,
     the prompt strategy for Groq, the hyperparameters for DistilBERT, and
     justify the picks. -->

**Baseline — Groq zero-shot:**
`llama-3.1-70b-versatile` via Groq API. Prompt includes the three label
definitions verbatim from the *Labels* section + one canonical example per
label + an instruction to output a single label string and nothing else.
Parsed with a regex that accepts `analysis|hot_take|reaction` case-insensitive
and falls back to majority class on parse failure (logged separately).

**Fine-tuned model — DistilBERT:**
Base: `distilbert-base-uncased`. Split: 160 train / 40 test, stratified by
label *and* by thread_id (a comment from a thread in train should not be
matched by a comment from the same thread in test — keeps leakage honest).

Hyperparameters and why:
- **epochs = 4.** 3 underfits on a 160-row train set (pilot loss still
  dropping), 5 starts overfitting (train loss << dev loss). 4 is the sweet
  spot.
- **lr = 2e-5.** HF default for DistilBERT, stable across pilot runs. Anything
  >5e-5 caused loss to diverge once.
- **batch size = 16.** Fits free-Colab T4 with 128 max_length; 32 OOMs.
- **max_length = 128 tokens.** 95th percentile of comment length in my pilot
  pull was ~110 tokens, so 128 captures the tail without padding waste.

**Success criterion:** fine-tuned model beats Groq zero-shot on **macro-F1**
by ≥3 points. If not, I report it honestly and dig into why in the error-
pattern analysis (stretch §3).

---

## Evaluation Plan

<!-- What metrics, what splits, and what would convince you the system actually works? -->

| # | Metric | What I'll report |
|---|--------|------------------|
| 1 | Overall accuracy | Both models, on the same 40-comment test split |
| 2 | Per-class precision / recall / F1 | Both models, both as a table |
| 3 | Macro-F1 | Headline number; success criterion above |
| 4 | Confusion matrix | DistilBERT, plotted; called out in README |
| 5 | 10 misclassified examples | DistilBERT, with my interpretation per row |

Test set is the same 40 comments for both models so the comparison is fair.

---
## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning. -->

1. **The hot_take/reaction boundary will eat my macro-F1.** Even with the
   main-payload test, race-thread comments mix venting and verdicts in ways
   that are hard to call. I expect DistilBERT's per-class F1 on `reaction` to
   be the lowest of the three, and a chunk of the confusion matrix to live in
   the hot_take↔reaction cells. My mitigation is to be explicit about it in
   the rubric and the README rather than pretending the boundary is cleaner
   than it is. If κ on the second-annotator subset comes back below 0.5 on
   those two classes specifically, that's the real ceiling and the model
   can't outperform it.

2. **160 training examples is small, so the fine-tuned model may not actually
   beat Groq.** Groq's 70B has the advantage of a huge prior; DistilBERT only
   has my labels. The risk is that I run the fine-tune, it lands at 0.58
   macro-F1, Groq lands at 0.62, and the project's "fine-tuning beats
   baseline" story doesn't hold. My plan if that happens is to report it
   honestly, run the error-pattern stretch on both models, and write up
   *which* classes Groq is winning and why (probably the rare-vocabulary
   `analysis` comments where the 70B's world knowledge helps). That's still a
   real finding.

3. **PRAW or Reddit's TLS wall.** Documented in the Data Sources note above —
   fallback is hand-fetching `old.reddit.com` JSON with delays.

---

## Architecture

<!-- Diagram the pipeline. Label each stage with the tool/library. -->

Data Collection Labeling Training & Eval ────────────────── ──────────────── ────────────────── PRAW (Reddit API) --> manual + rubric --> Colab T4 threads → comments pandas CSV HuggingFace Trainer race / post / weekday 200 rows DistilBERT-base 160 train / 40 test │ ▼ ┌─────────────────────────────────────────────────┐ │ Baseline: Groq llama-3.1-70b zero-shot │ │ Fine-tuned: distilbert-base-uncased │ │ Metrics: accuracy, P/R/F1, macro-F1, CM │ └─────────────────────────────────────────────────┘ │ ▼ ┌─────────────────────────────────────────────────┐ │ Gradio app on HF Spaces │ │ text → label + per-class confidence bars │ └─────────────────────────────────────────────────┘


---

## AI Tool Plan

<!-- For each milestone, describe which tool, what context you'll give it,
     what you expect back, and how you'll verify the output. -->

**Milestone 3 — Data collection script:**
Give Claude my *Data Sources* section + the CSV schema and ask it to write a
PRAW script that pulls top-level comments from the listed threads, filters by
length and bot-author rules, and dumps to a CSV with the right columns and
empty `label`/`notes` columns ready for me to fill in. I verify by spot-
checking 20 random rows against the live thread to confirm the comment text
matches and no replies or removed comments slipped through.

**Milestone 4 — Baseline (Groq zero-shot):**
Give Claude my *Labels* section verbatim + my *Modeling Approach* prompt
spec + the CSV schema and ask it to write the Colab cell that builds the
prompt, hits Groq, parses the label, and writes predictions back to a
`groq_preds` column. I verify by reading the generated prompt to make sure
my exact label definitions are in it (not paraphrases), running on 10 hand-
labeled comments, and confirming the parser handles "Analysis", "analysis.",
and "label: analysis" all returning `analysis`.

**Milestone 5 — Fine-tune DistilBERT:**
Give Claude my *Modeling Approach* section with the hyperparameters and
justifications, plus the CSV schema, and ask for the HuggingFace Trainer
cell — tokenizer, dataset wrapping, Trainer config, eval that produces
accuracy, per-class P/R/F1, and a confusion matrix. I verify the split is
stratified by thread_id (not just label), the eval metrics function returns
the right shape, and that running on 10 epochs (deliberately overshoot)
shows the train/dev gap I predicted.

**Milestone 6 — Gradio app + stretch features:**
Give Claude my *Architecture* diagram + the trained model artifact path +
the per-class confidence requirement, and ask for a Gradio Blocks app with
a textarea, a "Classify" button, a label output, and a bar chart of all
three softmax scores. For the stretches I'll feed it the relevant section
(*Labeling Strategy* for κ, *Evaluation Plan* for calibration and error
patterns) one at a time so each script stays focused. I verify by hitting
the deployed Space with 5 hand-picked comments (one clear-cut per class,
one hot_take/reaction borderline, one sarcastic) and checking the
confidence bars look reasonable.

---



## Stretch Features (added after the base system)

After the base system is working, I'll add four stretch features. Full
writeups and numbers will live in the README's "Stretch Features" section;
in short:

1. **Inter-annotator agreement** — second annotator labels a 40-comment
   subset (20%), Cohen's κ overall and per class, disagreements analyzed.
2. **Confidence calibration** — reliability diagram (10 bins) + Expected
   Calibration Error (ECE) for DistilBERT, comparing raw softmax vs
   temperature-scaled probabilities.
3. **Error pattern analysis** — every test-set misclassification tagged with
   a failure mode (`sarcasm`, `stat-decorated take`, `short ambiguous`,
   `mixed payload`, `out-of-distribution`), distribution reported with 2–3
   examples per mode.
4. **Deployed interface** — Gradio app on HF Spaces, textarea → label +
   per-class confidence bars. Link in README.