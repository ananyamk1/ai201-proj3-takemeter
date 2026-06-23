"""
TakeMeter — Reddit collector.
Pulls top-level comments from a list of r/formula1 threads and writes
data/takemeter_dataset.csv with header `text,label` and empty label column.

Usage:
    export REDDIT_CLIENT_ID=...
    export REDDIT_CLIENT_SECRET=...
    export REDDIT_USERNAME=your_reddit_username   # used only for User-Agent
    python src/collect.py
"""

import csv
import os
import re
import time
from pathlib import Path

import praw

# ---- Config ----------------------------------------------------------------

THREAD_IDS = [
    "1udc5gc",  # McLaren / Hamilton Barcelona counterfactual
    "1udfm4o",  # Throwback: Lando's final two laps
    "1ucfpot",  # Race week again
    "1ucmw78",  # Important week for Red Bull & Verstappen
    "1ucrcp8",  # Five poles / five wins / three sprint wins
    # add more thread IDs here until you have ~250 candidate comments
]

PER_THREAD_CAP   = 60   # take at most this many top-level comments per thread
MIN_CHARS        = 10
MAX_CHARS        = 500
OUT_PATH         = Path("data/takemeter_dataset.csv")
FULL_PATH        = Path("data/takemeter_dataset_full.csv")  # for your own tracking

# ---- Helpers ---------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+")

def clean(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = URL_RE.sub("[link]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def keep(comment) -> bool:
    if comment.author is None:                       # deleted user
        return False
    if str(comment.author).lower() in {"automoderator", "f1-bot"}:
        return False
    if comment.body in {"[deleted]", "[removed]"}:
        return False
    body = clean(comment.body)
    return MIN_CHARS <= len(body) <= MAX_CHARS

# ---- Main ------------------------------------------------------------------

def main() -> None:
    reddit = praw.Reddit(
        client_id     = os.environ["REDDIT_CLIENT_ID"],
        client_secret = os.environ["REDDIT_CLIENT_SECRET"],
        user_agent    = f"TakeMeter/0.1 by /u/{os.environ['REDDIT_USERNAME']}",
    )
    reddit.read_only = True

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submitted_rows = []   # (text,)              — final submission CSV
    full_rows      = []   # (id, thread_id, text) — your tracking copy

    for tid in THREAD_IDS:
        sub = reddit.submission(id=tid)
        sub.comment_sort = "top"
        sub.comments.replace_more(limit=0)     # drop "load more" stubs
        kept = 0
        for c in sub.comments:                  # top-level only
            if not keep(c):
                continue
            text = clean(c.body)
            submitted_rows.append((text,))
            full_rows.append((c.id, tid, text))
            kept += 1
            if kept >= PER_THREAD_CAP:
                break
        print(f"thread {tid}: kept {kept}")
        time.sleep(1.0)                         # be polite

    # --- write 2-column submission file ---
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["text", "label"])
        for (text,) in submitted_rows:
            w.writerow([text, ""])              # label empty, you fill it

    # --- write tracking copy ---
    with FULL_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["comment_id", "thread_id", "text", "label", "notes"])
        for cid, tid, text in full_rows:
            w.writerow([cid, tid, text, "", ""])

    print(f"\nwrote {len(submitted_rows)} rows → {OUT_PATH}")
    print(f"wrote {len(full_rows)} rows → {FULL_PATH}")

if __name__ == "__main__":
    main()
