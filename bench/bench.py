"""Benchmark mojo-seqeval against seqeval 1.2.2 on identical label corpora."""

from __future__ import annotations

import math
import os
import platform
import random
import sys
import time

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojoseqeval as mse  # noqa: E402
from mojoseqeval.metrics.sequence_labeling import get_entities as mojo_entities  # noqa: E402
from seqeval.metrics import (  # noqa: E402
    accuracy_score as upstream_accuracy,
    classification_report as upstream_report,
    f1_score as upstream_f1,
)
from seqeval.metrics.sequence_labeling import get_entities as upstream_entities  # noqa: E402


def machine_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def timeit(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def corpus(sentences=1000, tokens=500):
    rng = random.Random(2026)
    labels = ["O"] * 85 + [
        "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
    ] * 3
    truth = [[rng.choice(labels) for _ in range(tokens)] for _ in range(sentences)]
    pred = [
        [tag if rng.random() < 0.96 else rng.choice(labels) for tag in sentence]
        for sentence in truth
    ]
    return truth, pred


def main():
    truth, pred = corpus()
    cases = [
        (
            "f1_score, 500k tokens",
            lambda: mse.f1_score(truth, pred),
            lambda: upstream_f1(truth, pred),
        ),
        (
            "classification_report, 500k tokens",
            lambda: mse.classification_report(truth, pred, output_dict=True),
            lambda: upstream_report(truth, pred, output_dict=True),
        ),
        (
            "get_entities, 500k tokens",
            lambda: mojo_entities(truth),
            lambda: upstream_entities(truth),
        ),
        (
            "accuracy_score, 500k tokens",
            lambda: mse.accuracy_score(truth, pred),
            lambda: upstream_accuracy(truth, pred),
        ),
    ]

    print(f"Machine: {machine_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| operation | mojo-seqeval | seqeval 1.2.2 | speedup |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs in cases:
        ours()
        theirs()
        mojo_time = timeit(ours)
        upstream_time = timeit(theirs)
        print(
            f"| {name} | {mojo_time * 1000:.1f} ms | "
            f"{upstream_time * 1000:.1f} ms | {upstream_time / mojo_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
