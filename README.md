# mojo-seqeval

`mojo-seqeval` is a Mojo port of the compute-heavy core of
[`seqeval`](https://github.com/chakki-works/seqeval), the standard Python
library for exact-span evaluation of sequence labelling and named-entity
recognition. It keeps the familiar Python API while replacing entity scanning
and count aggregation with compiled Mojo kernels.

The Python package is named `mojoseqeval`, so it can be installed beside the
upstream `seqeval` package for parity checks.

## Coverage

The tested compatibility surface includes:

- `accuracy_score`, `precision_score`, `recall_score`, `f1_score`,
  `classification_report`, and `performance_measure`
- `precision_recall_fscore_support`, `get_entities`, `start_of_chunk`, and
  `end_of_chunk`
- default conlleval-compatible evaluation and strict evaluation
- `IOB1`, `IOB2`, `IOE1`, `IOE2`, `IOBES`, and `BILOU`
- prefix and suffix tag forms; micro, macro, weighted, and per-class results
- the strict `metrics.v1` scoring/report entry points and `Tokens.entities`

The reporting layer and input validation run in Python because they are not
compute-bound. Direct use of upstream's internal `seqeval.reporters` classes
and custom third-party `Token` subclasses is not covered. `sample_weight` is
accepted for signature compatibility; like seqeval 1.2.2 for these supported
averages, it does not change entity counts.

## Install and build

The pinned Mojo nightly and all Python dependencies are managed by Pixi:

```bash
pixi install
pixi run build
pixi run test
```

The build creates `dist/libmojo-seqeval.so`. The Python wrapper builds a stale
or missing library on demand when `mojo` is available; deployments can instead
point `MOJOSEQEVAL_LIB` at a prebuilt shared library.

## Usage

```python
from mojoseqeval.metrics import classification_report, f1_score
from mojoseqeval.scheme import IOB2

y_true = [["B-PER", "I-PER", "O", "B-LOC"]]
y_pred = [["B-PER", "I-PER", "O", "B-LOC"]]

print(f1_score(y_true, y_pred))
print(classification_report(y_true, y_pred, mode="strict", scheme=IOB2))
```

Run the example from the checkout with:

```bash
pixi run python -c 'from mojoseqeval.metrics import f1_score; print(f1_score([["B-X"]], [["B-X"]]))'
```

## Benchmark

Measured on 2026-07-30 with an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
x86-64. Times are the best of three warm runs on the same 500,000-token
corpus. Run them only through `pixi run bench`, which takes a machine-wide
lock.

| operation | mojo-seqeval | seqeval 1.2.2 | speedup |
|---|---:|---:|---:|
| f1_score, 500k tokens | 163.3 ms | 1080.7 ms | 6.62x |
| classification_report, 500k tokens | 171.8 ms | 6813.3 ms | 39.65x |
| get_entities, 500k tokens | 85.9 ms | 423.1 ms | 4.93x |
| accuracy_score, 500k tokens | 37.7 ms | 89.0 ms | 2.36x |

Entity-heavy metrics benefit from cached tag encoding and the compiled state
machine. Independent true/predicted scans run in parallel above 1.5 million
combined tokens; smaller inputs stay serial to avoid thread-launch overhead.
Token count aggregation uses SIMD with a scalar remainder loop. Accuracy avoids
integer compaction entirely and compares the original strings without flattening
copies.

GPU acceleration is intentionally not included. These kernels perform
single-pass integer comparisons and state transitions with well under roughly
two operations per byte moved, so device transfer and launch costs would
outweigh useful work.

## How it works

Python validates the nested label sequences and assigns integer IDs to entity
types. Each tag becomes an `int64`: the low four bits hold the prefix and the
remaining bits hold the type ID. Contiguous NumPy buffers cross the C ABI as
integer addresses.

Mojo scans those buffers using the conlleval transition rules or one of the six
strict scheme state machines. It writes entity type/start/end triples into
caller-owned arrays, intersects true and predicted spans, and aggregates
per-type counts. No string or heap allocation crosses the FFI boundary.
Python then applies seqeval's averaging, zero-division, warning, and report
formatting behavior.

## License

MIT
