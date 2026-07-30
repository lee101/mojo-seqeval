import random
import warnings

import numpy as np
import pytest

import mojoseqeval as mse
import mojoseqeval.metrics.sequence_labeling as ours
import mojoseqeval.metrics.v1 as ours_v1
import mojoseqeval.scheme as our_scheme

upstream = pytest.importorskip("seqeval.metrics.sequence_labeling")
upstream_v1 = pytest.importorskip("seqeval.metrics.v1")
upstream_scheme = pytest.importorskip("seqeval.scheme")


Y_TRUE = [
    ["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
    ["B-PER", "I-PER", "O"],
]
Y_PRED = [
    ["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],
    ["B-PER", "I-PER", "O"],
]


@pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
def test_default_scores_match_upstream(average):
    for name in ("precision_score", "recall_score", "f1_score"):
        got = getattr(ours, name)(Y_TRUE, Y_PRED, average=average)
        ref = getattr(upstream, name)(Y_TRUE, Y_PRED, average=average)
        assert np.allclose(got, ref)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 2.0, np.inf])
def test_precision_recall_fscore_support_matches(beta):
    got = ours.precision_recall_fscore_support(
        Y_TRUE, Y_PRED, average=None, beta=beta, zero_division=0
    )
    ref = upstream.precision_recall_fscore_support(
        Y_TRUE, Y_PRED, average=None, beta=beta, zero_division=0
    )
    for actual, expected in zip(got, ref):
        assert np.allclose(actual, expected)


def test_sample_weight_signature_compatibility_matches_upstream():
    weights = [1, 10]
    for name in ("precision_score", "recall_score", "f1_score"):
        got = getattr(ours, name)(Y_TRUE, Y_PRED, sample_weight=weights)
        ref = getattr(upstream, name)(Y_TRUE, Y_PRED, sample_weight=weights)
        assert got == pytest.approx(ref)


def test_published_example_values():
    assert mse.accuracy_score(Y_TRUE, Y_PRED) == pytest.approx(0.8)
    assert mse.precision_score(Y_TRUE, Y_PRED) == pytest.approx(0.5)
    assert mse.recall_score(Y_TRUE, Y_PRED) == pytest.approx(0.5)
    assert mse.f1_score(Y_TRUE, Y_PRED) == pytest.approx(0.5)


def test_get_entities_published_vector():
    labels = ["B-PER", "I-PER", "O", "B-LOC"]
    assert ours.get_entities(labels) == [
        ("PER", 0, 1),
        ("LOC", 3, 3),
    ]
    assert ours.get_entities(labels) == upstream.get_entities(labels)


def test_nested_entity_offsets_match_upstream():
    labels = [["B-X", "I-X"], ["O", "S-Y"], ["I-Z", "O"]]
    assert ours.get_entities(labels) == upstream.get_entities(labels)


def test_suffix_form_matches_upstream():
    truth = [["PER-B", "PER-I", "O", "LOC-S"]]
    pred = [["PER-B", "PER-I", "O", "LOC-S"]]
    assert ours.get_entities(truth, suffix=True) == upstream.get_entities(
        truth, suffix=True
    )
    assert ours.f1_score(truth, pred, suffix=True) == 1.0


def test_default_mode_accepts_illegal_iob2_transition_like_upstream():
    truth = [["B-NP", "I-NP", "O"]]
    pred = [["I-NP", "I-NP", "O"]]
    assert ours.f1_score(truth, pred) == upstream.f1_score(truth, pred) == 1.0
    assert ours.f1_score(
        truth, pred, mode="strict", scheme=our_scheme.IOB2
    ) == 0.0


SCHEMES = [
    (our_scheme.IOB1, upstream_scheme.IOB1, "IOB"),
    (our_scheme.IOE1, upstream_scheme.IOE1, "IOE"),
    (our_scheme.IOB2, upstream_scheme.IOB2, "IOB"),
    (our_scheme.IOE2, upstream_scheme.IOE2, "IOE"),
    (our_scheme.IOBES, upstream_scheme.IOBES, "IOBES"),
    (our_scheme.BILOU, upstream_scheme.BILOU, "BILOU"),
]


@pytest.mark.parametrize("our_cls,upstream_cls,prefixes", SCHEMES)
def test_randomized_strict_scheme_parity(our_cls, upstream_cls, prefixes):
    rng = random.Random(100 + len(prefixes))

    def sequence(size):
        result = []
        for _ in range(size):
            prefix = rng.choice(prefixes)
            result.append("O" if prefix == "O" else f"{prefix}-{rng.choice('ABC')}")
        return result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(40):
            lengths = [rng.randrange(1, 30) for _ in range(rng.randrange(1, 6))]
            truth = [sequence(size) for size in lengths]
            pred = [sequence(size) for size in lengths]
            for average in (None, "micro", "macro", "weighted"):
                got = ours.f1_score(
                    truth,
                    pred,
                    average=average,
                    mode="strict",
                    scheme=our_cls,
                    zero_division=0,
                )
                ref = upstream.f1_score(
                    truth,
                    pred,
                    average=average,
                    mode="strict",
                    scheme=upstream_cls,
                    zero_division=0,
                )
                assert np.allclose(got, ref, equal_nan=True)


def test_randomized_default_entity_and_score_parity():
    rng = random.Random(71)

    def sequence(size):
        result = []
        for _ in range(size):
            prefix = rng.choice("OBIES")
            result.append("O" if prefix == "O" else f"{prefix}-{rng.choice('ABC')}")
        return result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(75):
            lengths = [rng.randrange(1, 40) for _ in range(rng.randrange(1, 7))]
            truth = [sequence(size) for size in lengths]
            pred = [sequence(size) for size in lengths]
            assert ours.get_entities(truth) == upstream.get_entities(truth)
            assert ours.f1_score(
                truth, pred, average="macro", zero_division=0
            ) == pytest.approx(
                upstream.f1_score(
                    truth, pred, average="macro", zero_division=0
                ),
                nan_ok=True,
            )


def test_classification_report_text_matches_exactly():
    assert ours.classification_report(Y_TRUE, Y_PRED) == \
        upstream.classification_report(Y_TRUE, Y_PRED)


def test_classification_report_dict_matches():
    got = ours.classification_report(Y_TRUE, Y_PRED, output_dict=True)
    ref = upstream.classification_report(Y_TRUE, Y_PRED, output_dict=True)
    assert got.keys() == ref.keys()
    for name in got:
        assert got[name] == pytest.approx(ref[name])


def test_strict_report_auto_detect_matches_v1():
    got = ours.classification_report(
        Y_TRUE, Y_PRED, mode="strict", output_dict=True
    )
    ref = upstream.classification_report(
        Y_TRUE, Y_PRED, mode="strict", output_dict=True
    )
    for name in got:
        assert got[name] == pytest.approx(ref[name])


def test_v1_api_matches_upstream():
    got = ours_v1.precision_recall_fscore_support(
        Y_TRUE, Y_PRED, scheme=our_scheme.IOB2, average=None
    )
    ref = upstream_v1.precision_recall_fscore_support(
        Y_TRUE, Y_PRED, scheme=upstream_scheme.IOB2, average=None
    )
    for actual, expected in zip(got, ref):
        assert np.allclose(actual, expected)


def test_accuracy_random_parity():
    rng = random.Random(9)
    tags = ["O", "B-X", "I-X", "B-Y", "I-Y"]
    truth = [[rng.choice(tags) for _ in range(137)] for _ in range(29)]
    pred = [[rng.choice(tags) for _ in sentence] for sentence in truth]
    assert ours.accuracy_score(truth, pred) == upstream.accuracy_score(truth, pred)


def test_performance_measure_matches_upstream():
    assert ours.performance_measure(Y_TRUE, Y_PRED) == \
        upstream.performance_measure(Y_TRUE, Y_PRED)


def test_token_counts_simd_tail_matches_upstream():
    tags = ["O", "B-X", "I-X", "B-Y", "I-Y"]
    truth = [[tags[idx % len(tags)] for idx in range(13)]]
    pred = [[tags[(idx + (idx % 3 == 0)) % len(tags)] for idx in range(13)]]
    assert ours.performance_measure(truth, pred) == \
        upstream.performance_measure(truth, pred)


@pytest.mark.parametrize("size", range(0, 33))
def test_token_counts_all_small_simd_tails_match_upstream(size):
    tags = ["O", "B-X", "I-X", "B-Y", "I-Y"]
    truth = [[tags[idx % len(tags)] for idx in range(size)]]
    pred = [[tags[(idx * 3 + 1) % len(tags)] for idx in range(size)]]
    assert ours.performance_measure(truth, pred) == \
        upstream.performance_measure(truth, pred)


def test_ffi_rejects_wrong_dtype_and_strided_buffers():
    with pytest.raises(TypeError, match="require int64"):
        ours._extract(np.arange(4, dtype=np.int32), 0)
    with pytest.raises(ValueError, match="C-contiguous"):
        ours._extract(np.arange(8, dtype=np.int64)[::2], 0)


@pytest.mark.parametrize(
    "args",
    [
        ("B", "I", "X", "X"),
        ("I", "O", "X", "X"),
        ("O", "E", "", "X"),
        ("S", "I", "X", "X"),
    ],
)
def test_chunk_transition_helpers_match_upstream(args):
    assert ours.start_of_chunk(*args) == upstream.start_of_chunk(*args)
    assert ours.end_of_chunk(*args) == upstream.end_of_chunk(*args)


def test_v1_classification_report_matches_upstream():
    got = ours_v1.classification_report(
        Y_TRUE, Y_PRED, scheme=our_scheme.IOB2, output_dict=True
    )
    ref = upstream_v1.classification_report(
        Y_TRUE, Y_PRED, scheme=upstream_scheme.IOB2, output_dict=True
    )
    for name in got:
        assert got[name] == pytest.approx(ref[name])


@pytest.mark.parametrize("scheme_id", [0, our_scheme.IOB2.scheme_id])
@pytest.mark.parametrize("size", [97, 750_003])
def test_paired_extract_serial_and_parallel_threshold(scheme_id, size):
    codes = np.full(size, 2 + 16, dtype=np.int64)
    paired_true, paired_pred = ours._extract_pair(codes, codes, scheme_id)
    reference = ours._extract(codes, scheme_id)
    for paired in (paired_true, paired_pred):
        for actual, expected in zip(paired, reference):
            assert np.array_equal(actual, expected)


def test_zero_division_behavior_matches():
    truth = [["O", "O"]]
    pred = [["O", "O"]]
    for value in (0, 1):
        assert ours.f1_score(
            truth, pred, zero_division=value
        ) == upstream.f1_score(truth, pred, zero_division=value)


def test_inconsistent_sentence_lengths_raise_same_error():
    truth = [["O", "O"], ["B-X"]]
    pred = [["O"], ["B-X"]]
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        ours.f1_score(truth, pred)


def test_non_nested_score_input_rejected():
    with pytest.raises(TypeError, match="without list of list"):
        ours.f1_score(["O"], ["O"])


@pytest.mark.parametrize("average", ["binary", "samples", "invalid"])
def test_invalid_average_rejected(average):
    with pytest.raises(ValueError, match="average has to be one of"):
        ours.f1_score(Y_TRUE, Y_PRED, average=average)


def test_negative_beta_rejected():
    with pytest.raises(ValueError, match="beta should be >=0"):
        ours.precision_recall_fscore_support(Y_TRUE, Y_PRED, beta=-1)


@pytest.mark.parametrize("our_cls,upstream_cls,prefixes", SCHEMES)
def test_scheme_tokens_extract_same_entities(our_cls, upstream_cls, prefixes):
    rng = random.Random(len(prefixes))
    labels = []
    for _ in range(100):
        prefix = rng.choice(prefixes)
        labels.append("O" if prefix == "O" else f"{prefix}-{rng.choice('AB')}")
    got = [
        entity.to_tuple()
        for entity in our_scheme.Tokens(labels, our_cls, sent_id=3).entities
    ]
    ref = [
        entity.to_tuple()
        for entity in upstream_scheme.Tokens(labels, upstream_cls, sent_id=3).entities
    ]
    assert got == ref


@pytest.mark.parametrize(
    "labels,expected",
    [
        ([["B-X", "I-X", "O"]], our_scheme.IOB2),
        ([["I-X", "E-X", "O"]], our_scheme.IOE2),
        ([["S-X"]], our_scheme.IOBES),
        ([["U-X"]], our_scheme.BILOU),
    ],
)
def test_auto_detect(labels, expected):
    assert our_scheme.auto_detect(labels) is expected
