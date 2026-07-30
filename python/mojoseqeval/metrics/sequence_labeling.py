"""Sequence-labelling metrics with entity scans implemented in Mojo."""

from __future__ import annotations

from itertools import chain
from operator import eq
import warnings

import numpy as np

from .._lib import addr, lib
from ..scheme import BILOU, IOB1, IOB2, IOBES, IOE1, IOE2, Token, auto_detect

try:
    from sklearn.exceptions import UndefinedMetricWarning
except ImportError:
    class UndefinedMetricWarning(UserWarning):
        pass


_PREFIX = {"O": 0, "I": 1, "B": 2, "E": 3, "S": 4, "U": 5, "L": 6, ".": 7}
_SCHEMES = {IOB1: 1, IOE1: 2, IOB2: 3, IOE2: 4, IOBES: 5, BILOU: 6}


def check_consistent_length(y_true, y_pred):
    len_true = list(map(len, y_true))
    len_pred = list(map(len, y_pred))
    is_list = set(map(type, y_true)) | set(map(type, y_pred))
    if is_list != {list}:
        raise TypeError("Found input variables without list of list.")
    if len(y_true) != len(y_pred) or len_true != len_pred:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples:\n"
            f"{len_true}\n{len_pred}"
        )


def _flatten_entities(seq):
    if any(isinstance(item, list) for item in seq):
        return [tag for sentence in seq for tag in sentence + ["O"]]
    return list(seq)


def _parts(chunk, suffix):
    if suffix:
        char = chunk[-1]
        type_name = chunk[:-1].rsplit("-", maxsplit=1)[0] or "_"
    else:
        char = chunk[0]
        type_name = chunk[1:].split("-", maxsplit=1)[-1] or "_"
    return char, type_name


def _warn_tag(chunk, suffix):
    valid = chunk in ("O", "B", "I", "E", "S") or (
        chunk.endswith(("-B", "-I", "-E", "-S")) if suffix else
        chunk.startswith(("B-", "I-", "E-", "S-"))
    )
    if not valid:
        warnings.warn(f"{chunk} seems not to be NE tag.")


def _validate_strict(tags, scheme, suffix):
    allowed = scheme.allowed_chars
    for chunk in tags:
        char = chunk[-1] if suffix else chunk[0]
        if char not in _PREFIX:
            raise KeyError(char)
        if char not in allowed:
            names = "|".join(sorted(allowed))
            raise ValueError(
                f"Invalid token is found: {chunk}. Allowed prefixes are: {names}."
            )


def _empty_i64(size):
    return np.empty(max(1, size), dtype=np.int64)


def _extract(codes, scheme_id):
    capacity = max(1, len(codes))
    storage = np.empty((3, capacity), dtype=np.int64)
    types, starts, ends = storage
    if scheme_id:
        count = lib().mse_extract_strict(
            addr(codes), len(codes), scheme_id, addr(types), addr(starts), addr(ends)
        )
    else:
        count = lib().mse_extract_default(
            addr(codes), len(codes), addr(types), addr(starts), addr(ends)
        )
    if count < 0 or count > len(codes):
        raise RuntimeError(
            f"Mojo entity extractor returned invalid count {count} "
            f"for capacity {len(codes)}"
        )
    return types[:count], starts[:count], ends[:count]


def _extract_pair(true_codes, pred_codes, scheme_id):
    capacity = max(1, len(true_codes), len(pred_codes))
    storage = np.empty((6, capacity), dtype=np.int64)
    true_types, true_starts, true_ends, pred_types, pred_starts, pred_ends = storage
    counts = np.empty(2, dtype=np.int64)
    if scheme_id:
        lib().mse_extract_pair_strict(
            addr(true_codes), len(true_codes), addr(pred_codes), len(pred_codes),
            scheme_id,
            addr(true_types), addr(true_starts), addr(true_ends),
            addr(pred_types), addr(pred_starts), addr(pred_ends), addr(counts),
        )
    else:
        lib().mse_extract_pair_default(
            addr(true_codes), len(true_codes),
            addr(true_types), addr(true_starts), addr(true_ends),
            addr(pred_codes), len(pred_codes),
            addr(pred_types), addr(pred_starts), addr(pred_ends), addr(counts),
        )
    n_true, n_pred = map(int, counts)
    if not 0 <= n_true <= len(true_codes) or not 0 <= n_pred <= len(pred_codes):
        raise RuntimeError(
            "Mojo paired extractor returned invalid counts "
            f"({n_true}, {n_pred}) for capacities "
            f"({len(true_codes)}, {len(pred_codes)})"
        )
    return (
        (true_types[:n_true], true_starts[:n_true], true_ends[:n_true]),
        (pred_types[:n_pred], pred_starts[:n_pred], pred_ends[:n_pred]),
    )


def _encode_pair(y_true, y_pred, suffix, scheme=None):
    true_flat = _flatten_entities(y_true)
    pred_flat = _flatten_entities(y_pred)
    unique_tags = set(true_flat)
    unique_tags.update(pred_flat)
    parsed = {chunk: _parts(chunk, suffix) for chunk in unique_tags}

    if scheme is None:
        if any(
            chunk not in ("O", "B", "I", "E", "S")
            and not (
                chunk.endswith(("-B", "-I", "-E", "-S")) if suffix else
                chunk.startswith(("B-", "I-", "E-", "S-"))
            )
            for chunk in unique_tags
        ):
            for chunk in chain(true_flat, pred_flat):
                _warn_tag(chunk, suffix)
    else:
        allowed = scheme.allowed_chars
        if any(
            (chunk[-1] if suffix else chunk[0]) not in allowed
            for chunk in unique_tags
        ):
            _validate_strict(chain(true_flat, pred_flat), scheme, suffix)

    type_names = sorted({type_name for _, type_name in parsed.values()})
    type_ids = {name: idx for idx, name in enumerate(type_names)}
    packed = {
        chunk: type_ids[type_name] * 16 + _PREFIX.get(char, 8)
        for chunk, (char, type_name) in parsed.items()
    }
    true_codes = np.fromiter(
        map(packed.__getitem__, true_flat), dtype=np.int64, count=len(true_flat)
    )
    pred_codes = np.fromiter(
        map(packed.__getitem__, pred_flat), dtype=np.int64, count=len(pred_flat)
    )
    return true_codes, pred_codes, type_names


def _statistics(y_true, y_pred, suffix=False, scheme=None):
    check_consistent_length(y_true, y_pred)
    true_codes, pred_codes, all_type_names = _encode_pair(
        y_true, y_pred, suffix, scheme
    )
    scheme_id = _SCHEMES.get(scheme, 0)
    true_entities, pred_entities = _extract_pair(
        true_codes, pred_codes, scheme_id
    )
    size = max(1, len(all_type_names))
    count_storage = np.empty((3, size), dtype=np.int64)
    true_counts, pred_counts, tp_counts = count_storage
    lib().mse_count_entities(
        addr(true_entities[0]),
        addr(true_entities[1]),
        addr(true_entities[2]),
        len(true_entities[0]),
        addr(pred_entities[0]),
        addr(pred_entities[1]),
        addr(pred_entities[2]),
        len(pred_entities[0]),
        len(all_type_names),
        addr(true_counts),
        addr(pred_counts),
        addr(tp_counts),
    )
    used = (true_counts[:len(all_type_names)] + pred_counts[:len(all_type_names)]) > 0
    names = [name for name, keep in zip(all_type_names, used) if keep]
    return (
        names,
        pred_counts[:len(all_type_names)][used],
        tp_counts[:len(all_type_names)][used],
        true_counts[:len(all_type_names)][used],
    )


def _warn_prf(average, modifier, msg_start, result_size):
    axis0, axis1 = "sample", "label"
    if average == "samples":
        axis0, axis1 = axis1, axis0
    msg = (
        f"{msg_start} ill-defined and being set to 0.0 {{0}} no "
        f"{modifier} {axis0}s. Use `zero_division` parameter to control this behavior."
    )
    detail = "due to" if result_size == 1 else f"in {axis1}s with"
    warnings.warn(msg.format(detail), UndefinedMetricWarning, stacklevel=3)


def _divide(numerator, denominator, metric, modifier, average, warn_for, zero_division):
    mask = denominator == 0
    safe = denominator.copy()
    safe[mask] = 1
    result = numerator / safe
    if not np.any(mask):
        return result
    result[mask] = 0.0 if zero_division in ("warn", 0) else 1.0
    if zero_division == "warn" and metric in warn_for:
        if metric in warn_for and "f-score" in warn_for:
            start = f"{metric.title()} and F-score are"
        elif metric in warn_for:
            start = f"{metric.title()} is"
        elif "f-score" in warn_for:
            start = "F-score is"
        else:
            return result
        _warn_prf(average, modifier, start, len(result))
    return result


def _scores_from_statistics(
    stats,
    *,
    average,
    warn_for,
    beta,
    zero_division,
):
    if beta < 0:
        raise ValueError("beta should be >=0 in the F-beta score")
    options = (None, "micro", "macro", "weighted")
    if average not in options:
        raise ValueError(f"average has to be one of {options}")

    _, pred_sum, tp_sum, true_sum = stats
    if average == "micro":
        pred_sum = np.array([pred_sum.sum()])
        tp_sum = np.array([tp_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    precision = _divide(
        tp_sum, pred_sum, "precision", "predicted", average, warn_for, zero_division
    )
    recall = _divide(
        tp_sum, true_sum, "recall", "true", average, warn_for, zero_division
    )
    if zero_division == "warn" and warn_for == ("f-score",):
        if (pred_sum[true_sum == 0] == 0).any():
            _warn_prf(
                average, "true nor predicted", "F-score is", len(true_sum)
            )

    if np.isposinf(beta):
        f_score = recall
    else:
        beta2 = beta ** 2
        denom = beta2 * precision + recall
        denom[denom == 0] = 1
        f_score = (1 + beta2) * precision * recall / denom

    if average == "weighted":
        weights = true_sum
        if weights.sum() == 0:
            value = 0.0 if zero_division in ("warn", 0) else 1.0
            return (
                value if pred_sum.sum() == 0 else 0.0,
                value,
                value if pred_sum.sum() == 0 else 0.0,
                sum(true_sum),
            )
    else:
        weights = None

    if average is not None:
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = sum(true_sum)
    return precision, recall, f_score, true_sum


def precision_recall_fscore_support(
    y_true,
    y_pred,
    *,
    average=None,
    warn_for=("precision", "recall", "f-score"),
    beta=1.0,
    sample_weight=None,
    zero_division="warn",
    suffix=False,
):
    del sample_weight
    stats = _statistics(y_true, y_pred, suffix=suffix)
    return _scores_from_statistics(
        stats,
        average=average,
        warn_for=warn_for,
        beta=beta,
        zero_division=zero_division,
    )


def _metric_score(
    metric,
    y_true,
    y_pred,
    *,
    average,
    suffix,
    mode,
    sample_weight,
    zero_division,
    scheme,
):
    del sample_weight
    strict_scheme = scheme if mode == "strict" and scheme else None
    stats = _statistics(y_true, y_pred, suffix=suffix, scheme=strict_scheme)
    p, r, f, _ = _scores_from_statistics(
        stats,
        average=average,
        warn_for=(metric,),
        beta=1,
        zero_division=zero_division,
    )
    return {"precision": p, "recall": r, "f-score": f}[metric]


def precision_score(
    y_true,
    y_pred,
    *,
    average="micro",
    suffix=False,
    mode=None,
    sample_weight=None,
    zero_division="warn",
    scheme=None,
):
    return _metric_score(
        "precision", y_true, y_pred, average=average, suffix=suffix, mode=mode,
        sample_weight=sample_weight, zero_division=zero_division, scheme=scheme
    )


def recall_score(
    y_true,
    y_pred,
    *,
    average="micro",
    suffix=False,
    mode=None,
    sample_weight=None,
    zero_division="warn",
    scheme=None,
):
    return _metric_score(
        "recall", y_true, y_pred, average=average, suffix=suffix, mode=mode,
        sample_weight=sample_weight, zero_division=zero_division, scheme=scheme
    )


def f1_score(
    y_true,
    y_pred,
    *,
    average="micro",
    suffix=False,
    mode=None,
    sample_weight=None,
    zero_division="warn",
    scheme=None,
):
    return _metric_score(
        "f-score", y_true, y_pred, average=average, suffix=suffix, mode=mode,
        sample_weight=sample_weight, zero_division=zero_division, scheme=scheme
    )


def _token_arrays(y_true, y_pred):
    true_flat = [tag for sentence in y_true for tag in sentence] if any(
        isinstance(item, list) for item in y_true
    ) else list(y_true)
    pred_flat = [tag for sentence in y_pred for tag in sentence] if any(
        isinstance(item, list) for item in y_pred
    ) else list(y_pred)
    labels = sorted((set(true_flat) | set(pred_flat)) - {"O"})
    ids = {"O": 0, **{tag: idx + 1 for idx, tag in enumerate(labels)}}
    n = min(len(true_flat), len(pred_flat))
    truth = np.fromiter((ids[tag] for tag in true_flat[:n]), dtype=np.int64, count=n)
    pred = np.fromiter((ids[tag] for tag in pred_flat[:n]), dtype=np.int64, count=n)
    return truth, pred, len(true_flat)


def _token_counts(y_true, y_pred):
    truth, pred, denominator = _token_arrays(y_true, y_pred)
    result = np.empty(5, dtype=np.int64)
    lib().mse_token_counts(addr(truth), addr(pred), len(truth), addr(result))
    return result, denominator


def accuracy_score(y_true, y_pred):
    nested = any(isinstance(item, list) for item in y_true)
    truth = chain.from_iterable(y_true) if nested else iter(y_true)
    pred = chain.from_iterable(y_pred) if nested else iter(y_pred)
    denominator = sum(map(len, y_true)) if nested else len(y_true)
    return sum(map(eq, truth, pred)) / denominator


def performance_measure(y_true, y_pred):
    counts, _ = _token_counts(y_true, y_pred)
    return {
        "TP": int(counts[1]),
        "FP": int(counts[2]),
        "FN": int(counts[3]),
        "TN": int(counts[4]),
    }


def end_of_chunk(prev_tag, tag, prev_type, type_):
    return (
        prev_tag in ("E", "S")
        or (prev_tag in ("B", "I") and tag in ("B", "S", "O"))
        or (prev_tag not in ("O", ".") and prev_type != type_)
    )


def start_of_chunk(prev_tag, tag, prev_type, type_):
    return (
        tag in ("B", "S")
        or (prev_tag in ("E", "S") and tag in ("E", "I"))
        or (prev_tag == "O" and tag in ("E", "I"))
        or (tag not in ("O", ".") and prev_type != type_)
    )


def get_entities(seq, suffix=False):
    flat = _flatten_entities(seq)
    unique_tags = set(flat)
    if any(
        chunk not in ("O", "B", "I", "E", "S")
        and not (
            chunk.endswith(("-B", "-I", "-E", "-S")) if suffix else
            chunk.startswith(("B-", "I-", "E-", "S-"))
        )
        for chunk in unique_tags
    ):
        for chunk in flat:
            _warn_tag(chunk, suffix)
    parsed = {chunk: _parts(chunk, suffix) for chunk in unique_tags}
    names = sorted({name for _, name in parsed.values()})
    type_ids = {name: idx for idx, name in enumerate(names)}
    codes = np.fromiter(
        map(
            {
                tag: type_ids[name] * 16 + _PREFIX.get(char, 8)
                for tag, (char, name) in parsed.items()
            }.__getitem__,
            flat,
        ),
        dtype=np.int64,
        count=len(flat),
    )
    types, starts, ends = _extract(codes, 0)
    return [
        (names[int(type_id)], int(start), int(end))
        for type_id, start, end in zip(types, starts, ends)
    ]


def _report(stats, digits, output_dict, zero_division):
    names = stats[0]
    if not names:
        raise ValueError("max() iterable argument is empty")
    p, r, f, support = _scores_from_statistics(
        stats, average=None,
        warn_for=("precision", "recall", "f-score"),
        beta=1, zero_division=zero_division
    )
    rows = list(zip(names, p, r, f, support))
    for average in ("micro", "macro", "weighted"):
        values = _scores_from_statistics(
            stats, average=average,
            warn_for=("precision", "recall", "f-score"),
            beta=1, zero_division=zero_division
        )
        rows.append((f"{average} avg", *values))
    if output_dict:
        return {
            name: {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": count,
            }
            for name, precision, recall, f1, count in rows
        }

    width = max(max(map(len, names)), len("weighted avg"), digits)
    head = ("{:>{width}s} " + " {:>9}" * 4).format(
        "", "precision", "recall", "f1-score", "support", width=width
    )
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}"
    rendered = []
    for idx, row in enumerate(rows):
        if idx == len(names) or idx == len(rows):
            rendered.append("")
        rendered.append(row_fmt.format(*row, width=width, digits=digits))
    rendered.append("")
    return head + "\n\n" + "\n".join(rendered)


def classification_report(
    y_true,
    y_pred,
    digits=2,
    suffix=False,
    output_dict=False,
    mode=None,
    sample_weight=None,
    zero_division="warn",
    scheme=None,
):
    del sample_weight
    strict_scheme = None
    if mode == "strict":
        strict_scheme = scheme
        if strict_scheme is None or not issubclass(strict_scheme, Token):
            strict_scheme = auto_detect(y_true, suffix)
    stats = _statistics(y_true, y_pred, suffix=suffix, scheme=strict_scheme)
    return _report(stats, digits, output_dict, zero_division)
