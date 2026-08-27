"""Sequence-labelling metric kernels exposed through a small C ABI."""

from std.sys.info import simd_width_of

comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime O = 0
comptime I = 1
comptime B = 2
comptime E = 3
comptime S = 4
comptime U = 5
comptime L = 6
comptime DOT = 7


def ptr(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def prefix(code: Int) -> Int:
    return code % 16


def entity_type(code: Int) -> Int:
    return code // 16


def default_end(prev: Int, current: Int) -> Bool:
    var pp = prefix(prev)
    var cp = prefix(current)
    if pp == E or pp == S:
        return True
    if (pp == B or pp == I) and (cp == B or cp == S or cp == O):
        return True
    return pp != O and pp != DOT and entity_type(prev) != entity_type(current)


def default_start(prev: Int, current: Int) -> Bool:
    var pp = prefix(prev)
    var cp = prefix(current)
    if cp == B or cp == S:
        return True
    if (pp == E or pp == S) and (cp == E or cp == I):
        return True
    if pp == O and (cp == E or cp == I):
        return True
    return cp != O and cp != DOT and entity_type(prev) != entity_type(current)


def strict_start(prev: Int, current: Int, scheme: Int) -> Bool:
    var pp = prefix(prev)
    var cp = prefix(current)
    var same = entity_type(prev) == entity_type(current)
    if scheme == 1:
        return (
            (pp == O and cp == I)
            or (pp == I and cp == I and not same)
            or (pp == B and cp == I)
            or (pp == I and cp == B and same)
            or (pp == B and cp == B and same)
        )
    if scheme == 2:
        return (
            (pp == O and cp == I)
            or (pp == I and cp == I and not same)
            or (pp == E and cp == I)
            or (pp == E and cp == E and same)
        )
    if scheme == 3:
        return cp == B
    if scheme == 4:
        return (
            ((pp == O or pp == E) and (cp == I or cp == E))
            or (pp == I and (cp == I or cp == E) and not same)
        )
    if scheme == 5:
        return cp == B or cp == S
    return cp == B or cp == U


def strict_inside(prev: Int, current: Int, scheme: Int) -> Bool:
    var pp = prefix(prev)
    var cp = prefix(current)
    if entity_type(prev) != entity_type(current):
        return False
    if scheme == 1 or scheme == 3:
        return (pp == B or pp == I) and cp == I
    if scheme == 2 or scheme == 4:
        return pp == I and (cp == I or cp == E)
    if scheme == 5:
        return (pp == B or pp == I) and (cp == I or cp == E)
    return (pp == B or pp == I) and (cp == I or cp == L)


def strict_end(prev: Int, current: Int, scheme: Int) -> Bool:
    var pp = prefix(prev)
    var cp = prefix(current)
    var same = entity_type(prev) == entity_type(current)
    if scheme == 1:
        return (
            (pp == I and cp == I and not same)
            or ((pp == I or pp == B) and cp == O)
            or (pp == I and cp == B)
            or (pp == B and cp == I and not same)
            or (pp == B and cp == B and same)
        )
    if scheme == 2:
        return (
            (pp == I and cp == I and not same)
            or (pp == I and cp == O)
            or (pp == I and cp == E and not same)
            or (pp == E and cp == I and same)
            or (pp == E and cp == E and same)
        )
    if scheme == 3:
        return (
            ((pp == I or pp == B) and cp == O)
            or (pp == I and cp == I and not same)
            or ((pp == I or pp == B) and cp == B)
            or (pp == B and cp == I and not same)
        )
    if scheme == 4:
        return pp == E
    if scheme == 5:
        return pp == S or pp == E
    return pp == U or pp == L


def write_entity(
    types: IPtr, starts: IPtr, ends: IPtr, count: Int, code: Int, start: Int, end: Int
):
    types[unsafe_offset=count] = Int64(entity_type(code))
    starts[unsafe_offset=count] = Int64(start)
    ends[unsafe_offset=count] = Int64(end)


def extract_default(
    codes: IPtr,
    n: Int,
    types: IPtr,
    starts: IPtr,
    ends: IPtr,
) -> Int:
    var prev = O
    var begin = 0
    var count = 0
    for pos in range(n + 1):
        var current = Int(codes[unsafe_offset=pos]) if pos < n else O
        if default_end(prev, current):
            write_entity(types, starts, ends, count, prev, begin, pos - 1)
            count += 1
        if default_start(prev, current):
            begin = pos
        prev = current
    return count


def extract_strict(
    codes: IPtr,
    n: Int,
    scheme: Int,
    types: IPtr,
    starts: IPtr,
    ends: IPtr,
) -> Int:
    var pos = 0
    var prev = O
    var count = 0
    while pos < n:
        var current = Int(codes[unsafe_offset=pos])
        if strict_start(prev, current, scheme):
            var scan = pos + 1
            var inside_prev = current
            while scan <= n:
                var following = Int(codes[unsafe_offset=scan]) if scan < n else O
                if not strict_inside(inside_prev, following, scheme):
                    break
                inside_prev = following
                scan += 1
            var following = Int(codes[unsafe_offset=scan]) if scan < n else O
            if strict_end(inside_prev, following, scheme):
                write_entity(types, starts, ends, count, current, pos, scan - 1)
                count += 1
            pos = scan
        else:
            pos += 1
        prev = Int(codes[unsafe_offset=pos - 1])
    return count


@export("mse_extract_default")
def mse_extract_default(
    codes_addr: Int,
    n: Int,
    types_addr: Int,
    starts_addr: Int,
    ends_addr: Int,
) abi("C") -> Int:
    return extract_default(
        ptr(codes_addr), n, ptr(types_addr), ptr(starts_addr), ptr(ends_addr)
    )


@export("mse_extract_strict")
def mse_extract_strict(
    codes_addr: Int,
    n: Int,
    scheme: Int,
    types_addr: Int,
    starts_addr: Int,
    ends_addr: Int,
) abi("C") -> Int:
    return extract_strict(
        ptr(codes_addr), n, scheme, ptr(types_addr), ptr(starts_addr), ptr(ends_addr)
    )


@export("mse_extract_pair_default")
def mse_extract_pair_default(
    true_codes_addr: Int,
    n_true: Int,
    true_types_addr: Int,
    true_starts_addr: Int,
    true_ends_addr: Int,
    pred_codes_addr: Int,
    n_pred: Int,
    pred_types_addr: Int,
    pred_starts_addr: Int,
    pred_ends_addr: Int,
    counts_addr: Int,
) abi("C"):
    var true_codes = ptr(true_codes_addr)
    var true_types = ptr(true_types_addr)
    var true_starts = ptr(true_starts_addr)
    var true_ends = ptr(true_ends_addr)
    var pred_codes = ptr(pred_codes_addr)
    var pred_types = ptr(pred_types_addr)
    var pred_starts = ptr(pred_starts_addr)
    var pred_ends = ptr(pred_ends_addr)
    var counts = ptr(counts_addr)

    counts[unsafe_offset=0] = Int64(
        extract_default(true_codes, n_true, true_types, true_starts, true_ends)
    )
    counts[unsafe_offset=1] = Int64(
        extract_default(pred_codes, n_pred, pred_types, pred_starts, pred_ends)
    )


@export("mse_extract_pair_strict")
def mse_extract_pair_strict(
    true_codes_addr: Int,
    n_true: Int,
    pred_codes_addr: Int,
    n_pred: Int,
    scheme: Int,
    true_types_addr: Int,
    true_starts_addr: Int,
    true_ends_addr: Int,
    pred_types_addr: Int,
    pred_starts_addr: Int,
    pred_ends_addr: Int,
    counts_addr: Int,
) abi("C"):
    var true_codes = ptr(true_codes_addr)
    var true_types = ptr(true_types_addr)
    var true_starts = ptr(true_starts_addr)
    var true_ends = ptr(true_ends_addr)
    var pred_codes = ptr(pred_codes_addr)
    var pred_types = ptr(pred_types_addr)
    var pred_starts = ptr(pred_starts_addr)
    var pred_ends = ptr(pred_ends_addr)
    var counts = ptr(counts_addr)

    counts[unsafe_offset=0] = Int64(
        extract_strict(true_codes, n_true, scheme, true_types, true_starts, true_ends)
    )
    counts[unsafe_offset=1] = Int64(
        extract_strict(pred_codes, n_pred, scheme, pred_types, pred_starts, pred_ends)
    )


def entity_less(
    a_start: Int, a_end: Int, a_type: Int, b_start: Int, b_end: Int, b_type: Int
) -> Bool:
    if a_start != b_start:
        return a_start < b_start
    if a_end != b_end:
        return a_end < b_end
    return a_type < b_type


@export("mse_count_entities")
def mse_count_entities(
    true_types_addr: Int,
    true_starts_addr: Int,
    true_ends_addr: Int,
    n_true: Int,
    pred_types_addr: Int,
    pred_starts_addr: Int,
    pred_ends_addr: Int,
    n_pred: Int,
    n_types: Int,
    true_counts_addr: Int,
    pred_counts_addr: Int,
    tp_counts_addr: Int,
) abi("C"):
    var true_types = ptr(true_types_addr)
    var true_starts = ptr(true_starts_addr)
    var true_ends = ptr(true_ends_addr)
    var pred_types = ptr(pred_types_addr)
    var pred_starts = ptr(pred_starts_addr)
    var pred_ends = ptr(pred_ends_addr)
    var true_counts = ptr(true_counts_addr)
    var pred_counts = ptr(pred_counts_addr)
    var tp_counts = ptr(tp_counts_addr)
    for t in range(n_types):
        true_counts[unsafe_offset=t] = 0
        pred_counts[unsafe_offset=t] = 0
        tp_counts[unsafe_offset=t] = 0
    for idx in range(n_true):
        true_counts[unsafe_offset=Int(true_types[unsafe_offset=idx])] += 1
    for idx in range(n_pred):
        pred_counts[unsafe_offset=Int(pred_types[unsafe_offset=idx])] += 1
    var ti = 0
    var pi = 0
    while ti < n_true and pi < n_pred:
        var tt = Int(true_types[unsafe_offset=ti])
        var ts = Int(true_starts[unsafe_offset=ti])
        var te = Int(true_ends[unsafe_offset=ti])
        var pt = Int(pred_types[unsafe_offset=pi])
        var ps = Int(pred_starts[unsafe_offset=pi])
        var pe = Int(pred_ends[unsafe_offset=pi])
        if tt == pt and ts == ps and te == pe:
            tp_counts[unsafe_offset=tt] += 1
            ti += 1
            pi += 1
        elif entity_less(ts, te, tt, ps, pe, pt):
            ti += 1
        else:
            pi += 1


@export("mse_token_counts")
def mse_token_counts(
    true_addr: Int,
    pred_addr: Int,
    n: Int,
    result_addr: Int,
) abi("C"):
    var truth = ptr(true_addr)
    var pred = ptr(pred_addr)
    var result = ptr(result_addr)
    var correct = 0
    var tp = 0
    var fp = 0
    var false_neg = 0
    var tn = 0
    comptime W = simd_width_of[DType.float64]()
    var correct_vec = SIMD[DType.int64, W](0)
    var tp_vec = SIMD[DType.int64, W](0)
    var fp_vec = SIMD[DType.int64, W](0)
    var false_neg_vec = SIMD[DType.int64, W](0)
    var tn_vec = SIMD[DType.int64, W](0)
    var idx = 0
    while idx + W <= n:
        var a = truth.unsafe_load[width=W](idx)
        var b = pred.unsafe_load[width=W](idx)
        var equal = a.eq(b)
        var a_zero = a.eq(0)
        var b_zero = b.eq(0)
        correct_vec += equal.cast[DType.int64]()
        tp_vec += (equal & ~(a_zero & b_zero)).cast[DType.int64]()
        fp_vec += (~equal & ~b_zero).cast[DType.int64]()
        false_neg_vec += (~a_zero & b_zero).cast[DType.int64]()
        tn_vec += (a_zero & b_zero).cast[DType.int64]()
        idx += W
    correct += Int(correct_vec.reduce_add())
    tp += Int(tp_vec.reduce_add())
    fp += Int(fp_vec.reduce_add())
    false_neg += Int(false_neg_vec.reduce_add())
    tn += Int(tn_vec.reduce_add())
    while idx < n:
        var a = Int(truth[unsafe_offset=idx])
        var b = Int(pred[unsafe_offset=idx])
        if a == b:
            correct += 1
        if a == b and (a != 0 or b != 0):
            tp += 1
        if a != b and b != 0:
            fp += 1
        if a != 0 and b == 0:
            false_neg += 1
        if a == 0 and b == 0:
            tn += 1
        idx += 1
    result[unsafe_offset=0] = Int64(correct)
    result[unsafe_offset=1] = Int64(tp)
    result[unsafe_offset=2] = Int64(fp)
    result[unsafe_offset=3] = Int64(false_neg)
    result[unsafe_offset=4] = Int64(tn)
