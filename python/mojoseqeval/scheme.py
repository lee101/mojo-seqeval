"""Tagging-scheme classes compatible with seqeval's public scheme module."""

from __future__ import annotations

import enum
from itertools import chain


class Entity:
    def __init__(self, sent_id: int, start: int, end: int, tag: str):
        self.sent_id = sent_id
        self.start = start
        self.end = end
        self.tag = tag

    def __repr__(self):
        return f"({self.sent_id}, {self.tag}, {self.start}, {self.end})"

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())

    def to_tuple(self):
        return self.sent_id, self.tag, self.start, self.end


class Prefix(enum.Flag):
    I = enum.auto()
    O = enum.auto()
    B = enum.auto()
    E = enum.auto()
    S = enum.auto()
    U = enum.auto()
    L = enum.auto()
    ANY = I | O | B | E | S | U | L


Prefixes = dict(Prefix.__members__)


class Tag(enum.Flag):
    SAME = enum.auto()
    DIFF = enum.auto()
    ANY = SAME | DIFF


class Token:
    scheme_id = 0
    allowed_chars = frozenset()
    allowed_prefix = Prefix(0)

    def __init__(self, token: str, suffix: bool = False, delimiter: str = "-"):
        self.token = token
        self._suffix = suffix
        char = token[-1] if suffix else token[0]
        self.prefix = Prefixes[char]
        tag = token[:-1] if suffix else token[1:]
        self.tag = tag.strip(delimiter) or "_"

    def __repr__(self):
        return self.token

    def is_valid(self):
        if self.prefix not in self.allowed_prefix:
            allowed = str(self.allowed_prefix).replace("Prefix.", "")
            raise ValueError(
                f"Invalid token is found: {self.token}. Allowed prefixes are: {allowed}."
            )
        return True

    def is_start(self, prev):
        return _start(
            (prev.prefix.name, prev.tag),
            (self.prefix.name, self.tag),
            self.__class__,
        )

    def is_inside(self, prev):
        return _inside(
            (prev.prefix.name, prev.tag),
            (self.prefix.name, self.tag),
            self.__class__,
        )

    def is_end(self, prev):
        return _end(
            (prev.prefix.name, prev.tag),
            (self.prefix.name, self.tag),
            self.__class__,
        )


class IOB1(Token):
    scheme_id = 1
    allowed_chars = frozenset("IOB")
    allowed_prefix = Prefix.I | Prefix.O | Prefix.B


class IOE1(Token):
    scheme_id = 2
    allowed_chars = frozenset("IOE")
    allowed_prefix = Prefix.I | Prefix.O | Prefix.E


class IOB2(Token):
    scheme_id = 3
    allowed_chars = frozenset("IOB")
    allowed_prefix = Prefix.I | Prefix.O | Prefix.B


class IOE2(Token):
    scheme_id = 4
    allowed_chars = frozenset("IOE")
    allowed_prefix = Prefix.I | Prefix.O | Prefix.E


class IOBES(Token):
    scheme_id = 5
    allowed_chars = frozenset("IOBES")
    allowed_prefix = Prefix.I | Prefix.O | Prefix.B | Prefix.E | Prefix.S


class BILOU(Token):
    scheme_id = 6
    allowed_chars = frozenset("BILOU")
    allowed_prefix = Prefix.B | Prefix.I | Prefix.L | Prefix.O | Prefix.U


def _parts(token: str, suffix: bool, delimiter: str):
    char = token[-1] if suffix else token[0]
    tag = token[:-1] if suffix else token[1:]
    return char, tag.strip(delimiter) or "_"


def _start(prev, current, scheme):
    pp, pt = prev
    cp, ct = current
    same = pt == ct
    if scheme is IOB1:
        return (
            (pp == "O" and cp == "I")
            or (pp == "I" and cp == "I" and not same)
            or (pp == "B" and cp == "I")
            or (pp == "I" and cp == "B" and same)
            or (pp == "B" and cp == "B" and same)
        )
    if scheme is IOE1:
        return (
            (pp == "O" and cp == "I")
            or (pp == "I" and cp == "I" and not same)
            or (pp == "E" and cp == "I")
            or (pp == "E" and cp == "E" and same)
        )
    if scheme is IOB2:
        return cp == "B"
    if scheme is IOE2:
        return (
            (pp in "OE" and cp in "IE")
            or (pp == "I" and cp in "IE" and not same)
        )
    if scheme is IOBES:
        return cp in "BS"
    return cp in "BU"


def _inside(prev, current, scheme):
    pp, pt = prev
    cp, ct = current
    if pt != ct:
        return False
    if scheme in (IOB1, IOB2):
        return pp in "BI" and cp == "I"
    if scheme in (IOE1, IOE2):
        return pp == "I" and cp in "IE"
    if scheme is IOBES:
        return pp in "BI" and cp in "IE"
    return pp in "BI" and cp in "IL"


def _end(prev, current, scheme):
    pp, pt = prev
    cp, ct = current
    same = pt == ct
    if scheme is IOB1:
        return (
            (pp == "I" and cp == "I" and not same)
            or (pp in "IB" and cp == "O")
            or (pp == "I" and cp == "B")
            or (pp == "B" and cp == "I" and not same)
            or (pp == "B" and cp == "B" and same)
        )
    if scheme is IOE1:
        return (
            (pp == "I" and cp == "I" and not same)
            or (pp == "I" and cp == "O")
            or (pp == "I" and cp == "E" and not same)
            or (pp == "E" and cp in "IE" and same)
        )
    if scheme is IOB2:
        return (
            (pp in "IB" and cp == "O")
            or (pp == "I" and cp == "I" and not same)
            or (pp in "IB" and cp == "B")
            or (pp == "B" and cp == "I" and not same)
        )
    if scheme is IOE2:
        return pp == "E"
    if scheme is IOBES:
        return pp in "SE"
    return pp in "UL"


class Tokens:
    def __init__(self, tokens, scheme, suffix=False, delimiter="-", sent_id=None):
        self.raw_tokens = tokens
        self.scheme = scheme
        self.suffix = suffix
        self.delimiter = delimiter
        self.sent_id = sent_id

    @property
    def entities(self):
        parsed = []
        for value in self.raw_tokens:
            char, tag = _parts(value, self.suffix, self.delimiter)
            if char not in self.scheme.allowed_chars:
                allowed = "|".join(sorted(self.scheme.allowed_chars))
                raise ValueError(
                    f"Invalid token is found: {value}. Allowed prefixes are: {allowed}."
                )
            parsed.append((char, tag))
        extended = parsed + [("O", "_")]
        found = []
        pos = 0
        prev = ("O", "_")
        while pos < len(extended):
            current = extended[pos]
            if _start(prev, current, self.scheme):
                scan = pos + 1
                inside_prev = current
                while scan < len(extended) and _inside(
                    inside_prev, extended[scan], self.scheme
                ):
                    inside_prev = extended[scan]
                    scan += 1
                if scan < len(extended) and _end(
                    inside_prev, extended[scan], self.scheme
                ):
                    found.append(
                        Entity(self.sent_id, pos, scan, current[1])
                    )
                pos = scan
            else:
                pos += 1
            prev = extended[pos - 1]
        return found


class Entities:
    def __init__(self, sequences, scheme, suffix=False, delimiter="-"):
        self.entities = [
            Tokens(seq, scheme, suffix, delimiter, sent_id).entities
            for sent_id, seq in enumerate(sequences)
        ]

    def filter(self, tag_name):
        return {entity for entity in chain(*self.entities) if entity.tag == tag_name}

    @property
    def unique_tags(self):
        return {entity.tag for entity in chain(*self.entities)}


def auto_detect(sequences, suffix=False, delimiter="-"):
    prefixes = set()
    for tokens in sequences:
        for token in tokens:
            char = token[-1] if suffix else token[0]
            try:
                prefixes.add(Prefixes[char])
            except KeyError:
                raise ValueError(f"This scheme is not supported: {token}") from None

    chars = {prefix.name for prefix in prefixes}
    iob2 = [
        set("IOB"), set("IB"), set("BO"), set("B")
    ]
    ioe2 = [
        set("IOE"), set("IE"), set("EO"), set("E")
    ]
    iobes = [
        set("IOBES"), set("IBES"), set("IOBE"), set("OBES"),
        set("IBE"), set("BES"), set("OBE"), set("BE"), set("S"),
    ]
    bilou = [
        set("IOBLU"), set("IBLU"), set("IOBL"), set("OBLU"),
        set("IBL"), set("BLU"), set("OBL"), set("BL"), set("U"),
    ]
    if chars in iob2:
        return IOB2
    if chars in ioe2:
        return IOE2
    if chars in iobes:
        return IOBES
    if chars in bilou:
        return BILOU
    raise ValueError(f"This scheme is not supported: {prefixes}")
