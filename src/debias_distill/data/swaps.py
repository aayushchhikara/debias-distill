"""Word-pair substitution lists and substitution.

Substitution works at two levels:

- `counterfactual(text, table)` rewrites plain text (used for inspection and tests).
- `TokenSwapper` rewrites token-id sequences in place, replacing each listed word
  with a word of exactly the same token length, so the substituted sequence has
  the same length as the original. A sequence is rejected (no variant produced)
  when any listed word in it cannot be replaced this way, when an ambiguous
  pronoun has no following context, or when it contains a blocked word, so
  variants are never partially substituted.

List files in the swap directory:
  *.tsv          bidirectional male<TAB>female pairs (lowercase; applied to lowercase
                 and Capitalized forms)
  one_way.tsv    source<TAB>target substitutions with no reverse direction; a
                 Capitalized source matches only that exact form
  blockers.txt   words whose presence means the text is legitimately sex-specific
                 (e.g. pregnancy); such sequences get no variant
  corpus_names.tsv  male<TAB>female first-name pairs used only for substituting the
                 training corpus (never in evaluation templates), exact case

A table can also carry reject words: exact-case words (held-out or unpaired first
names) whose presence removes the whole sequence from training.

Names match only in their exact (Capitalized) form. Other casings (e.g. ALL CAPS)
are reported as unhandled. Singular "they" and other gender-neutral words are
deliberately left untouched.
"""

import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WORD_RE = re.compile(r"[A-Za-z]+")
ONE_WAY_FILE = "one_way.tsv"
CORPUS_NAMES_FILE = "corpus_names.tsv"
BLOCKERS_FILE = "blockers.txt"

# Words after "her" that indicate the object reading ("saw her and ...") -> "him".
# Anything else alphabetic is read as possessive ("her book", "her family") -> "his".
_OBJECT_FOLLOWERS = frozenset(
    """
    a an the this that these those some any every each no all
    and or but nor so yet because if when while as than though although until unless
    to of in on at by for with from into onto about after before over under around
    through during against between without within upon across behind beyond near
    up down out off away back again too also even just still then there here
    very really now today tonight tomorrow yesterday always never often once twice well
    is was are were be been being has had have will would can could should may might must
    i you he she it we they me him us them my your our their its who whom which what
    not
    deeply dearly closely completely carefully nicely quickly gently directly personally
    immediately properly badly softly tightly warmly firmly slowly seriously regularly
    constantly repeatedly recently finally eventually especially
    """.split()  # noqa: SIM905
)

# Followers where both readings are common ("her back hurt" / "brought her back"); logged.
AMBIGUOUS_FOLLOWERS = frozenset(
    {"back", "home", "up", "down", "out", "off", "over", "in", "on", "very", "more"}
)


def _is_letter(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


# What follows a word, as seen by the resolvers: a word, BOUNDARY (sentence
# punctuation, dashes, end of text), MONEY (a currency sign) or OTHER (digits,
# symbols, opening quotes...).
BOUNDARY = None
OTHER = "#"
MONEY = "$"
_BOUNDARY_CHARS = frozenset(".,;:!?)]}\n\r-–—…")
_MONEY_CHARS = frozenset("$£€")
_QUOTES = frozenset("\"'“”‘’")


def next_context(text: str, pos: int) -> str | None:
    """Classify what follows position `pos` in `text`: a word, BOUNDARY, MONEY or OTHER."""
    i = pos
    while i < len(text) and text[i] in " \t":
        i += 1
    if i >= len(text):
        return BOUNDARY
    c = text[i]
    if _is_letter(c):
        return WORD_RE.match(text, i).group()
    if c in _BOUNDARY_CHARS:
        return BOUNDARY
    if c in _MONEY_CHARS:
        return MONEY
    if c in _QUOTES:  # adjacent quote closes ("her."), spaced quote opens ('her "best"')
        return OTHER if i > pos else BOUNDARY
    return OTHER


def _is_word(nxt: str | None) -> bool:
    return nxt not in (BOUNDARY, OTHER, MONEY)


def _resolve_her(nxt: str | None) -> str:
    if nxt is BOUNDARY or nxt == MONEY:  # "saw her.", "paid her $500"
        return "him"
    if nxt == OTHER:  # "her 2 children", "her (new) job"
        return "his"
    return "him" if nxt.lower() in _OBJECT_FOLLOWERS else "his"


def _resolve_his(nxt: str | None) -> str:
    return "hers" if nxt is BOUNDARY else "her"


def _resolve_miss(nxt: str | None) -> str | None:
    # Only the title before a name ("Miss Smith"); the verb ("Miss you") is unhandled.
    return "Mr" if _is_word(nxt) and nxt[0].isupper() else None


_RESOLVERS: dict[str, Callable[[str | None], str]] = {"her": _resolve_her, "his": _resolve_his}
# Exact-case words whose substitute depends on the next word; None means unhandled.
_CONDITIONAL: dict[str, Callable[[str | None], str | None]] = {"Miss": _resolve_miss}


@dataclass
class SwapTable:
    """Substitution table: bidirectional pairs, one-way substitutions, names and blockers."""

    words: dict[str, str] = field(default_factory=dict)  # lowercase -> lowercase
    ambiguous: frozenset[str] = frozenset()  # lowercase words resolved by context
    exact: dict[str, str] = field(default_factory=dict)  # exact case -> exact case (names, titles)
    conditional: frozenset[str] = frozenset()  # exact-case words resolved by context
    blockers: frozenset[str] = frozenset()  # lowercase
    rejects: dict[str, str] = field(default_factory=dict)  # exact-case word -> reason

    def is_listed(self, word: str) -> bool:
        """True if `word` is one this table is responsible for substituting."""
        if word in self.exact or word in self.conditional:
            return True
        low = word.lower()
        return low in self.words or low in self.ambiguous

    def source_forms(self) -> set[str]:
        """Every exact-case word form the table can match."""
        forms = set(self.exact) | set(self.conditional)
        for w in (*self.words, *self.ambiguous):
            forms.update({w, w.capitalize()})
        return forms

    def reject_reason(self, text: str) -> str | None:
        """The reason of the first reject word in `text`, if any."""
        if not self.rejects:
            return None
        for m in WORD_RE.finditer(text):
            if m.group() in self.rejects:
                return self.rejects[m.group()]
        return None

    def blocked_word(self, text: str) -> str | None:
        """The first blocker word in `text`, if any."""
        if not self.blockers:
            return None
        for m in WORD_RE.finditer(text):
            if m.group().lower() in self.blockers:
                return m.group().lower()
        return None


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            a, b = line.split("\t")
            pairs.append((a, b))
    return pairs


def build_table(
    word_pairs: list[tuple[str, str]],
    name_pairs: list[tuple[str, str]] | None = None,
    one_way: list[tuple[str, str]] | None = None,
    blockers: list[str] | None = None,
    rejects: dict[str, str] | None = None,
) -> SwapTable:
    """Build a table. A word with several bidirectional targets must have a resolver."""
    targets: dict[str, set[str]] = {}
    for a, b in word_pairs:
        a, b = a.lower(), b.lower()
        targets.setdefault(a, set()).add(b)
        targets.setdefault(b, set()).add(a)

    words, ambiguous = {}, set()
    for w, ts in targets.items():
        if len(ts) == 1:
            words[w] = next(iter(ts))
        elif w in _RESOLVERS:
            ambiguous.add(w)
        else:
            raise ValueError(f"'{w}' maps to several words {sorted(ts)} and has no resolver")

    exact: dict[str, str] = {}

    def add_exact(src: str, dst: str) -> None:
        if src in exact and exact[src] != dst:
            raise ValueError(f"'{src}' is paired twice")
        exact[src] = dst

    for a, b in name_pairs or []:
        add_exact(a, b)
        add_exact(b, a)
    for src, dst in one_way or []:
        if src.lower() in targets or src.lower() in words:
            raise ValueError(f"one-way source '{src}' is already a bidirectional word")
        if src[0].isupper():
            add_exact(src, dst)
        else:
            words[src] = dst
    rejects = dict(rejects or {})
    if clash := sorted(set(rejects) & set(exact)):
        raise ValueError(f"words are both substituted and rejected: {clash}")
    return SwapTable(
        words=words,
        ambiguous=frozenset(ambiguous),
        exact=exact,
        conditional=frozenset(_CONDITIONAL),
        blockers=frozenset(w.lower() for w in blockers or []),
        rejects=rejects,
    )


def load_swap_pairs(
    swap_dir: str | Path = "data/swaps",
    name_pairs: list[tuple[str, str]] | None = None,
    rejects: dict[str, str] | None = None,
) -> SwapTable:
    """Load the list files in `swap_dir` (see module docstring).

    `name_pairs` are added to the corpus-only pairs from `corpus_names.tsv`.
    """
    swap_dir = Path(swap_dir)
    pairs: list[tuple[str, str]] = []
    for path in sorted(swap_dir.glob("*.tsv")):
        if path.name not in (ONE_WAY_FILE, CORPUS_NAMES_FILE):
            pairs.extend(_read_pairs(path))
    corpus_names_path = swap_dir / CORPUS_NAMES_FILE
    all_names = list(name_pairs or [])
    if corpus_names_path.exists():
        all_names.extend(_read_pairs(corpus_names_path))
    one_way_path = swap_dir / ONE_WAY_FILE
    one_way = _read_pairs(one_way_path) if one_way_path.exists() else []
    blockers_path = swap_dir / BLOCKERS_FILE
    blockers = []
    if blockers_path.exists():
        for line in blockers_path.read_text().splitlines():
            if not line.lstrip().startswith("#"):
                blockers.extend(line.split())
    return build_table(pairs, all_names, one_way, blockers, rejects)


def swap_word(word: str, next_word: str | None, table: SwapTable) -> str | None:
    """Return the substitute for `word` (same casing), or None if it is not handled."""
    if word in table.exact:
        return table.exact[word]
    if word in table.conditional:
        return _CONDITIONAL[word](next_word)
    low = word.lower()
    if low in table.ambiguous:
        repl = _RESOLVERS[low](next_word)
    elif low in table.words:
        repl = table.words[low]
    else:
        return None
    if word == low:
        return repl
    if word == low.capitalize():
        return repl.capitalize()
    return None  # ALL CAPS / mixed case


def counterfactual(text: str, table: SwapTable) -> tuple[str, list[str]]:
    """Return (substituted_text, unhandled_words). Blockers and rejects are not applied here."""
    out, unhandled, last = [], [], 0
    for m in WORD_RE.finditer(text):
        word = m.group()
        if not table.is_listed(word):
            continue
        repl = swap_word(word, next_context(text, m.end()), table)
        if repl is None:
            unhandled.append(word)
            continue
        out.append(text[last : m.start()])
        out.append(repl)
        last = m.end()
    out.append(text[last:])
    return "".join(out), unhandled


def count_listed(text: str, table: SwapTable) -> int:
    return sum(table.is_listed(m.group()) for m in WORD_RE.finditer(text))


@dataclass
class SwapStats:
    sequences: int = 0
    with_listed_words: int = 0
    produced: int = 0
    rejected: Counter = field(default_factory=Counter)  # reason -> count
    swaps: Counter = field(default_factory=Counter)  # "src->dst" -> count
    ambiguous_contexts: Counter = field(default_factory=Counter)  # "her <next>" -> count

    def to_dict(self) -> dict:
        return {
            "sequences": self.sequences,
            "with_listed_words": self.with_listed_words,
            "produced": self.produced,
            "rejected": dict(self.rejected.most_common()),
            "swaps": dict(self.swaps.most_common()),
            "ambiguous_contexts": dict(self.ambiguous_contexts.most_common(50)),
        }


class TokenSwapper:
    """Length-preserving substitution on token-id sequences."""

    def __init__(self, tokenizer, table: SwapTable):
        self.tokenizer = tokenizer
        self.table = table
        self.text = [tokenizer.decode([i]) for i in range(len(tokenizer))]
        # first token id -> [(token ids of the form, word, has leading space)], longest first.
        # Byte-level BPE splits on spaces first, so a word's tokens do not depend on context
        # (except when glued to neighbouring letters, which the boundary checks exclude).
        phrases: dict[int, list[tuple[tuple[int, ...], str, bool]]] = defaultdict(list)
        for word in table.source_forms():
            for space in (True, False):
                ids = tuple(self._encode((" " if space else "") + word))
                phrases[ids[0]].append((ids, word, space))
        self._phrases = {k: sorted(v, key=lambda p: -len(p[0])) for k, v in phrases.items()}
        self._first_ids = np.fromiter(self._phrases, dtype=np.int64)
        self._encoded: dict[str, tuple[int, ...]] = {}
        self.stats = SwapStats()
        self.last_status = ""  # "unlisted", "rejected" or "produced" for the last swap() call

    def _encode(self, s: str) -> list[int]:
        return self.tokenizer(s, add_special_tokens=False)["input_ids"]

    def _encode_cached(self, s: str) -> tuple[int, ...]:
        if s not in self._encoded:
            self._encoded[s] = tuple(self._encode(s))
        return self._encoded[s]

    def _match(self, ids: np.ndarray, pos: int) -> tuple[tuple[int, ...], str, bool] | None:
        """The listed word starting at `pos` as a whole word, if any."""
        for phrase, word, space in self._phrases[int(ids[pos])]:
            end = pos + len(phrase)
            if end > len(ids) or tuple(int(x) for x in ids[pos:end]) != phrase:
                continue
            if end < len(ids) and _is_letter(self.text[ids[end]][:1]):
                continue  # glued to the following letters: part of a longer word
            if not space and (pos == 0 or _is_letter(self.text[ids[pos - 1]][-1:])):
                continue  # glued to the preceding letters, or no left context
            return phrase, word, space
        return None

    def _reject(self, reason: str) -> None:
        self.stats.rejected[reason] += 1

    def swap(self, ids: np.ndarray) -> np.ndarray | None:
        """Return the substituted copy of `ids`, or None if there is nothing to swap or it is rejected.

        `last_status` tells the two None cases apart.
        """
        self.stats.sequences += 1
        self.last_status = "rejected"  # until proven otherwise
        decoded = self.tokenizer.decode(ids)
        if reason := self.table.reject_reason(decoded):
            self._reject(reason)
            return None
        listed_in_text = count_listed(decoded, self.table)
        if not listed_in_text:
            self.last_status = "unlisted"
            return None
        self.stats.with_listed_words += 1
        if blocked := self.table.blocked_word(decoded):
            self._reject(f"blocked:{blocked}")
            return None

        out = ids.copy()
        n_swapped, next_free = 0, 0
        pending: Counter = Counter()
        for pos in np.flatnonzero(np.isin(ids, self._first_ids)):
            if pos < next_free:
                continue
            match = self._match(ids, pos)
            if match is None:
                continue
            phrase, word, space = match
            end = pos + len(phrase)
            is_contextual = word.lower() in self.table.ambiguous or word in self.table.conditional
            if is_contextual and end >= len(ids):
                self._reject(f"no_context_at_row_end:{word}")
                return None
            nxt = next_context("".join(self.text[i] for i in ids[end : end + 3]), 0)
            repl = swap_word(word, nxt, self.table)
            if repl is None:
                self._reject(f"unhandled:{word}")
                return None
            new = self._encode_cached((" " if space else "") + repl)
            if len(new) != len(phrase):
                self._reject(f"length_mismatch:{word}->{repl}")
                return None
            out[pos:end] = new
            next_free = end
            n_swapped += 1
            pending[f"{word}->{repl}"] += 1
            if word.lower() == "her" and _is_word(nxt) and nxt.lower() in AMBIGUOUS_FOLLOWERS:
                self.stats.ambiguous_contexts[f"her {nxt.lower()} -> {repl}"] += 1

        if n_swapped != listed_in_text:
            # A listed word was not found as a whole-word token run (e.g. cut at the row edge).
            self._reject("not_matched_in_tokens")
            return None
        self.stats.produced += 1
        self.stats.swaps.update(pending)
        self.last_status = "produced"
        return out
