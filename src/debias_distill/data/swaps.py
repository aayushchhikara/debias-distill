"""Word-pair substitution lists and substitution.

`counterfactual(text, table)` rewrites plain text: each listed word is replaced
by its counterpart, with context-dependent words ("her" -> "him"/"his", the title
"Miss") resolved from the word that follows.

List files in the swap directory:
  *.tsv          bidirectional male<TAB>female pairs (lowercase; applied to lowercase
                 and Capitalized forms)
  one_way.tsv    source<TAB>target substitutions with no reverse direction; a
                 Capitalized source matches only that exact form
  blockers.txt   words whose presence means the text is legitimately sex-specific
                 (e.g. pregnancy); such sequences get no variant

Names match only in their exact (Capitalized) form. Other casings (e.g. ALL CAPS)
are reported as unhandled. Singular "they" and other gender-neutral words are
deliberately left untouched.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-z]+")
ONE_WAY_FILE = "one_way.tsv"
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
    return SwapTable(
        words=words,
        ambiguous=frozenset(ambiguous),
        exact=exact,
        conditional=frozenset(_CONDITIONAL),
        blockers=frozenset(w.lower() for w in blockers or []),
    )


def load_swap_pairs(
    swap_dir: str | Path = "data/swaps", name_pairs: list[tuple[str, str]] | None = None
) -> SwapTable:
    """Load the list files in `swap_dir` (see module docstring), plus optional name pairs."""
    swap_dir = Path(swap_dir)
    pairs: list[tuple[str, str]] = []
    for path in sorted(swap_dir.glob("*.tsv")):
        if path.name != ONE_WAY_FILE:
            pairs.extend(_read_pairs(path))
    one_way_path = swap_dir / ONE_WAY_FILE
    one_way = _read_pairs(one_way_path) if one_way_path.exists() else []
    blockers_path = swap_dir / BLOCKERS_FILE
    blockers = []
    if blockers_path.exists():
        for line in blockers_path.read_text().splitlines():
            if not line.lstrip().startswith("#"):
                blockers.extend(line.split())
    return build_table(pairs, name_pairs, one_way, blockers)


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
    """Return (substituted_text, unhandled_words). Blockers are not applied here."""
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
