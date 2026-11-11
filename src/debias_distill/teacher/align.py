"""Map a next-token distribution between a text and its gender-swapped copy.

A row and its copy differ only in gendered words, so the teacher's predictions for
the two differ in two ways: in what it expects of gender-neutral continuations
(the bias we want to remove) and in which gendered word it expects next (after
"She said that" it expects "she"; after "He said that", "he"). Averaging the two
distributions directly would also average the second kind, and teach the student
that "he" and "she" are interchangeable regardless of context.

`align` moves the probability a distribution puts on each gendered token to that
token's counterpart, so the copy's distribution is expressed in the original's
terms before the two are averaged (and the other way round for the copy's target).

Only words whose source and target forms are single tokens are mapped. A source
with several targets ("her" -> "him" / "his") has its mass split in proportion to
the destination distribution's own mass on those targets.
"""

from dataclasses import dataclass

import torch

from debias_distill.data.swaps import _RESOLVERS, BOUNDARY, MONEY, OTHER, SwapTable


@dataclass
class TokenMap:
    """Token-level edges source -> target. Edges sharing a source form one group."""

    src: torch.Tensor  # [E] source token id of each edge
    dst: torch.Tensor  # [E] target token id of each edge
    group: torch.Tensor  # [E] index of the edge's source among the unique sources
    sources: torch.Tensor  # [G] unique source token ids
    group_size: torch.Tensor  # [G] number of edges per source

    def to(self, device) -> "TokenMap":
        return TokenMap(
            *(t.to(device) for t in (self.src, self.dst, self.group, self.sources, self.group_size))
        )

    @classmethod
    def from_edges(cls, edges: list[tuple[int, int]]) -> "TokenMap":
        edges = sorted(set(edges))
        sources = sorted({s for s, _ in edges})
        index = {s: i for i, s in enumerate(sources)}
        group = torch.tensor([index[s] for s, _ in edges], dtype=torch.long)
        return cls(
            src=torch.tensor([s for s, _ in edges], dtype=torch.long),
            dst=torch.tensor([t for _, t in edges], dtype=torch.long),
            group=group,
            sources=torch.tensor(sources, dtype=torch.long),
            group_size=torch.bincount(group, minlength=len(sources)),
        )


def _targets(word: str, table: SwapTable) -> list[str]:
    """Every word `word` can be swapped to, in the same casing."""
    if word in table.exact:
        return [table.exact[word]]
    low = word.lower()
    if low in table.ambiguous:
        resolver = _RESOLVERS[low]
        found = {resolver(context) for context in (BOUNDARY, OTHER, MONEY, "the", "zzz")}
    elif low in table.words:
        found = {table.words[low]}
    else:
        return []
    return sorted(t if word == low else t.capitalize() for t in found)


def build_token_map(tokenizer, table: SwapTable) -> tuple[TokenMap, list[str]]:
    """Token edges for every single-token word form in `table`, and the forms left unmapped.

    Words resolved by the following word in running text (the title "Miss") are not
    mapped: as a predicted token their reading is unknown.
    """

    def encode(s: str) -> list[int]:
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    edges: list[tuple[int, int]] = []
    unmapped: list[str] = []
    for word in sorted(table.source_forms() - table.conditional):
        targets = _targets(word, table)
        for prefix in (" ", ""):
            source_ids = encode(prefix + word)
            target_ids = [encode(prefix + t) for t in targets]
            if len(source_ids) == 1 and target_ids and all(len(t) == 1 for t in target_ids):
                edges.extend((source_ids[0], t[0]) for t in target_ids)
            else:
                unmapped.append(prefix + word)
    return TokenMap.from_edges(edges), unmapped


def align(p_src: torch.Tensor, p_dst: torch.Tensor, token_map: TokenMap) -> torch.Tensor:
    """Express `p_src` in the frame of `p_dst` by moving mass along the token edges.

    Both are probability tensors [..., vocab]. The result sums to the same total as `p_src`.
    """
    moved = p_src[..., token_map.src]  # [..., E] mass leaving each source
    weight = p_dst[..., token_map.dst]  # [..., E] destination's own mass on each target
    group_total = torch.zeros(
        *weight.shape[:-1], len(token_map.sources), device=weight.device, dtype=weight.dtype
    )
    group_total.index_add_(-1, token_map.group, weight)
    total = group_total[..., token_map.group]
    even = 1.0 / token_map.group_size[token_map.group].to(weight.dtype)
    share = torch.where(total > 0, weight / total.clamp_min(1e-30), even)
    out = p_src.clone()
    out[..., token_map.sources] = 0
    out.index_add_(-1, token_map.dst, moved * share)
    return out
