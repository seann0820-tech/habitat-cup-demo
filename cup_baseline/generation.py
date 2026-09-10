"""Geometry-only episode generation helpers (no policy or model access)."""
import numpy as np

GENERATION_VERSION = 'passage_v2'


class EpisodeGenerationError(RuntimeError):
    pass


def passage_candidates(points, rng, legacy=False):
    """Keep the old eight proposals first; broaden position and lateral offset.

    The parcel size and robot clearance are fixed. Candidate order depends only
    on the episode RNG. A proposal is not an accepted evaluation episode.
    """
    points = np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 1e-8:
        return []
    if legacy:
        pairs = [(float(f), 0.) for f in rng.permutation(np.linspace(.3, .65, 8))]
    else:
        fractions = rng.permutation(np.linspace(.18, .82, 13))
        pairs = [(float(f), 0.) for f in fractions]
        sides = [(float(f), float(s)) for f in fractions for s in (-.32, -.18, .18, .32)]
        pairs += [sides[i] for i in rng.permutation(len(sides))[:19]]
    cumulative = np.cumsum(lengths)
    result = []
    for fraction, offset in pairs:
        distance = total * fraction
        k = min(int(np.searchsorted(cumulative, distance)), len(lengths)-1)
        prior = cumulative[k-1] if k else 0.
        direction = points[k+1] - points[k]
        p = points[k] + direction * ((distance-prior)/max(lengths[k], 1e-8))
        sideways = np.array([-direction[2], 0., direction[0]])
        sideways /= max(np.linalg.norm(sideways), 1e-8)
        result.append((p + offset*sideways, fraction, offset))
    return result
