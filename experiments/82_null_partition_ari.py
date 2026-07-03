"""
Exp 82: Null-partition ARI reference table for the clustering paper.

Independently verifies the external review's (docs/external_reviews/
2026-07-03_external_review_clustering_and_floor.md, Section 2.2) claim that
the observed HDBSCAN ARI ceiling (~0.484-0.510, exp62-67) is the arithmetic
fingerprint of the partition {EW+KPZ+Eden}, {BD}, {KS}, {RD} against the
paper's evaluation labels (n=480; 6 systems x 80; classes EW | KPZ-class |
KS | RD).

Verified 2026-07-03 (this script's output, matching the review to 3 dp):
  1.000  perfect class recovery
  0.726  BD splits off, rest correct
  0.569  perfect simulator separation
  0.498  {EW+KPZ+Eden},{BD},{KS},{RD}   <-- the observed ceiling
  0.452  {EW+KPZ},{BD},{Eden},{KS},{RD}
  0.220  {EW+KPZ+Eden+KS},{BD},{RD}

Destination: clustering paper v2, Section IV/V ("the ceiling is a
fingerprint, not a wall"), per the review's recommendation.
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari

SYS = ['EW', 'KPZ', 'BD', 'Eden', 'KS', 'RD']
N_PER = 80
CLS = {'EW': 0, 'KPZ': 1, 'BD': 1, 'Eden': 1, 'KS': 2, 'RD': 3}

sysid = np.repeat(np.arange(6), N_PER)
labels = np.array([CLS[SYS[s]] for s in sysid])

def partition(groups):
    m = {s: gi for gi, g in enumerate(groups) for s in g}
    return np.array([m[SYS[s]] for s in sysid])

TESTS = {
    'perfect class recovery': [['EW'], ['KPZ', 'BD', 'Eden'], ['KS'], ['RD']],
    'BD splits off, rest correct': [['EW'], ['KPZ', 'Eden'], ['BD'], ['KS'], ['RD']],
    'perfect simulator separation': [[s] for s in SYS],
    '{EW+KPZ+Eden},{BD},{KS},{RD}': [['EW', 'KPZ', 'Eden'], ['BD'], ['KS'], ['RD']],
    '{EW+KPZ},{BD},{Eden},{KS},{RD}': [['EW', 'KPZ'], ['BD'], ['Eden'], ['KS'], ['RD']],
    '{EW+KPZ+Eden+KS},{BD},{RD}': [['EW', 'KPZ', 'Eden', 'KS'], ['BD'], ['RD']],
}

if __name__ == '__main__':
    for name, groups in TESTS.items():
        print(f"{ari(labels, partition(groups)):.3f}  {name}")
