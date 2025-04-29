# Semi-Supervised Spectral Clustering

This implements semi-supervised spectral clustering by Ding et al.


## Errors

Similarity graph is not necessarily symmetrical - needs adaptation.

Time complexity upper bounds is really O(n^3) for large K (as K approaches N, you can no longer use a sparse graph).

