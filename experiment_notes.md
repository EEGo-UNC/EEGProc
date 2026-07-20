# Experimental Notes

- 07/18/2026: 
    - Without class weight sampling the model predicts the same class always.
- 07/19/2026: 
    - AE term >1.0 impedes the model from learning, so it does not predict better than random.
    - Divergence terms are collapsing the model. With added terms, they ovveride deterministic loss terms and destroy learning. Model predicts basically random.
- 07/20/2066:
    - AE + CE classifier is overfitting. reduce model complexity