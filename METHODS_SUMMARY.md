# Methods Summary

## Frequency-Domain Feature Disentanglement

SMD-Mamba applies a 3D Fourier transform to MRI features. Low-frequency components are used to recover scanner-invariant morphology, while high-frequency components are treated as scanner-specific style/noise.

```text
Z_morph = F^{-1}(A_morph, P)
Z_style = F^{-1}(A_style, P)
```

## Orthogonal Disentanglement Loss

```text
L_ortho = || Z_morph^T Z_style ||_F^2
```

## Total Objective

```text
L_total = L_CE + lambda * L_ortho
```

## Morphology-Guided Mamba Modeling

The purified biological representation `Z_morph` guides the sequence/state updates so the spatial memory learns domain-invariant anatomical information rather than scanner-specific artifacts.
