# Paper's Method

## Key Contribution

GI-CANN (generalized-invariant constitutive artificial neural network) learns the invariant
itself. Instead of fixing I1 or I2, it uses generalized invariants J_alpha = l1^alpha + l2^alpha
+ l3^alpha with the exponent alpha as a trainable network parameter, and learns it together with
the strain energy function in one training run (no two-step "find alpha, then fit psi").

## Architecture (Eq. 13)

For every exponent alpha_k in the set S:

    psi = sum_k  wx2_k1 * w11_k * (J_k - 3)
               + wx2_k2 * [exp(w21_k (J_k - 3)) - 1]
               + wx2_k3 * [-ln(1 - w31_k (J_k - 3))]

All weights are non-negative. alpha = 2 gives J = I1 and alpha = -2 gives J = I2, so the
SI-CANN (S = {2}, {-2} or {2, -2}) is the special case with fixed exponents. Stresses follow from
psi in closed form for uniaxial tension/compression, equibiaxial tension, pure shear and simple
shear (Eqs. 15-18).

## Training (as in the authors' notebooks, LivingMatterLab/CANN/GICANN)

- Loss (Eq. 14): sum over training modes of the mean squared stress error, plus L1 with r = 0.01
  on every weight.
- Adam, learning rate 1e-3, mini-batches of 12, 30,000 epochs (40,000 for the brain
  two-invariant model), early stopping with patience 3,000 on the training loss, best epoch
  restored.
- Exponent starts: rubber 2.0 (two invariants: 1.7 and -0.1), brain 5.0 in magnitude (two
  invariants: 2 and -2).
- Families: positive (one alpha > 0), negative (one alpha < 0), combined (one of each).

## Reproduction in this gym (method/gicann.py, method/train.py)

Pure-Python implementation of the above with analytic gradients. Two additions make the reference
reach the law the paper reports instead of a random local stop:
- after the mini-batch phase, full-batch Adam with a decaying learning rate finishes minimizing the
  same objective (the objective is flat along an (alpha, weight) valley and early stopping halts
  at random places along it);
- each law is the best of 20 runs from random initializations, selected by the training objective
  (on the paper's cortex data only some starts escape a local optimum near alpha = -15).
On the paper's cortex data this gives alpha* = -18.11 with one identity term, w = 0.0092 kPa
(paper: -18.016, 0.009 kPa); on its rubber data alpha* = 1.676 with w_id = 0.231 MPa and
exponential weights 0.109, 0.282 MPa (paper Eq. 19: 1.675, 0.227, 0.107, 0.297).

## Main Claims

1. A learned exponent beats the fixed classical invariant: rubber avg R2 0.998 vs 0.982 (SI-CANN),
   brain 0.966 vs 0.610.
2. Rubber prefers positive (I1-like) exponents, alpha* ~ 1.7; negative exponents fail (0.264).
3. Brain tissue prefers a single large negative exponent, alpha* ~ -18; positive ones fail (0.526).
4. Combining both signs gives the best or an equally good fit (rubber 0.999, brain 0.966).
5. The discovered laws are sparse and interpretable (one or two active terms), close to dense
   PS-CANN or FNN accuracy.
