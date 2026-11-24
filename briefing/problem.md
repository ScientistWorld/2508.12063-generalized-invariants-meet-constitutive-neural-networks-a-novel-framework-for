# Problem Definition

## Research Question

Given stress-stretch measurements of an isotropic, incompressible hyperelastic material in a few
loading modes, discover a strain energy function that (1) reproduces those measurements, (2)
predicts the material's response in a loading mode it was never fitted to, and (3) stays a
compact, closed-form, physically admissible constitutive law that an engineer can read.

Classical models (neo-Hookean, Mooney-Rivlin, Ogden, Gent, ...) fix the invariants and the
functional form in advance; generic regressors fit curves but give no constitutive law. The task
is automated discovery of the law itself.

## Data

Real experimental data, the curves used by the paper (its authors' own data files):

| material | source | training modes (fit) | held-out mode (predict) |
|---|---|---|---|
| rubber_T20 | Treloar (1944), 20 C | uniaxial tension, equibiaxial tension | pure shear |
| rubber_T50 | Treloar (1944), 50 C | uniaxial tension, equibiaxial tension | pure shear |
| brain_CX | Budday et al. (2017), cortex | uniaxial tension, uniaxial compression | simple shear |
| brain_CC | Budday et al. (2017), corpus callosum | uniaxial tension, uniaxial compression | simple shear |
| brain_CR | Budday et al. (2017), corona radiata | uniaxial tension, uniaxial compression | simple shear |
| brain_BG | Budday et al. (2017), basal ganglia | uniaxial tension, uniaxial compression | simple shear |

Rubber stretches reach 5-8 (stresses in MPa); brain tissue is tested to 10% stretch and 0.2 shear
(stresses in kPa) and is soft, nonlinear and stiffer in compression than in tension.

`data/curves.json` (loaded by `data/load_data.py`) holds every training-mode curve, and the
held-out curve of the development materials (rubber_T20, brain_CX, brain_CC). For the test
materials (rubber_T50, brain_CR, brain_BG) only the stretches / shears at which the held-out
mode is evaluated are given; their held-out stresses are the hidden test.

## Exponent families

The task is posed for three families of laws built on generalized invariants
`J_a = l1^a + l2^a + l3^a` (principal stretches l_i, l1 l2 l3 = 1): exponents a > 0 only
("positive", I1-like; a = 2 gives I1), a < 0 only ("negative", I2-like; a = -2 gives I2), or
either sign ("combined"). Each material is studied in all three.

## Why It Matters

Hyperelastic laws drive simulations of tyres, seals, soft robots, medical devices and brain
mechanics. A law that only fits the tests it was calibrated on fails in the multiaxial states that
matter in practice; one that generalizes across loading modes, and stays interpretable, is what
model discovery is for.

## Success Criteria

1. High accuracy on the held-out loading mode of materials whose held-out mode was never seen.
2. Faithful fit of the training modes.
3. Small relative stress errors, including at small strains.
4. A compact law: few active terms, every term a closed-form function of a generalized invariant.
5. Physical admissibility: non-negative energy contributions, a stress-free reference state, a
   well-defined energy wherever the law is evaluated.
