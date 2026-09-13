# Finite-Y Tevatron boundary study

This is a standalone check of the 24 additional Tevatron rows at
approximately `0.20 < qT/Q < 0.30`. It does not alter the nominal main result.

The tested transition is

```text
Y_unitary = p(qT/Q) * (FO_NLO - W)
matched   = W + Y_unitary = (1-p) * W + p * FO_NLO
```

`p` is the smooth C2 profile: zero in the TMD core and one above the selected
transition window.

## Run the validation

```bash
python finite-y/scripts/verify_finite_y.py
```

The script validates the profile algebra, the 24-row scope, NLO-node
convergence/positivity decision, the 96-endpoint x 50-replica propagation,
and the fact that this study was not promoted to replace the main result.

## Meaning of the result

This is a validated unitary transition for the stated Tevatron boundary scope.
It is not a validation of conventional additive `FO - ASY` matching, a
universal collider result, or an LHCb finite-Y fit.
