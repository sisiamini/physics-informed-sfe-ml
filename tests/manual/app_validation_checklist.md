# Final Streamlit App Validation Checklist

The following application-level checks were manually completed for the final Streamlit workflow.

| Check | Status | Evidence |
|---|---|---|
| Known dataset composition returns the expected production-model prediction | PASS | `docs/app/screenshots/known_composition_prediction.png` |
| Blank Fe is calculated as balance Fe | PASS | `docs/app/screenshots/fe_balance.png` |
| Composition satisfying visible ranges and kNN dense-core criterion is labelled `INSIDE` | PASS | `docs/app/screenshots/inside_domain.png` |
| Composition outside the visible operational ranges is labelled `OUTSIDE` | PASS | `docs/app/screenshots/outside_visible_range.png` |
| Composition inside visible ranges but outside the kNN dense core is labelled `OUTSIDE` | PASS | `docs/app/screenshots/outside_knn_dense_core.png` |
| Total composition different from 100 wt.% is rejected | PASS | `docs/app/screenshots/invalid_total.png` |
| Negative composition input is rejected | PASS | `docs/app/screenshots/invalid_input.png` |
| Non-numeric composition input is rejected | PASS | `docs/app/screenshots/invalid_input.png` |
| Four composition-derived descriptors are displayed | PASS | confirmed in final application |
| Eight retained published SFE outputs are displayed | PASS | confirmed in final application |
| Exact 28-feature model input is available for inspection | PASS | confirmed in final application |
| Operational nitrogen range is displayed as 0–0.48 wt.% | PASS | confirmed in final application |

## Final schema

The application must expose the following final scientific representation:

```text
16 composition variables
4 composition-derived descriptors
8 retained published SFE-model outputs
28 total predictors
```

## Final applicability rule

A prediction is labelled `INSIDE` only when:

```text
inside visible composition ranges
AND
inside kNN dense-composition region
```

A syntactically and arithmetically valid composition outside the applicability domain may still receive a numerical prediction, but it must be labelled `OUTSIDE` and presented with greater caution.

## Final validation status

All checks listed above passed in the final working application before repository freeze.
