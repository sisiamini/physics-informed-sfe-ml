"""Streamlit dashboard for the final stacking-fault-energy prediction workflow."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from sfe_workflow import (
    DOMAIN_RANGES,
    ELEMENTS,
    InputValidationError,
    PredictionWorkflowError,
    parse_composition,
    run_prediction,
)


st.set_page_config(
    page_title="SFE Predictor",
    page_icon="⚙️",
    layout="wide",
)

st.title("Stacking Fault Energy Predictor")
st.caption(
    "Predict experimental SFE from alloy composition using the frozen final "
    "XGBoost workflow and an applicability-domain check evaluated with nested "
    "exact-composition-grouped cross-validation."
)

with st.expander("How this app works", expanded=False):
    st.markdown(
        """
1. Blank Fe is treated as **balance Fe**. Other blank fields are treated as **0 wt.%**.
2. The total composition must equal **100 wt.%** and individual values must lie between 0 and 100 wt.%.
3. `SFE_with_Descriptors.py` calculates `R_mean (pm)`, `R_std (pm)`, `Δχ`, and `VEC_mean`.
4. `SFE_Master.py` calculates the **eight** retained published `SFE_*` outputs.
5. The final full-data XGBoost production pipeline predicts SFE from the **28 predictors**.
6. The composition is checked against the operational applicability rule: **visible composition ranges AND kNN dense-composition region**.
7. Applicability-domain evidence is from nested grouped validation and is **not external validation**.
        """
    )

st.subheader("Composition input")
st.write(
    "Leave Fe blank to use balance Fe. Leave any other field blank to use 0 wt.%. "
    "The placeholders show the operational visible-domain ranges. Values outside "
    "them are not blocked, but the prediction will be flagged as outside the domain."
)


def clear_inputs() -> None:
    for element_name in ELEMENTS:
        st.session_state[f"composition_{element_name}"] = ""


raw_values: dict[str, str] = {}
cols = st.columns(4)

for i, element in enumerate(ELEMENTS):
    low, high = DOMAIN_RANGES[element]
    with cols[i % 4]:
        raw_values[element] = st.text_input(
            label=f"{element} (wt.%)",
            placeholder=f"AD: {low:g}–{high:g}",
            key=f"composition_{element}",
            help=(
                f"Operational visible-domain range: {low:g}–{high:g} wt.%. "
                "The kNN dense-core criterion must also be satisfied."
            ),
        )

try:
    current_composition = parse_composition(raw_values)
    current_total = sum(current_composition.values())

    if str(raw_values.get("Fe", "")).strip() == "":
        st.info(
            f"Fe balance: **{current_composition['Fe']:.6f} wt.%**  \n"
            f"Current total: **{current_total:.6f} wt.%**"
        )
    else:
        st.info(f"Current total: **{current_total:.6f} wt.%**")

except InputValidationError as exc:
    st.warning(str(exc))

left, middle, right = st.columns([1, 1, 4])
with left:
    predict_clicked = st.button("Predict SFE", type="primary", use_container_width=True)
with middle:
    st.button("Clear inputs", use_container_width=True, on_click=clear_inputs)

if predict_clicked:
    try:
        composition = parse_composition(raw_values)
        result = run_prediction(composition)

    except InputValidationError as exc:
        st.error(str(exc))

    except PredictionWorkflowError as exc:
        st.error(
            "Prediction stopped because validation, feature generation, applicability "
            "assessment, or model prediction failed."
        )
        st.code(str(exc), language="text")

    except Exception as exc:
        st.error("Prediction stopped because an unexpected error occurred.")
        st.code(str(exc), language="text")

    else:
        st.success("Prediction completed successfully.")

        metric_col, domain_col = st.columns([1, 2])
        with metric_col:
            st.metric("Predicted SFE", f"{result.prediction:.3f} mJ/m²")

        with domain_col:
            if result.domain.inside_applicability_domain:
                st.success(
                    "Applicability-domain status: **INSIDE**  \n"
                    "The composition satisfies both the visible-range and kNN dense-core criteria."
                )
            else:
                st.warning(
                    "Applicability-domain status: **OUTSIDE**  \n"
                    "The prediction is still calculated, but it should be interpreted with greater caution."
                )

        if str(raw_values.get("Fe", "")).strip() == "":
            st.caption(
                f"Fe balance used: {composition['Fe']:.6f} wt.% | "
                f"Validated total composition: {result.total:.6f} wt.%"
            )
        else:
            st.caption(f"Validated total composition: {result.total:.6f} wt.%")

        st.subheader("Applicability-domain assessment")
        domain_table = pd.DataFrame([
            {
                "Criterion": "Visible composition ranges",
                "Satisfied": result.domain.inside_visible_ranges,
            },
            {
                "Criterion": "kNN dense-composition region",
                "Satisfied": result.domain.inside_knn_dense_core,
            },
            {
                "Criterion": "Final applicability domain",
                "Satisfied": result.domain.inside_applicability_domain,
            },
        ])
        st.dataframe(domain_table, use_container_width=True, hide_index=True)
        st.caption(
            f"kNN distance = {result.domain.knn_distance:.4f}; "
            f"training-fitted threshold = {result.domain.knn_threshold:.4f}."
        )

        if result.domain.visible_range_violations:
            with st.expander("Visible-range checks not satisfied"):
                for violation in result.domain.visible_range_violations:
                    st.write(f"- {violation}")

        summary = result.domain_summary
        if summary:
            reliability = summary.get("nested_inside_domain_reliability_percent")
            coverage = summary.get("nested_domain_coverage_percent")
            inside_rmse = summary.get("nested_inside_domain_RMSE")
            outside_rmse = summary.get("nested_outside_domain_RMSE")

            if reliability is not None and coverage is not None:
                st.info(
                    "Nested exact-composition-grouped validation: "
                    f"{reliability:.1f}% of predictions classified inside the domain "
                    "had absolute percentage error ≤20%, with "
                    f"{coverage:.1f}% validation coverage. This is internal out-of-sample "
                    "evidence, not an external-validation guarantee."
                )

            if inside_rmse is not None and outside_rmse is not None:
                st.caption(
                    "Nested grouped-validation RMSE: "
                    f"{inside_rmse:.2f} mJ/m² inside the domain versus "
                    f"{outside_rmse:.2f} mJ/m² outside."
                )

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Calculated descriptors")
            st.dataframe(
                pd.DataFrame([result.descriptors]).T.rename(columns={0: "value"}),
                use_container_width=True,
            )

        with col_b:
            st.subheader("Calculated published SFE features")
            st.dataframe(
                pd.DataFrame([result.sfe_features]).T.rename(columns={0: "value"}),
                use_container_width=True,
            )

        with st.expander("Exact 28-feature model input row"):
            st.dataframe(result.model_input, use_container_width=True)

        with st.expander("Script output logs"):
            st.markdown("**SFE_with_Descriptors.py**")
            st.code(result.descriptor_script_output or "No output", language="text")
            st.markdown("**SFE_Master.py**")
            st.code(result.sfe_master_script_output or "No output", language="text")

st.divider()
st.caption(
    "The applicability-domain check is a nested cross-validated internal assessment "
    "of the represented composition space. It does not replace independent external validation."
)
