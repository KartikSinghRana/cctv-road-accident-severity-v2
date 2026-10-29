import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model_inference import (
    decode_uploaded_image,
    load_trained_model,
    predict_with_explanations,
)


PROJECT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = PROJECT_DIR / "artifacts" / "r_evaluation"
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
CHECKPOINT_PATH = PROJECT_DIR / "artifacts" / "best_model.pth"
CLASS_COLORS = {
    "No_Accident": "#38bdf8",
    "Minor_Accident": "#fbbf24",
    "Severe_Accident": "#fb7185",
}
PLOT_TEMPLATE = "plotly_dark"

st.set_page_config(
    page_title="RoadSafe | Model Insights",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the trained severity model...")
def get_trained_model(
    checkpoint_path: str,
    checkpoint_mtime: int,
    inference_code_mtime: int,
):
    del checkpoint_mtime, inference_code_mtime
    return load_trained_model(checkpoint_path)


st.markdown(
    """
    <style>
    :root { --ink: #e8eef8; --muted: #9aa9bd; --panel: #111a2b; --line: #22304a; --aqua: #62e6c5; }
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .stApp { background: radial-gradient(ellipse at 75% -15%, #19314b 0, #0b1220 45%, #0b1220 100%); color: var(--ink); }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif !important; letter-spacing: -0.035em; }
    h1 { font-size: 2.45rem !important; }
    [data-testid="stSidebar"] { background: #0e1726; border-right: 1px solid var(--line); }
    [data-testid="stMetric"] { background: linear-gradient(145deg, #151f32, #101827); border: 1px solid var(--line); padding: 18px 20px; border-radius: 16px; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { font-family: 'Segoe UI', sans-serif; }
    [data-testid="stTabs"] button { font-weight: 600; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    .hero-kicker { color: var(--aqua); font-size: .78rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .hero-copy { color: var(--muted); max-width: 760px; font-size: 1.02rem; }
    .section-note { color: var(--muted); font-size: .9rem; }
    .pill { display: inline-block; border: 1px solid #285a59; background: #102b30; color: #85f1d6; border-radius: 999px; padding: 4px 10px; font-size: .78rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_evaluation(directory: str) -> dict[str, object]:
    root = Path(directory)
    required = {
        "summary": "summary_metrics.csv",
        "classes": "class_metrics.csv",
        "confusion": "confusion_matrix.csv",
        "errors": "error_analysis.csv",
    }
    missing = [filename for filename in required.values() if not (root / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing evaluation output(s): "
            + ", ".join(missing)
            + ". Run `Rscript evaluate_model.R` from the project folder first."
        )

    summary_frame = pd.read_csv(root / required["summary"])
    if not {"metric", "value"}.issubset(summary_frame.columns):
        raise ValueError("summary_metrics.csv must contain metric and value columns.")
    if summary_frame["metric"].duplicated().any():
        raise ValueError("summary_metrics.csv contains duplicate metric names.")

    class_metrics = pd.read_csv(root / required["classes"])
    errors = pd.read_csv(root / required["errors"])
    confusion = pd.read_csv(root / required["confusion"], index_col=0)

    required_class_columns = {
        "class", "support", "precision", "recall", "specificity", "f1", "auc_ovr",
    }
    required_error_columns = {
        "relative_path", "group_id", "true", "pred", "correct",
        "predicted_confidence", "true_class_probability", "top_two_margin", "error_type",
    }
    for label, frame, columns in (
        ("class_metrics.csv", class_metrics, required_class_columns),
        ("error_analysis.csv", errors, required_error_columns),
    ):
        absent = columns.difference(frame.columns)
        if absent:
            raise ValueError(f"{label} is missing required columns: {', '.join(sorted(absent))}.")
    if errors.empty:
        raise ValueError("error_analysis.csv contains no prediction rows.")

    errors["correct"] = errors["correct"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if errors["correct"].isna().any():
        raise ValueError("error_analysis.csv has invalid values in the correct column.")
    for column in ("predicted_confidence", "true_class_probability", "top_two_margin"):
        errors[column] = pd.to_numeric(errors[column], errors="coerce")
        if errors[column].isna().any():
            raise ValueError(f"error_analysis.csv contains invalid {column} values.")

    group_path = root / "group_error_analysis.csv"
    groups = pd.read_csv(group_path) if group_path.is_file() else pd.DataFrame()
    return {
        "summary": dict(zip(summary_frame["metric"], summary_frame["value"])),
        "classes": class_metrics,
        "confusion": confusion,
        "errors": errors,
        "groups": groups,
    }


try:
    data = load_evaluation(str(EVALUATION_DIR))
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load evaluation results: {error}")
    st.stop()

summary = data["summary"]
class_metrics = data["classes"]
confusion = data["confusion"]
all_errors = data["errors"]
group_analysis = data["groups"]
classes = class_metrics["class"].tolist()

st.markdown('<div class="hero-kicker">CNN evaluation · R-powered insights</div>', unsafe_allow_html=True)
st.title("Road accident severity")
st.markdown(
    '<p class="hero-copy">A closer look at how the classifier performs on held-out CCTV images — '
    'from overall scores to the individual predictions it gets wrong.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="pill">HELD-OUT TEST SET</span> &nbsp; '
    '<span class="section-note">Metrics are loaded from the reproducible R evaluation outputs.</span>',
    unsafe_allow_html=True,
)

accuracy = float(summary["accuracy"])
accuracy_lower = float(summary["accuracy_bootstrap_95ci_lower"])
accuracy_upper = float(summary["accuracy_bootstrap_95ci_upper"])
macro_f1 = float(summary["macro_f1"])
f1_lower = float(summary["macro_f1_bootstrap_95ci_lower"])
f1_upper = float(summary["macro_f1_bootstrap_95ci_upper"])

with st.container(horizontal=True):
    st.metric("Test images", f"{int(float(summary['test_samples'])):,}", border=True)
    st.metric(
        "Accuracy",
        f"{accuracy:.1%}",
        help=f"Stratified image-bootstrap 95% CI: {accuracy_lower:.1%}–{accuracy_upper:.1%}",
        border=True,
    )
    st.metric(
        "Macro F1",
        f"{macro_f1:.3f}",
        help=f"Stratified image-bootstrap 95% CI: {f1_lower:.3f}–{f1_upper:.3f}",
        border=True,
    )
    st.metric("Weighted F1", f"{float(summary['weighted_f1']):.3f}", border=True)
    st.metric("Macro ROC AUC", f"{float(summary['macro_auc_ovr']):.3f}", border=True)

with st.sidebar:
    st.markdown("## Explore results")
    st.caption("Filter predictions across the test set.")
    selected_actual = st.multiselect("Actual class", classes, default=classes)
    selected_predicted = st.multiselect("Predicted class", classes, default=classes)
    error_filter = st.selectbox(
        "Prediction outcome",
        ["All predictions", "Errors only", "High-confidence errors", "Correct predictions"],
    )
    confidence_floor = st.slider(
        "Minimum predicted confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        format="%.2f",
    )
    path_search = st.text_input("Find image path", placeholder="e.g. 10400.jpg")
    st.divider()
    st.caption("Data source")
    st.code("artifacts/r_evaluation/", language=None)

filtered = all_errors[
    all_errors["true"].isin(selected_actual)
    & all_errors["pred"].isin(selected_predicted)
    & (all_errors["predicted_confidence"] >= confidence_floor)
].copy()
if error_filter == "Errors only":
    filtered = filtered[~filtered["correct"]]
elif error_filter == "High-confidence errors":
    filtered = filtered[filtered["error_type"] == "High-confidence error"]
elif error_filter == "Correct predictions":
    filtered = filtered[filtered["correct"]]
if path_search.strip():
    filtered = filtered[
        filtered["relative_path"].str.contains(path_search.strip(), case=False, na=False)
    ]

predict_tab, overview_tab, errors_tab, groups_tab, report_tab = st.tabs(
    ["Predict & explain", "Overview", "Error explorer", "Group analysis", "R report"]
)

with predict_tab:
    st.subheader("Try a road image")
    st.caption(
        "Upload a road-scene image to run the trained classifier and generate "
        "class-specific Grad-CAM and gradient-saliency visualizations."
    )
    uploader_column, settings_column = st.columns([1.2, 0.8], gap="large")
    with uploader_column:
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=False,
            max_upload_size=10,
            help="Maximum file size: 10 MB. The model accepts a single road-scene image.",
        )
    with settings_column:
        explanation_target = st.selectbox(
            "Explain which class?",
            ["Predicted class", *classes],
            help=(
                "Grad-CAM explains the model's score for the selected class. "
                "The prediction itself is unchanged."
            ),
        )
        run_prediction = st.button(
            "Analyze image",
            type="primary",
            icon=":material/analytics:",
            disabled=uploaded_image is None,
            width="stretch",
        )

    upload_bytes = uploaded_image.getvalue() if uploaded_image is not None else None
    if upload_bytes is not None:
        upload_fingerprint = hashlib.sha256(upload_bytes).hexdigest()
        stored_result = st.session_state.get("upload_prediction")
        if stored_result is not None and stored_result["fingerprint"] != upload_fingerprint:
            st.session_state.pop("upload_prediction", None)
            stored_result = None

        if run_prediction:
            st.session_state.pop("upload_prediction", None)
            stored_result = None
            try:
                uploaded_rgb = decode_uploaded_image(upload_bytes)
                checkpoint_mtime = CHECKPOINT_PATH.stat().st_mtime_ns
                inference_code_mtime = Path(__file__).with_name(
                    "model_inference.py"
                ).stat().st_mtime_ns
                loaded_model = get_trained_model(
                    str(CHECKPOINT_PATH),
                    checkpoint_mtime,
                    inference_code_mtime,
                )
                explanation_class = (
                    None if explanation_target == "Predicted class" else explanation_target
                )
                result = predict_with_explanations(
                    loaded_model,
                    uploaded_rgb,
                    target_class=explanation_class,
                )
                st.session_state["upload_prediction"] = {
                    "fingerprint": upload_fingerprint,
                    "result": result,
                }
                stored_result = st.session_state["upload_prediction"]
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                st.error(f"Could not analyze this image: {error}")

        if stored_result is not None:
            result = stored_result["result"]
            prediction_column, probability_column = st.columns([1, 1], gap="large")
            with prediction_column:
                with st.container(border=True):
                    st.caption("Model prediction")
                    st.markdown(f"### {result['predicted_class'].replace('_', ' ')}")
                    st.caption(f"Model confidence: {result['confidence']:.1%}")
                st.caption(
                    f"Input resized to {result['image_size']} × {result['image_size']} pixels · "
                    f"device: {result['device']}"
                )
            with probability_column:
                probability_frame = pd.DataFrame(
                    {
                        "Class": [name.replace("_", " ") for name in classes],
                        "Model score": [result["probabilities"][name] for name in classes],
                    }
                )
                probability_chart = px.bar(
                    probability_frame,
                    x="Model score",
                    y="Class",
                    orientation="h",
                    range_x=[0, 1],
                    color="Class",
                    color_discrete_map={
                        name.replace("_", " "): color
                        for name, color in CLASS_COLORS.items()
                    },
                    template=PLOT_TEMPLATE,
                )
                probability_chart.update_layout(
                    height=180,
                    showlegend=False,
                    xaxis_title="Model score",
                    yaxis_title=None,
                    margin={"l": 10, "r": 10, "t": 5, "b": 5},
                )
                st.plotly_chart(
                    probability_chart,
                    width="stretch",
                    alt="Class scores for the uploaded road image",
                )

            st.markdown("#### Visual explanations")
            st.caption(
                f"Grad-CAM target: **{result['target_class'].replace('_', ' ')}**. "
                "Brighter regions indicate stronger contribution to the selected score; "
                "these maps are approximate explanations, not proof of model reasoning."
            )
            image_columns = st.columns(3, gap="medium")
            image_columns[0].image(
                result["display_image"],
                caption="Uploaded image",
                alt="Uploaded road-scene image",
                width="stretch",
            )
            image_columns[1].image(
                result["gradcam_overlay"],
                caption="Grad-CAM · selected class",
                alt="Grad-CAM heatmap overlay highlighting image regions relevant to the selected class",
                width="stretch",
            )
            image_columns[2].image(
                result["saliency_overlay"],
                caption="Input-gradient saliency",
                alt="Gradient saliency overlay highlighting pixels that affect the selected class score",
                width="stretch",
            )
            st.info(
                "This EfficientNetV2-S is a convolutional network and has no native "
                "self-attention layers. The input-gradient view is an attention-style "
                "saliency map, not a true attention map. Both visualizations are "
                "diagnostic only; do not use this model as a real-world safety system."
            )
            gradcam_buffer = BytesIO()
            result["gradcam_overlay"].save(gradcam_buffer, format="PNG")
            st.download_button(
                "Download Grad-CAM image",
                data=gradcam_buffer.getvalue(),
                file_name="road_severity_gradcam.png",
                mime="image/png",
            )
        elif not CHECKPOINT_PATH.is_file():
            st.warning(
                "The trained checkpoint is missing. Run the training notebook so "
                "`artifacts/best_model.pth` is available before using image prediction."
            )
        else:
            st.caption("Upload a supported image, choose an explanation target, and select Analyze image.")

with overview_tab:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Where predictions land")
        confusion_plot = go.Figure(
            data=go.Heatmap(
                z=confusion.to_numpy(),
                x=confusion.columns,
                y=confusion.index,
                colorscale=[[0, "#111a2b"], [0.55, "#176b79"], [1, "#62e6c5"]],
                text=confusion.to_numpy(),
                texttemplate="%{text}",
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Images: %{z}<extra></extra>",
                colorbar={"title": "Images"},
            )
        )
        confusion_plot.update_layout(
            template=PLOT_TEMPLATE,
            height=410,
            xaxis_title="Predicted class",
            yaxis_title="Actual class",
            margin={"l": 10, "r": 10, "t": 20, "b": 10},
        )
        st.plotly_chart(
            confusion_plot,
            width="stretch",
            alt="Confusion matrix showing actual and predicted accident severity classes",
        )
    with right:
        st.subheader("Class-by-class performance")
        class_long = class_metrics.melt(
            id_vars="class",
            value_vars=["precision", "recall", "f1", "auc_ovr"],
            var_name="Metric",
            value_name="Score",
        )
        class_long["Metric"] = class_long["Metric"].replace(
            {"auc_ovr": "ROC AUC", "f1": "F1"}
        )
        class_plot = px.bar(
            class_long,
            x="class",
            y="Score",
            color="Metric",
            barmode="group",
            range_y=[0, 1],
            color_discrete_sequence=["#62e6c5", "#38bdf8", "#fbbf24", "#c4b5fd"],
            template=PLOT_TEMPLATE,
        )
        class_plot.update_layout(
            height=410,
            xaxis_title=None,
            yaxis_title="Score",
            legend_title=None,
            margin={"l": 10, "r": 10, "t": 20, "b": 10},
        )
        st.plotly_chart(
            class_plot,
            width="stretch",
            alt="Precision, recall, F1, and ROC AUC compared across classes",
        )

    st.subheader("Per-class metrics")
    display_metrics = class_metrics.rename(
        columns={
            "class": "Class",
            "support": "Test support",
            "precision": "Precision",
            "recall": "Recall",
            "specificity": "Specificity",
            "f1": "F1",
            "auc_ovr": "ROC AUC",
        }
    ).copy()
    st.dataframe(
        display_metrics.style.format(
            {
                "Test support": "{:,.0f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "Specificity": "{:.3f}",
                "F1": "{:.3f}",
                "ROC AUC": "{:.3f}",
            }
        ).background_gradient(subset=["F1", "ROC AUC"], cmap="GnBu", vmin=0, vmax=1),
        hide_index=True,
        width="stretch",
        alt="Per-class test-set performance metrics",
    )
with errors_tab:
    st.subheader("Find patterns in mistakes")
    error_count = int((~all_errors["correct"]).sum())
    high_confidence_count = int((all_errors["error_type"] == "High-confidence error").sum())
    e1, e2, e3 = st.columns(3)
    e1.metric("Misclassified images", f"{error_count:,}")
    e2.metric("High-confidence errors", f"{high_confidence_count:,}")
    e3.metric("Rows matching filters", f"{len(filtered):,}")

    error_pairs = (
        all_errors.loc[~all_errors["correct"]]
        .groupby(["true", "pred"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )
    if not error_pairs.empty:
        error_pairs["Confusion"] = error_pairs["true"] + " → " + error_pairs["pred"]
        pair_plot = px.bar(
            error_pairs.head(10),
            x="Confusion",
            y="size",
            color="true",
            color_discrete_map=CLASS_COLORS,
            labels={"size": "Misclassified images", "true": "Actual class"},
            template=PLOT_TEMPLATE,
        )
        pair_plot.update_layout(
            height=330,
            showlegend=False,
            xaxis_title=None,
            margin={"l": 10, "r": 10, "t": 15, "b": 10},
        )
        st.plotly_chart(
            pair_plot,
            width="stretch",
            alt="Most common actual-to-predicted class errors",
        )

    st.markdown(f"#### Matching predictions · {len(filtered):,} rows")
    st.dataframe(
        filtered[
            [
                "relative_path", "group_id", "true", "pred", "error_type",
                "predicted_confidence", "true_class_probability", "top_two_margin",
            ]
        ].style.format(
            {
                "predicted_confidence": "{:.3f}",
                "true_class_probability": "{:.3f}",
                "top_two_margin": "{:.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
        height=420,
        alt="Filtered test predictions with classes, confidence, and error details",
    )
    st.download_button(
        "Download filtered predictions as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_error_analysis.csv",
        mime="text/csv",
    )

    with st.expander("Preview source images for the most confident errors"):
        preview_errors = all_errors.loc[~all_errors["correct"]].nlargest(
            6, "predicted_confidence"
        )
        if not RAW_DATA_DIR.is_dir():
            st.info("Source images are not present under data/raw, so previews are unavailable.")
        elif preview_errors.empty:
            st.info("No misclassified images are available to preview.")
        else:
            preview_columns = st.columns(3)
            for index, (_, row) in enumerate(preview_errors.iterrows()):
                image_path = (RAW_DATA_DIR / str(row["relative_path"])).resolve()
                try:
                    image_path.relative_to(RAW_DATA_DIR.resolve())
                except ValueError:
                    st.warning(f"Skipped unsafe image path: {row['relative_path']}")
                    continue
                if image_path.is_file():
                    with preview_columns[index % 3]:
                        st.image(
                            str(image_path),
                            width="stretch",
                            alt=f"Test image predicted {row['pred']} instead of {row['true']}",
                        )
                        st.caption(
                            f"Actual: {row['true']} · Predicted: {row['pred']} · "
                            f"Confidence: {row['predicted_confidence']:.1%}"
                        )

with groups_tab:
    st.subheader("Where do errors cluster?")
    st.warning(
        "Group IDs are consecutive-filename blocks used as a split proxy. They are not "
        "verified source videos, cameras, lighting, or weather conditions."
    )
    if group_analysis.empty or "group_id" not in group_analysis:
        st.info("No group-level error summary is available. Re-run the R evaluator.")
    else:
        top_groups = group_analysis.sort_values(
            ["error_rate", "errors"], ascending=False
        ).head(20)
        group_plot = px.bar(
            top_groups,
            x="group_id",
            y="error_rate",
            color="error_rate",
            color_continuous_scale=["#38bdf8", "#fbbf24", "#fb7185"],
            hover_data=["samples", "errors", "most_common_error"],
            labels={
                "group_id": "Split-group proxy",
                "error_rate": "Error rate",
                "samples": "Images",
                "errors": "Errors",
            },
            template=PLOT_TEMPLATE,
        )
        group_plot.update_layout(
            height=420,
            yaxis={"range": [0, 1], "tickformat": ".0%"},
            xaxis_title=None,
            coloraxis_showscale=False,
            margin={"l": 10, "r": 10, "t": 20, "b": 10},
        )
        st.plotly_chart(
            group_plot,
            width="stretch",
            alt="Error rates for split-group proxy blocks with the highest error rates",
        )
        st.dataframe(
            group_analysis.sort_values(["error_rate", "errors"], ascending=False),
            hide_index=True,
            width="stretch",
            alt="Group-level error rates and most common class confusions",
        )
        st.download_button(
            "Download group summary as CSV",
            data=group_analysis.to_csv(index=False).encode("utf-8"),
            file_name="group_error_analysis.csv",
            mime="text/csv",
        )

with report_tab:
    st.subheader("Reproducible R evaluation summary")
    st.markdown("#### Summary metrics")
    summary_frame = pd.DataFrame(
        [{"Metric": metric.replace("_", " ").title(), "Value": value} for metric, value in summary.items()]
    )
    st.dataframe(
        summary_frame,
        hide_index=True,
        width="stretch",
        alt="Summary metrics generated by the R evaluation script",
    )
    st.download_button(
        "Download summary metrics",
        data=summary_frame.to_csv(index=False).encode("utf-8"),
        file_name="summary_metrics.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "The confidence intervals quantify image-sample uncertainty for this split. "
    "They do not establish video-independent performance."
)
