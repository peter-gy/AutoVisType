import marimo

__generated_with = "0.13.14"
app = marimo.App(width="full", layout_file="layouts/evaluation.grid.json")


@app.cell(hide_code=True)
def _(DIFFICULTIES, FEATURES, MODELS, evaluate_model, tqdm):
    evaluation_configs: list[dict] = []
    for m in MODELS:
        for f in FEATURES:
            for d in DIFFICULTIES + [None]:
                evaluation_configs.append(
                    {
                        "model_id": m,
                        "feature": f,
                        "difficulty": d,
                    }
                )

    evaluations = [
        {
            "config": config,
            "evaluation": evaluate_model(**config),
        }
        for config in tqdm(evaluation_configs, desc="🔬 Running Evaluations")
    ]
    return (evaluations,)


@app.cell(hide_code=True)
def _(alt, construct_metrics_overview_df, evaluations, metric_picker, mo):
    metrics_overview_df = construct_metrics_overview_df(evaluations)

    metric = metric_picker.value
    metrics_overview_chart = (
        alt.Chart(metrics_overview_df)
        .mark_bar()
        .encode(
            x=alt.X("difficulty_level:N", title="Difficulty"),
            y=alt.Y(f"{metric}:Q", title=metric),
            column=alt.Column(
                "model_id:N",
                title="Model",
                header=alt.Header(labelFontSize=9),
            ),
            row=alt.Row("feature:N", title="Predicted Feature"),
            tooltip=[
                alt.Tooltip("model_id:N", title="Model"),
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("difficulty:N", title="Difficulty"),
                alt.Tooltip(f"{metric}:Q", title=metric),
            ],
        )
        .properties(width=150, height=150)
    )
    mo.vstack(
        [
            metric_picker,
            metrics_overview_chart,
        ]
    )
    return (metrics_overview_df,)


@app.cell
def _(metrics_overview_df):
    metrics_overview_df
    return


@app.cell(hide_code=True)
def _(metrics, mo):
    metric_picker = mo.ui.dropdown(metrics, label="Metric", value=metrics[0])
    return (metric_picker,)


@app.cell(hide_code=True)
def _(evaluation, pl):
    metrics = evaluation.metrics_df.select("metric").to_numpy().squeeze()

    def construct_metrics_overview_df(evaluations: list[dict]) -> pl.DataFrame:
        overall_data = []

        for e in evaluations:
            config = e["config"]
            metrics_df = e["evaluation"].metrics_df
            overall_data.append(
                {
                    **config,
                    **{
                        metric: find_metric_value(metric, metrics_df)
                        for metric in metrics
                    },
                }
            )

        return (
            pl.from_dicts(overall_data)
            .with_columns(pl.col("difficulty").replace(pl.lit(None), pl.lit("all")))
            .with_columns(
                difficulty_level=pl.col("difficulty").replace(
                    {
                        "two_easy": 1,
                        "one_hard": 2,
                        "two_hard": 3,
                        "others": 4,
                        "all": 5,
                    }
                )
            )
        )

    def find_metric_value(metric: str, metrics_df: pl.DataFrame) -> float:
        value = metrics_df.filter(
            pl.col("metric").str.to_lowercase().str.contains(metric)
            | (pl.col("metric").str.to_lowercase() == metric.lower())
        ).select("value")

        if value.is_empty():
            raise ValueError(f"Metric '{metric}' not found in the DataFrame.")

        # Will be coerced to scalar if it contains a single value
        return value.to_numpy().squeeze().tolist()

    return construct_metrics_overview_df, metrics


@app.cell(hide_code=True)
def _(DIFFICULTIES, FEATURES, MODELS, mo):
    model_picker = mo.ui.dropdown(MODELS, label="Model", value=MODELS[0])
    feature_picker = mo.ui.dropdown(FEATURES, label="Feature", value=FEATURES[0])
    difficulty_picker = mo.ui.dropdown(
        DIFFICULTIES,
        label="Difficulty",
        value=DIFFICULTIES[0],
        allow_select_none=True,
    )
    mo.hstack([model_picker, feature_picker, difficulty_picker])
    return difficulty_picker, feature_picker, model_picker


@app.cell(hide_code=True)
def _(evaluation):
    evaluation.preview_df
    return


@app.cell(hide_code=True)
def _(evaluation, mo):
    mo.md(
        f"""
    ## Classification Report

    ```
    {evaluation.classification_report}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(evaluation, feature, mo, model_id):
    mo.vstack(
        [
            mo.md(
                f"## Confusion Matrices: `{model_id}` Model Performance on `{feature}` Prediction"
            ),
            evaluation.confusion_matrices_fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(evaluation, mo, pl):
    def metric_card(metric: str, value: float):
        return mo.Html(rf"""
        <div style="
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            width: 200px;
            display: inline-block;
            text-align: center;
        ">
            <div style="
                font-size: 14px;
                color: #555;
                margin-bottom: 8px;
                font-weight: bold;
            ">{metric}</div>
            <div style="
                font-size: 24px;
                color: #2c3e50;
                font-weight: bold;
            ">{value:.3f}</div>
        </div>
        """)

    metric_cards_data = evaluation.metrics_df.filter(
        ~pl.col("metric")
        .str.to_lowercase()
        .str.contains_any(["precision", "recall", "f1"])
    ).to_dicts()
    cards = [metric_card(**data) for data in metric_cards_data]
    mo.vstack([mo.md("## Metrics"), mo.hstack(cards[:2]), mo.hstack(cards[2:])])
    return


@app.cell(hide_code=True)
def _(difficulty_picker, evaluations, feature_picker, model_picker):
    # Access UI element values
    model_id = model_picker.value
    feature = feature_picker.value
    difficulty = difficulty_picker.value

    # Look up pre-computed evaluation based on UI config
    evaluation = [
        e
        for e in evaluations
        if all(
            [
                e["config"]["model_id"] == model_id,
                e["config"]["feature"] == feature,
                e["config"]["difficulty"] == difficulty,
            ]
        )
    ][0]["evaluation"]
    return evaluation, feature, model_id


@app.cell(hide_code=True)
def _(
    binarize_labels,
    compute_metrics_df,
    construct_eval_df,
    enumerate_unique_labels,
    go,
    np,
    pl,
    plot_multilabel_cm_with_annotations,
    skm,
    ty,
):
    class Evaluation(ty.NamedTuple):
        preview_df: pl.DataFrame
        classification_report: str
        confusion_matrices: list[np.ndarray]
        confusion_matrices_fig: go.Figure
        metrics_df: pl.DataFrame

    def evaluate_model(
        model_id: str,
        feature: str,
        difficulty: str | None,
    ) -> Evaluation:
        full_eval_df = construct_eval_df(model_id)
        unique_labels = enumerate_unique_labels(full_eval_df, feature)

        eval_df = (
            full_eval_df
            if difficulty is None
            else full_eval_df.filter(difficulty=difficulty)
        )
        preview_df = eval_df.select("url", feature, f"{feature}_true")
        true_labels_binarized, predicted_labels_binarized, mlb = binarize_labels(
            eval_df,
            unique_labels,
            feature,
        )

        classification_report = skm.classification_report(
            true_labels_binarized,
            predicted_labels_binarized,
            target_names=mlb.classes_,
            zero_division=0,
        )

        confusion_matrices = skm.multilabel_confusion_matrix(
            true_labels_binarized,
            predicted_labels_binarized,
        )
        confusion_matrices_fig = plot_multilabel_cm_with_annotations(
            confusion_matrices,
            mlb.classes_,
            figure_title="",
        )

        metrics_df = compute_metrics_df(
            true_labels_binarized,
            predicted_labels_binarized,
        )

        return Evaluation(
            preview_df=preview_df,
            classification_report=classification_report,
            confusion_matrices=confusion_matrices,
            confusion_matrices_fig=confusion_matrices_fig,
            metrics_df=metrics_df,
        )

    return (evaluate_model,)


@app.cell(hide_code=True)
def _(np, pl, skprep, ty):
    def binarize_labels(
        eval_df: pl.DataFrame,
        unique_labels: ty.Iterable[str],
        feature: str,
    ) -> tuple[np.ndarray, np.ndarray, skprep.MultiLabelBinarizer]:
        predicted_labels = eval_df.select(feature).to_numpy().squeeze()
        true_labels = eval_df.select(f"{feature}_true").to_numpy().squeeze()

        mlb = skprep.MultiLabelBinarizer()
        mlb.fit([unique_labels])
        true_labels_binarized = mlb.transform(true_labels)
        predicted_labels_binarized = mlb.transform(predicted_labels)

        return true_labels_binarized, predicted_labels_binarized, mlb

    return (binarize_labels,)


@app.cell(hide_code=True)
def _(itertools, np, pl, skm):
    def compute_metrics_df(
        true_labels_binarized: np.ndarray,
        predicted_labels_binarized: np.ndarray,
    ) -> pl.DataFrame:
        metrics: list[dict] = []

        # Exact Match Ratio (Subset Accuracy):
        metrics.append(
            {
                "metric": "Accuracy Score",
                "value": skm.accuracy_score(
                    true_labels_binarized,
                    predicted_labels_binarized,
                ),
            }
        )

        # Hamming Loss
        metrics.append(
            {
                "metric": "Hamming Loss",
                "value": skm.hamming_loss(
                    true_labels_binarized,
                    predicted_labels_binarized,
                ),
            }
        )

        # Jaccard Score (Intersection over Union)
        metrics.extend(
            [
                {
                    "metric": f"Jaccard Score - {avg_type} averaging",
                    "value": skm.jaccard_score(
                        true_labels_binarized,
                        predicted_labels_binarized,
                        average=avg_type,
                        zero_division=0,
                    ),
                }
                for avg_type in [
                    "micro",
                    "macro",
                    "weighted",
                    "samples",
                ]
            ]
        )

        # Precision, Recall, and F1-Score (Label-Based Metrics)
        metrics.extend(
            itertools.chain.from_iterable(
                [
                    [
                        {
                            "metric": f"Precision Score - {avg_type} averaging",
                            "value": skm.precision_score(
                                true_labels_binarized,
                                predicted_labels_binarized,
                                average=avg_type,
                                zero_division=0,
                            ),
                        },
                        {
                            "metric": f"Recall Score - {avg_type} averaging",
                            "value": skm.recall_score(
                                true_labels_binarized,
                                predicted_labels_binarized,
                                average=avg_type,
                                zero_division=0,
                            ),
                        },
                        {
                            "metric": f"F1 Score - {avg_type} averaging",
                            "value": skm.f1_score(
                                true_labels_binarized,
                                predicted_labels_binarized,
                                average=avg_type,
                                zero_division=0,
                            ),
                        },
                    ]
                    for avg_type in [
                        "micro",
                        "macro",
                        "weighted",
                        "samples",
                    ]
                ]
            )
        )

        return pl.from_dicts(metrics)

    return (compute_metrics_df,)


@app.cell(hide_code=True)
def _(ModelId, ai_df, human_df, pl):
    def construct_eval_df(
        model_id: ModelId,
        ai_df: pl.DataFrame = ai_df,
        human_df: pl.DataFrame = human_df,
    ) -> pl.DataFrame:
        provider, model = model_id.split(":")
        eval_df = (
            ai_df.filter(provider=provider, model=model)
            .select(
                "url",
                "purpose",
                "encoding",
                "dimensionality",
            )
            .join(
                human_df,
                on="url",
                how="left",
                suffix="_true",
            )
            # Convert scalar `purpose` to singleton list
            .with_columns(
                pl.concat_list("purpose"),
                pl.concat_list("purpose_true"),
            )
        )
        list_columns = eval_df.select(pl.col(pl.List(pl.String))).columns

        # Data massage & spa time: prepare features for downstream multi-label eval
        eval_df = (
            eval_df
            # Replace empty lists with singleton list ["none"]
            .with_columns(
                *[
                    pl.when(pl.col(col).list.len() == 0)
                    .then(pl.lit(["none"]))
                    .otherwise(col)
                    .alias(col)
                    for col in list_columns
                ]
            )
            # Replace scalar Nones with singleton list ["none"]
            .with_columns(
                *[
                    pl.when(pl.col(col).is_null())
                    .then(pl.lit(["none"]))
                    .otherwise(col)
                    .alias(col)
                    for col in list_columns
                ]
            )
            # Replace None values within lists with "none" label
            .with_columns(
                pl.col(pl.List(pl.String)).list.eval(
                    pl.when(pl.element().is_null())
                    .then(pl.lit("none"))
                    .otherwise(pl.element())
                )
            )
        )
        return eval_df.select(
            "url",
            "difficulty",
            "purpose",
            "purpose_true",
            "encoding",
            "encoding_true",
            "dimensionality",
            "dimensionality_true",
        )

    def enumerate_unique_labels(df: pl.DataFrame, list_col: str) -> list[str]:
        def _enumerate(col: str) -> set[str]:
            return set(
                df.select(col)
                .explode(col)
                .unique(col)
                .sort(col)
                .to_numpy()
                .squeeze()
                .tolist()
            )

        return sorted(_enumerate(list_col).union(_enumerate(list_col + "_true")))

    return construct_eval_df, enumerate_unique_labels


@app.cell(hide_code=True)
def _(human_df, ty):
    ModelId = ty.Literal[
        "google_genai:gemini-2.0-flash",
        "google_genai:gemini-2.5-flash-preview-05-20",
        "google_genai:gemini-2.5-pro-preview-05-06",
        "meta-llama:llama-4-maverick",
        "meta-llama:llama-4-scout",
        "mistralai:mistral-medium-3",
        "mistralai:mistral-small-3.1-24b-instruct",
        "mistralai:pixtral-large-2411",
        "openai:gpt-4.1",
        "openai:gpt-4.1-mini",
        "openai:gpt-4.1-nano",
        "openai:o4-mini",
        "qwen:qwen2.5-vl-32b-instruct",
    ]
    MODELS = list(ty.get_args(ModelId))
    FEATURES = ["purpose", "encoding", "dimensionality"]
    DIFFICULTIES = (
        human_df.select("difficulty")
        .unique("difficulty")
        .drop_nulls()
        .sort("difficulty")
        .to_numpy()
        .squeeze()
        .tolist()
    )
    return DIFFICULTIES, FEATURES, MODELS, ModelId


@app.cell(hide_code=True)
def _(pl):
    ai_df = pl.read_parquet(
        "data/results/provider=*/model=*/analysis.parquet",
        glob=True,
        hive_partitioning=True,
    ).drop("error")
    # ai_df
    return (ai_df,)


@app.cell(hide_code=True)
def _(pl):
    human_df = (
        pl.read_csv("data/vispubData30_updated_07112024.csv")
        .filter(check_encoding_type=1)
        .select(
            "url",
            "encoding_type",
            "dim_type",
            "hardness_type",
        )
        .with_columns(
            purpose_and_encoding=pl.when(
                pl.col("encoding_type").str.contains("schematic")
            )
            .then(
                pl.struct(
                    purpose=pl.lit("schematic"),
                    encoding=pl.col("encoding_type").str.replace("schematic;?", ""),
                )
            )
            .otherwise(
                pl.when(pl.col("encoding_type").str.contains("gui"))
                .then(
                    pl.struct(
                        purpose=pl.lit("gui"),
                        encoding=pl.col("encoding_type").str.replace("gui;?", ""),
                    )
                )
                .otherwise(
                    pl.struct(
                        purpose=pl.lit("vis"),
                        encoding=pl.col("encoding_type"),
                    )
                )
            )
        )
        .drop("encoding_type")
        .unnest("purpose_and_encoding")
        .select(
            "url",
            "purpose",
            pl.col("encoding").str.split(";"),
            pl.col("dim_type").str.split(";").alias("dimensionality"),
            pl.col("hardness_type").alias("difficulty"),
        )
        # In all list columns replace empty string elements with None
        .with_columns(
            pl.col(pl.List(pl.String)).list.eval(
                pl.when(pl.element().str.strip_chars(" ").str.len_chars() == 0)
                .then(pl.lit(None))
                .otherwise(pl.element())
            )
        )
    )
    # human_df
    return (human_df,)


@app.cell(hide_code=True)
def _(Sequence, np, ty):
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def plot_multilabel_cm_with_annotations(
        multilabel_cm: np.ndarray,
        class_names: ty.Iterable[str] | None = None,
        cols: int = 3,
        vertical_spacing: float = 0.15,
        horizontal_spacing: float = 0.1,
        figure_title: str = "Multilabel Confusion Matrices (Annotated)",
        height_per_row_px: int = 250,
        width_per_col_px: int = 300,
        colorscale: str | list[list[float | str]] = "Blues",
        heatmap_x_labels: ty.Iterable[str] = ("Pred 0", "Pred 1"),
        heatmap_y_labels: ty.Iterable[str] = ("True 0", "True 1"),
    ) -> go.Figure:
        if not isinstance(multilabel_cm, np.ndarray) or multilabel_cm.ndim != 3:
            raise ValueError("multilabel_cm must be a 3D NumPy array.")

        n_classes = multilabel_cm.shape[0]

        if n_classes == 0:
            # Return an empty figure with a title if there's no data
            fig = go.Figure()
            fig.update_layout(
                title_text=figure_title,
                xaxis={"visible": False, "showticklabels": False},
                yaxis={"visible": False, "showticklabels": False},
                annotations=[
                    {
                        "text": "No data to display.",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 16},
                    }
                ],
            )
            return fig

        if cols <= 0:
            raise ValueError("cols must be a positive integer.")

        if class_names is None:
            # Generate default class names if not provided
            actual_class_names: Sequence[str] = [f"Class {i}" for i in range(n_classes)]
        else:
            if len(class_names) != n_classes:
                raise ValueError(
                    f"Length of class_names ({len(class_names)}) must match "
                    f"n_classes derived from multilabel_cm ({n_classes})."
                )
            actual_class_names = class_names

        # Calculate number of rows needed for the subplots
        num_rows = (n_classes + cols - 1) // cols

        fig = make_subplots(
            rows=num_rows,
            cols=cols,
            subplot_titles=list(
                actual_class_names
            ),  # make_subplots expects a list or tuple
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
        )

        for i in range(n_classes):
            current_row_idx = i // cols + 1  # Plotly subplot indices are 1-based
            current_col_idx = i % cols + 1  # Plotly subplot indices are 1-based

            # Current confusion matrix for the class
            cm_slice: np.ndarray = multilabel_cm[i]

            # Create an individual annotated heatmap figure for this class's CM
            # Note: ff.create_annotated_heatmap expects x and y labels to be lists.
            # It expects z (cm_slice) to be a list of lists or a 2D numpy array.
            individual_cm_fig = ff.create_annotated_heatmap(
                z=cm_slice,
                x=list(heatmap_x_labels),
                y=list(heatmap_y_labels),
                colorscale=colorscale,
                showscale=False,  # Individual color scales are off; annotations show values.
            )

            # Extract the heatmap trace and annotations from the temporary individual figure
            # individual_cm_fig.data contains the heatmap trace(s)
            # individual_cm_fig.layout.annotations contains the text annotations for cells
            heatmap_trace = individual_cm_fig.data[0]
            annotations_from_ff = individual_cm_fig.layout.annotations

            # Add the heatmap trace to the main figure's appropriate subplot
            fig.add_trace(heatmap_trace, row=current_row_idx, col=current_col_idx)

            # Add annotations to the main figure, correctly positioned in the subplot
            # The x, y coordinates in `ann` are data coordinates for the individual heatmap.
            # By specifying `row` and `col` in `fig.add_annotation`, Plotly maps these
            # coordinates to the correct subplot's axes.
            # `ann.font` is copied to preserve text color (for contrast) and size.
            for ann in annotations_from_ff:
                fig.add_annotation(
                    x=ann.x,
                    y=ann.y,
                    text=ann.text,
                    font=ann.font,  # Preserves font properties (color, size)
                    showarrow=False,
                    row=current_row_idx,
                    col=current_col_idx,
                )

        # Update overall figure layout
        fig.update_layout(
            title_text=figure_title,
            height=height_per_row_px * num_rows,
            width=width_per_col_px * cols,
            # Example: To make all subplot axes consistent if data ranges vary wildly (not typical for CMs)
            # yaxis_autorange='reversed' # For matrix-like display, if needed
        )

        return fig

    return go, plot_multilabel_cm_with_annotations


@app.cell(hide_code=True)
def _():
    import itertools
    import typing as ty

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    import sklearn.metrics as skm
    import sklearn.preprocessing as skprep
    from tqdm.notebook import tqdm

    return alt, itertools, mo, np, pl, skm, skprep, tqdm, ty


if __name__ == "__main__":
    app.run()
