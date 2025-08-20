import marimo

__generated_with = "0.14.17"
app = marimo.App(width="columns", layout_file="layouts/evaluation.grid.json")

with app.setup(hide_code=True):
    import itertools
    import pathlib
    import typing as ty

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    import sklearn.metrics as skm
    import sklearn.preprocessing as skprep
    from tqdm.notebook import tqdm

    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


@app.cell(hide_code=True)
def _(DIFFICULTIES, FEATURES, MODELS, evaluate_model):
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
def _(
    MODEL_ID_LABELS,
    PROVIDER_ID_LABELS,
    construct_metrics_overview_df,
    evaluations,
    metric_picker,
):
    metrics_overview_df = (
        construct_metrics_overview_df(evaluations)
        .with_columns(
            pl.col("provider_id").replace(
                old=list(PROVIDER_ID_LABELS.keys()),
                new=list(PROVIDER_ID_LABELS.values()),
            ),
            pl.col("model_id").replace(
                old=list(MODEL_ID_LABELS.keys()),
                new=list(MODEL_ID_LABELS.values()),
            ),
            pl.col("feature").replace(
                old=["dimensionality"],
                new=["dim."],
            ),
        )
        # difficulty_level==5 stands for aggregated performance over all other difficulty levels.
        # We don't want to visualize those in this context as the comparison would be misleading
        .filter(pl.col("difficulty_level") != 5)
    )

    metric = metric_picker.value
    metrics_overview_chart = (
        alt.Chart(metrics_overview_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "difficulty_level:N",
                title="Difficulty",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                f"{metric}:Q",
                title=metric,
                axis=alt.Axis(titleFontSize=10),
            ),
            color=alt.Color(
                "provider_id:N",
                title="Provider",
                scale=alt.Scale(
                    domain=list(PROVIDER_ID_LABELS.values()),
                    range=[
                        "#75b9a1",
                        "#DB4437",
                        "#ee782f",
                        "#17A9FD",
                        "#5442c7",
                    ],
                ),
                legend=alt.Legend(
                    titleFontSize=16,
                    labelFontSize=14,
                    orient="top",
                    direction="horizontal",
                    offset=-20,
                ),
            ),
            column=alt.Column(
                "model_id:N",
                title="Model",
                sort=list(MODEL_ID_LABELS.values()),
                header=alt.Header(
                    titleFontSize=16,
                    labelFontSize=13.5,
                ),
            ),
            row=alt.Row(
                "feature:N",
                title="Predicted Feature",
                sort=["purpose", "encoding", "dim."],
                header=alt.Header(
                    titleFontSize=16,
                    labelFontSize=14,
                    titlePadding=-5,
                    labelPadding=5,
                ),
            ),
            tooltip=[
                alt.Tooltip("model_id:N", title="Model"),
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("difficulty:N", title="Difficulty"),
                alt.Tooltip(f"{metric}:Q", title=metric),
            ],
        )
        .properties(width=100, height=55)
        .configure_view(
            # strokeWidth=0.5,
        )
    )

    save_chart(metrics_overview_chart, f"nogit/figures/metric-overview-{metric}")

    mo.vstack(
        [
            metric_picker,
            metrics_overview_chart,
        ]
    )
    return metric, metrics_overview_df


@app.cell(hide_code=True)
def _(metrics_overview_df):
    metrics_overview_df
    return


@app.cell(hide_code=True)
def _(MODEL_ID_LABELS, PROVIDER_ID_LABELS, evaluations):
    classification_accuracy_overview_df = construct_classification_accuracy_overview_df(
        evaluations
    ).with_columns(
        pl.col("provider_id").replace(
            old=list(PROVIDER_ID_LABELS.keys()),
            new=list(PROVIDER_ID_LABELS.values()),
        ),
        pl.col("model_id").replace(
            old=list(MODEL_ID_LABELS.keys()),
            new=list(MODEL_ID_LABELS.values()),
        ),
    )

    def prepare_classification_accuracy_chart_df(
        overview_df: pl.DataFrame,
        feature: ty.Literal["purpose", "encoding", "dimensionality"],
        metric: ty.Literal["precision", "recall", "f1-score"],
    ) -> pl.DataFrame:
        return (
            overview_df.filter(feature=pl.lit(feature))
            .group_by("provider_id", "model_id", "feature", "label")
            .agg(pl.mean(metric))
        )

    def create_classification_accuracy_chart(
        feature: ty.Literal["purpose", "encoding", "dimensionality"],
        metric: ty.Literal["precision", "recall", "f1-score"],
        classification_accuracy_overview_df: pl.DataFrame,
        rect_size: tuple[float, float] = (30, 30),
        chart_size: tuple[float, float] | None = None,
        sort_cfg: dict | None = None,
        title_cfg: dict | None = None,
        axis_cfg: dict | None = None,
    ) -> alt.Chart:
        df = prepare_classification_accuracy_chart_df(
            classification_accuracy_overview_df,
            feature,
            metric,
        )
        num_rects_x = df.unique("model_id").height
        num_rects_y = df.unique("label").height
        rect_size_x, rect_size_y = rect_size

        sort_cfg = sort_cfg or {}
        title_cfg = title_cfg or {}
        axis_cfg = axis_cfg or {}

        def cfg(scope: dict, key: str, fallback):
            if key in scope:
                return scope[key]
            return fallback

        return (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=alt.X(
                    "model_id:N",
                    title=cfg(title_cfg, "x", "Model"),
                    sort=cfg(
                        sort_cfg,
                        "x",
                        alt.EncodingSortField(
                            field=metric,
                            op="mean",
                            order="descending",
                        ),
                    ),
                    axis=cfg(
                        axis_cfg,
                        "x",
                        alt.Axis(
                            titleFontSize=14,
                            labelFontSize=12,
                            labelAngle=-45,
                        ),
                    ),
                ),
                y=alt.Y(
                    "label:N",
                    title=cfg(title_cfg, "y", feature.capitalize()),
                    sort=cfg(
                        sort_cfg,
                        "y",
                        alt.EncodingSortField(
                            field=metric,
                            op="mean",
                            order="descending",
                        ),
                    ),
                    axis=cfg(
                        axis_cfg,
                        "y",
                        alt.Axis(
                            titleFontSize=14,
                            labelFontSize=12,
                            labelAngle=0,
                            titlePadding=15,
                        ),
                    ),
                ),
                color=alt.Color(
                    f"{metric}:Q",
                    title=metric.capitalize(),
                    scale=alt.Scale(
                        domain=[0, 1],
                        scheme="viridis",
                        reverse=True,
                    ),
                ),
                tooltip=[
                    alt.Tooltip("model_id:N", title="Model"),
                    alt.Tooltip("label:N", title="Label"),
                    alt.Tooltip(
                        f"{metric}:Q",
                        title=metric.capitalize(),
                        format=".2f",
                    ),
                ],
            )
            .properties(
                width=rect_size_x * num_rects_x
                if chart_size is None
                else chart_size[0],
                height=rect_size_y * num_rects_y
                if chart_size is None
                else chart_size[1],
            )
        )

    classification_accuracy_overview_chart = alt.hconcat(
        *[
            create_classification_accuracy_chart(
                "purpose",
                "f1-score",
                classification_accuracy_overview_df,
                chart_size=(300, 175),
            ),
            create_classification_accuracy_chart(
                "encoding",
                "f1-score",
                classification_accuracy_overview_df,
                chart_size=(300, 175),
            ),
            create_classification_accuracy_chart(
                "dimensionality",
                "f1-score",
                classification_accuracy_overview_df,
                chart_size=(300, 175),
            ),
        ],
        spacing=50,
    )

    save_chart(
        classification_accuracy_overview_chart,
        "nogit/figures/classification-accuracy-overview",
    )

    classification_accuracy_overview_chart
    return (
        classification_accuracy_overview_df,
        create_classification_accuracy_chart,
        prepare_classification_accuracy_chart_df,
    )


@app.cell(hide_code=True)
def _(
    classification_accuracy_overview_df,
    create_classification_accuracy_chart,
):
    RECT_SIZE = (22.5, 22.5)

    alt.vconcat(
        *[
            create_classification_accuracy_chart(
                "purpose",
                "f1-score",
                classification_accuracy_overview_df,
                sort_cfg={"x": alt.Undefined, "y": alt.Undefined},
                title_cfg={"x": None},
                axis_cfg={"x": None},
                rect_size=RECT_SIZE,
            ),
            create_classification_accuracy_chart(
                "encoding",
                "f1-score",
                classification_accuracy_overview_df,
                sort_cfg={"x": alt.Undefined, "y": alt.Undefined},
                title_cfg={"x": None},
                axis_cfg={"x": None},
                rect_size=RECT_SIZE,
            ),
            create_classification_accuracy_chart(
                "dimensionality",
                "f1-score",
                classification_accuracy_overview_df,
                sort_cfg={"x": alt.Undefined, "y": alt.Undefined},
                rect_size=RECT_SIZE,
            ),
        ],
        spacing=5,
    )
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""# Metrics""")
    return


@app.cell(hide_code=True)
def _(DIFFICULTIES, FEATURES, MODELS):
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
def _(evaluation, feature, model_id):
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
def _(evaluation):
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
def _(evaluation):
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
                difficulty_level=pl.col("difficulty")
                .replace(
                    {
                        "two_easy": 1,
                        "one_hard": 2,
                        "two_hard": 3,
                        "others": 4,
                        "all": 5,
                    }
                )
                .cast(pl.UInt8)
            )
            .with_columns(pl.col("model_id").str.split(":"))
            .with_columns(
                provider_id=pl.col("model_id").list.get(0),
                model_id=pl.col("model_id").list.get(1),
            )
            .sort("provider_id", "model_id")
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

    metrics = evaluation.metrics_df.select("metric").to_numpy().squeeze()
    return construct_metrics_overview_df, metrics


@app.cell(hide_code=True)
def _(
    construct_eval_df,
    enumerate_unique_labels,
    plot_multilabel_cm_with_annotations,
):
    class Evaluation(ty.NamedTuple):
        preview_df: pl.DataFrame
        classification_report: str
        classification_report_dict: dict
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
            output_dict=False,
        )

        classification_report_dict = skm.classification_report(
            true_labels_binarized,
            predicted_labels_binarized,
            target_names=mlb.classes_,
            zero_division=0,
            output_dict=True,
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
            classification_report_dict=classification_report_dict,
            confusion_matrices=confusion_matrices,
            confusion_matrices_fig=confusion_matrices_fig,
            metrics_df=metrics_df,
        )

    return (evaluate_model,)


@app.cell(hide_code=True)
def _(metrics):
    metric_picker = mo.ui.dropdown(
        sorted(metrics), label="Metric", value=sorted(metrics)[0]
    )
    return (metric_picker,)


@app.cell(hide_code=True)
def _(ModelId, ai_df, human_df):
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
def _():
    PROVIDER_ID_LABELS = {
        "openai": "OpenAI",
        "google_genai": "Google GenAI",
        "mistralai": "Mistral AI",
        "meta-llama": "Meta Llama",
        "qwen": "Qwen",
    }

    MODEL_ID_LABELS = {
        # OpenAI
        "o4-mini": "O4 Mini",
        "gpt-4.1": "GPT-4.1",
        "gpt-4.1-mini": "GPT-4.1 Mini",
        "gpt-4.1-nano": "GPT-4.1 Nano",
        # Google
        "gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro",
        "gemini-2.5-flash-preview-05-20": "Gemini 2.5 Flash",
        "gemini-2.0-flash": "Gemini 2.0 Flash",
        # Mistral
        "pixtral-large-2411": "Pixtral Large",
        "mistral-medium-3": "Mistral M. 3",
        "mistral-small-3.1-24b-instruct": "Mistral S. 3.1 24B",
        # Meta
        "llama-4-maverick": "Llama 4 Maverick",
        "llama-4-scout": "Llama 4 Scout",
        # Qwen
        "qwen2.5-vl-32b-instruct": "Qwen 2.5 VL 32B",
    }
    return MODEL_ID_LABELS, PROVIDER_ID_LABELS


@app.cell(hide_code=True)
def _(human_df):
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


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""# Poster Components""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Do Bigger Models Alays Perform Better?

    Examples below show instances where smaller models of the same family showed better alignment with human experts than the bigger models.
    """
    )
    return


@app.cell
def _(comparison_df):
    small_beats_big_demo = (
        "https://ivcl.jiangsn.com/VIS30K/Images/2010/InfoVisJ.1182.2.png"
    )

    comparison_df.filter(
        (
            # (pl.col("model").is_in([BIG_MODEL, SMALL_MODEL])) &
            pl.col("url") == small_beats_big_demo
        )
    )
    return


@app.cell(hide_code=True)
def _(comparison_df):
    BIG_MODEL = "gpt-4.1"
    SMALL_MODEL = "gpt-4.1-mini"
    (
        comparison_df.filter(
            (pl.col("model").is_in([BIG_MODEL, SMALL_MODEL]))
            # Human expert explicitly assigned an unambigious "encoding"
            & (~pl.col("encoding").list.contains(pl.lit(None)))
            & (~pl.col("encoding").list.contains(pl.lit("others")))
        )
        .sort("year", "url", descending=True)
        .pivot(
            "model",
            index=["year", "url", "encoding"],
            values=[
                "encoding_predicted",
                "encoding_metrics",
            ],
        )
        # Looking for all instances where the smaller model performed better than the bigger one
        .filter(
            (
                pl.col(f"encoding_metrics_{SMALL_MODEL}").struct.field("precision")
                > pl.col(f"encoding_metrics_{BIG_MODEL}").struct.field("precision")
            )
            & (
                pl.col(f"encoding_metrics_{SMALL_MODEL}").struct.field("recall")
                > pl.col(f"encoding_metrics_{BIG_MODEL}").struct.field("recall")
            )
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Consensus in Failure, Chaos in Specifics

    Examples below show instances where flagship models all failed in perfect recognition of encoding, but they did so differently.
    """
    )
    return


@app.cell(hide_code=True)
def _(comparison_df):
    # Hand-picked example
    different_failures_demo_pick = (
        "https://ivcl.jiangsn.com/VIS30K/Images/2020/SciVisJ.806.7.png"
    )
    (
        comparison_df.filter(
            (
                pl.col("url") == different_failures_demo_pick
                # & (pl.col("model").is_in(FLAGSHIP_MODELS))
            )
        )
    )
    return


@app.cell(hide_code=True)
def _(FLAGSHIP_MODELS, comparison_df):
    (
        comparison_df.filter(
            # Model recognized "purpose" perfectly
            (pl.col("purpose_metrics").struct.field("precision") == 1.0)
            & (pl.col("purpose_metrics").struct.field("recall") == 1.0)
            # But it did not recognize "encoding" perfectly
            & (pl.col("encoding_metrics").struct.field("precision") != 1.0)
            & (pl.col("encoding_metrics").struct.field("recall") != 1.0)
            # Even though human expert explicitly assigned an unambigious "encoding"
            & (~pl.col("encoding").list.contains(pl.lit(None)))
            & (~pl.col("encoding").list.contains(pl.lit("others")))
            # Even though model is big and smart
            & (pl.col("model").is_in(FLAGSHIP_MODELS))
        ).sort("year", descending=True)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## The "AI Vision Test"

    The items below are instances where a flagship model such as Gemini 2.5 Pro, GPT 4.1 or O4-mini failed to properly recognize the `purpose` and `encoding` of an image, even though experts were able to categorize them without ambiguity.
    """
    )
    return


@app.cell(hide_code=True)
def _(FLAGSHIP_MODELS, comparison_df):
    # Hand-picked example
    ai_vision_test_demo_pick = (
        "https://ivcl.jiangsn.com/VIS30K/Images/2010/VisJ.1291.9.png"
    )
    (
        comparison_df.filter(
            (pl.col("url") == ai_vision_test_demo_pick)
            & (pl.col("model").is_in(FLAGSHIP_MODELS))
        )
    )
    return


@app.cell(hide_code=True)
def _(FLAGSHIP_MODELS, comparison_df):
    (
        comparison_df.filter(
            # Model did not recognize "purpose" perfectly
            (pl.col("purpose_metrics").struct.field("precision") != 1.0)
            & (pl.col("purpose_metrics").struct.field("recall") != 1.0)
            # Model did not recognize "encoding" perfectly
            & (pl.col("encoding_metrics").struct.field("precision") != 1.0)
            & (pl.col("encoding_metrics").struct.field("recall") != 1.0)
            # Even though human expert explicitly assigned an unambigious "encoding"
            & (~pl.col("encoding").list.contains(pl.lit(None)))
            & (~pl.col("encoding").list.contains(pl.lit("others")))
            # Even though model is big and smart
            & (pl.col("model").is_in(FLAGSHIP_MODELS))
        ).sort("year", descending=True)
    )
    return


@app.cell(hide_code=True)
def _():
    FLAGSHIP_MODELS = [
        "gemini-2.5-pro-preview-05-06",
        "gpt-4.1",
        "o4-mini",
    ]
    return (FLAGSHIP_MODELS,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Per-Encoding F1-Score""")
    return


@app.cell(hide_code=True)
def _(
    classification_accuracy_overview_df,
    create_classification_accuracy_chart_for_poster,
):
    create_classification_accuracy_chart_for_poster(
        "encoding",
        "f1-score",
        classification_accuracy_overview_df,
        chart_size=(1.0 * 450, 1.0 * 500),
    )
    return


@app.cell(hide_code=True)
def _(prepare_classification_accuracy_chart_df):
    def create_classification_accuracy_chart_for_poster(
        feature: ty.Literal["purpose", "encoding", "dimensionality"],
        metric: ty.Literal["precision", "recall", "f1-score"],
        classification_accuracy_overview_df: pl.DataFrame,
        rect_size: tuple[float, float] = (30, 30),
        chart_size: tuple[float, float] | None = None,
        sort_cfg: dict | None = None,
        title_cfg: dict | None = None,
        axis_cfg: dict | None = None,
    ) -> alt.Chart:
        df = prepare_classification_accuracy_chart_df(
            classification_accuracy_overview_df,
            feature,
            metric,
        )
        num_rects_x = df.unique("model_id").height
        num_rects_y = df.unique("label").height
        rect_size_x, rect_size_y = rect_size

        sort_cfg = sort_cfg or {}
        title_cfg = title_cfg or {}
        axis_cfg = axis_cfg or {}

        def cfg(scope: dict, key: str, fallback):
            if key in scope:
                return scope[key]
            return fallback

        return (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=alt.X(
                    "model_id:N",
                    title=cfg(title_cfg, "x", "Model"),
                    sort=cfg(
                        sort_cfg,
                        "x",
                        alt.EncodingSortField(
                            field=metric,
                            op="mean",
                            order="descending",
                        ),
                    ),
                    axis=cfg(
                        axis_cfg,
                        "x",
                        alt.Axis(
                            titleFontSize=24,
                            labelFontSize=22,
                            labelAngle=-45,
                        ),
                    ),
                ),
                y=alt.Y(
                    "label:N",
                    title=cfg(title_cfg, "y", feature.capitalize()),
                    sort=cfg(
                        sort_cfg,
                        "y",
                        alt.EncodingSortField(
                            field=metric,
                            op="mean",
                            order="descending",
                        ),
                    ),
                    axis=cfg(
                        axis_cfg,
                        "y",
                        alt.Axis(
                            titleFontSize=24,
                            labelFontSize=22,
                            labelAngle=0,
                            titlePadding=0,
                        ),
                    ),
                ),
                color=alt.Color(
                    f"{metric}:Q",
                    title=metric.capitalize(),
                    scale=alt.Scale(
                        domain=[0, 1],
                        scheme="tealblues",
                        reverse=False,
                    ),
                    legend=alt.Legend(
                        orient="right",  # Position the legend at the bottom
                        titleOrient="top",  # Position the legend title above the gradient
                        padding=0,  # Add some space around the legend
                        titleFontSize=20,
                        labelFontSize=16,
                    ),
                ),
                tooltip=[
                    alt.Tooltip("model_id:N", title="Model"),
                    alt.Tooltip("label:N", title="Label"),
                    alt.Tooltip(
                        f"{metric}:Q",
                        title=metric.capitalize(),
                        format=".2f",
                    ),
                ],
            )
            .properties(
                width=rect_size_x * num_rects_x
                if chart_size is None
                else chart_size[0],
                height=rect_size_y * num_rects_y
                if chart_size is None
                else chart_size[1],
            )
        )

    return (create_classification_accuracy_chart_for_poster,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Alignment Measured via F1-Score""")
    return


@app.cell(hide_code=True)
def _(MODEL_ID_LABELS, PROVIDER_ID_LABELS, metric, metrics_overview_df):
    # 1. Create a base chart for the *inner* plot, WITHOUT the row/column faceting.
    # This base chart defines the common x, y, color, and tooltip encodings.
    base = alt.Chart(metrics_overview_df).encode(
        x=alt.X(
            "difficulty_level:N",
            title="Difficulty",
            axis=alt.Axis(
                labelAngle=0,
                titleFontSize=20,
                labelFontSize=20,
            ),
        ),
        y=alt.Y(
            f"{metric}:Q",
            title=metric,
            axis=alt.Axis(
                titleFontSize=16,
                labelFontSize=16,
            ),
        ),
        color=alt.Color(
            "provider_id:N",
            title="Provider",
            scale=alt.Scale(
                domain=list(PROVIDER_ID_LABELS.values()),
                range=[
                    "#75b9a1",
                    "#DB4437",
                    "#ee782f",
                    "#17A9FD",
                    "#5442c7",
                ],
            ),
            legend=alt.Legend(
                titleFontSize=26,
                labelFontSize=22,
                orient="bottom",
                direction="horizontal",
                offset=0,
                symbolSize=250,
            ),
        ),
        tooltip=[
            alt.Tooltip("model_id:N", title="Model"),
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("difficulty:N", title="Difficulty"),
            alt.Tooltip(f"{metric}:Q", title=metric),
        ],
    )

    # 2. Define the individual layers from the base.
    bars = base.mark_bar()

    text = base.mark_text(
        align="center",
        baseline="bottom",
        dy=-5,  # Nudge text up so it doesn't overlap with the bar
        fontWeight="bold",
        fontSize=14,
    ).encode(
        text=alt.Text(f"{metric}:Q", format=".2f")  # Display the metric value
    )

    # 3. Layer the charts first, THEN apply faceting and properties.
    final_chart = (
        (bars + text)
        .properties(width=145, height=80)
        .facet(
            column=alt.Column(
                "model_id:N",
                title=" ",
                sort=list(MODEL_ID_LABELS.values()),
                header=alt.Header(
                    titleFontSize=26,
                    titlePadding=0,
                    labelFontSize=18,
                    labelFontWeight="bold",
                ),
            ),
            row=alt.Row(
                "feature:N",
                title=None,
                sort=["purpose", "encoding", "dim."],
                header=alt.Header(
                    titleFontSize=26,
                    titlePadding=-5,
                    labelFontSize=22,
                    labelFontWeight="bold",
                    labelPadding=5,
                ),
            ),
            spacing={"row": 40, "column": 20},
        )
        .configure_view(stroke=None)
        .configure_axis(grid=False)
    )

    # Render the final chart
    mo.ui.altair_chart(final_chart)
    return


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""# Data""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Human Labels vs. VLM Labels

    This dataset lines up the human-created labels against the VLM-created labels for `purpose`, `encoding` and `dimensionality` of visualizations across all the probed models.
    """
    )
    return


@app.cell
def _(ai_df, human_df, metrics_series):
    comparison_df = (
        human_df.join(
            ai_df,
            on="url",
            how="right",
            suffix="_predicted",
        )
        .with_columns(
            pl.concat_list("purpose"),
            pl.concat_list("purpose_predicted"),
        )
        .select(
            "year",
            "url",
            "difficulty",
            "purpose",
            "purpose_predicted",
            metrics_series("purpose"),
            pl.col("encoding").list.sort(),
            pl.col("encoding_predicted").list.sort(),
            metrics_series("encoding"),
            pl.col("dimensionality").list.sort(),
            pl.col("dimensionality_predicted").list.sort(),
            metrics_series("dimensionality"),
            "provider",
            "model",
        )
    )
    comparison_df
    return (comparison_df,)


@app.cell(hide_code=True)
def _():
    def compute_individual_metrics(struct: dict | None):
        if struct is None:
            return None

        real: set[str] = set(struct["real"]) if struct["real"] else set()
        predicted: set[str] = set(struct["predicted"]) if struct["predicted"] else set()
        tp = len(real & predicted)
        precision = tp / len(predicted) if len(predicted) > 0 else 0
        recall = tp / len(real) if len(real) > 0 else 0
        return {
            "precision": float(precision),
            "recall": float(recall),
        }

    def metrics_series(attribute: str) -> pl.Series:
        return (
            pl.struct(
                real=attribute,
                predicted=f"{attribute}_predicted",
            )
            .map_elements(
                compute_individual_metrics,
                return_dtype=pl.Struct(
                    {
                        "precision": pl.Float64,
                        "recall": pl.Float64,
                    }
                ),
            )
            .alias(f"{attribute}_metrics")
        )

    return (metrics_series,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Raw Expert-Labeled Dataset

    This is the original, unprocessed dataset coming from a subset of VIS30K annotated by experts, loaded from [VISImageNavigator](https://visimagenavigator.github.io/index.html).
    """
    )
    return


@app.cell(hide_code=True)
def _():
    human_raw_df = read_human_raw_df()
    human_raw_df
    return (human_raw_df,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Processed Expert-labeled Dataset

    The original dataset had to be pre-processed to split the `purpose` and `encoding` from the `encoding_type` column.
    """
    )
    return


@app.cell
def _(human_raw_df):
    human_df = construct_human_df(human_raw_df)
    human_df
    return (human_df,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## VLM-Annotated Dataset

    This dataset holds the inference results across all probed models. Each model was prompted to determine the `purpose`, `encoding` and `dimensionality` of the visualizations.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    ai_df = read_ai_df()
    ai_df
    return (ai_df,)


@app.cell(column=4, hide_code=True)
def _():
    mo.md(r"""# Library""")
    return


@app.function(hide_code=True)
def construct_classification_accuracy_overview_df(
    evaluations: list[dict],
) -> pl.DataFrame:
    return (
        pl.from_dicts(
            itertools.chain.from_iterable(
                [
                    [
                        {
                            **item["config"],
                            "label": key,
                            **value,
                        }
                        for key, value in item[
                            "evaluation"
                        ].classification_report_dict.items()
                        if not key.endswith(" avg")
                    ]
                    for item in evaluations
                ]
            )
        )
        .with_columns(pl.col("model_id").str.split(":"))
        .with_columns(
            provider_id=pl.col("model_id").list.get(0),
            model_id=pl.col("model_id").list.get(1),
        )
    )


@app.function(hide_code=True)
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


@app.function(hide_code=True)
def compute_metrics_df(
    true_labels_binarized: np.ndarray,
    predicted_labels_binarized: np.ndarray,
) -> pl.DataFrame:
    metrics: list[dict] = []

    # Exact Match Ratio (Subset Accuracy):
    metrics.append(
        {
            "metric": "accuracy",
            "value": skm.accuracy_score(
                true_labels_binarized,
                predicted_labels_binarized,
            ),
        }
    )

    # Hamming Loss
    metrics.append(
        {
            "metric": "hamming",
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
                "metric": f"jaccard_{avg_type}",
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
                        "metric": f"precision_{avg_type}",
                        "value": skm.precision_score(
                            true_labels_binarized,
                            predicted_labels_binarized,
                            average=avg_type,
                            zero_division=0,
                        ),
                    },
                    {
                        "metric": f"recall_{avg_type}",
                        "value": skm.recall_score(
                            true_labels_binarized,
                            predicted_labels_binarized,
                            average=avg_type,
                            zero_division=0,
                        ),
                    },
                    {
                        "metric": f"f1_{avg_type}",
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


@app.function(hide_code=True)
def construct_human_df(human_raw_df: pl.DataFrame) -> pl.DataFrame:
    return (
        human_raw_df.with_columns(
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
            pl.exclude("encoding", "dim_type", "hardness_type"),
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


@app.function(hide_code=True)
def save_chart(chart: alt.Chart, name: str, **kwargs) -> pathlib.Path:
    out = pathlib.Path(".") / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    chart.save(out, "png", scale_factor=8, **kwargs)
    return out


@app.function(hide_code=True)
def read_ai_df() -> pl.DataFrame:
    return pl.read_parquet(
        "data/results/provider=*/model=*/analysis.parquet",
        glob=True,
        hive_partitioning=True,
    ).drop("error")


@app.function(hide_code=True)
def read_human_raw_df(
    vispubdata_csv_url: str = "https://media.githubusercontent.com/media/peter-gy/AutoVisType/refs/heads/main/data/vispubData30_updated_07112024.csv",
) -> pl.DataFrame:
    return (
        pl.read_csv(vispubdata_csv_url)
        .filter(check_encoding_type=1)
        .select(
            pl.col("Year").alias("year"),
            "url",
            "encoding_type",
            "dim_type",
            "hardness_type",
        )
    )


@app.cell(hide_code=True)
def _(Sequence):
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

    return (plot_multilabel_cm_with_annotations,)


if __name__ == "__main__":
    app.run()
