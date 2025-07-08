import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
async def _(
    llm,
    logger,
    mo,
    model_name,
    model_picker,
    pathlib,
    pl,
    run_many_with_progress,
    sample_df,
):
    urls = sample_df.select("url").to_numpy().squeeze().tolist()
    provider, model = model_name.replace("openrouter:", "").replace("/", ":").split(":")
    outputfile = pathlib.Path(
        f"data/results/provider={provider}/model={model}/analysis.parquet"
    )

    if outputfile.exists():
        logger.info(f"Loading cached results from {outputfile}")
        results = pl.read_parquet(str(outputfile))
    else:
        results = await run_many_with_progress(urls, llm, model_name)
        outputfile.parent.mkdir(parents=True, exist_ok=True)
        results.write_parquet(str(outputfile))

    mo.vstack(
        [
            model_picker,
            results,
        ]
    )
    return


@app.cell
def _(Analysis, init_llm, model_picker):
    model_name = model_picker.value
    llm = init_llm(model_name).with_structured_output(Analysis)
    return llm, model_name


@app.cell
def _(cs, pl, vispub_df):
    # Getting an image for each encoding for each dimensionality for each year bin
    sample_df = (
        vispub_df.with_columns(cs.matches("*_type").str.split(";"))
        .explode("encoding_type")
        .explode("dim_type")
        .explode("hardness_type")
        .sort("Year", "hardness_type", "dim_type", "encoding_type")
        .group_by(
            # Binning years uniformly
            pl.col("Year").qcut(3),
            "encoding_type",
            "dim_type",
            "hardness_type",
            maintain_order=True,
        )
        .agg(pl.first("url"))
        .select("url")
        .unique("url", maintain_order=True)
        .join(vispub_df, on="url", how="left", maintain_order="right")
        .select(vispub_df.columns)
    )
    sample_df
    return (sample_df,)


@app.cell(hide_code=True)
def _(pl):
    vispub_df = (
        # We load the VIS30K dataset which backs the VisImageNavigator app
        pl.read_csv("data/vispubData30_updated_07112024.csv")
        # We are only interested in the items where an expert encoding is present
        .filter(check_encoding_type=1)
        # We select the feature subset that's useful for stratified sampling and eval
        .select(
            "Year",
            "url",
            "cap_url",
            "encoding_type",
            "dim_type",
            "hardness_type",
        )
    )
    return (vispub_df,)


@app.cell(hide_code=True)
def _(DEFAULT_MODEL, MODELS, mo):
    model_picker = mo.ui.dropdown(
        MODELS,
        value=DEFAULT_MODEL,
        label="Model",
    )
    return (model_picker,)


@app.cell(hide_code=True)
def _(BaseChatModel, MAX_CONCURRENT_TASKS, mo, pl, run_many):
    async def run_many_with_progress(
        urls: list[str],
        llm: BaseChatModel,
        model_name: str,
    ) -> pl.DataFrame:
        with mo.status.progress_bar(
            total=len(urls),
            show_eta=True,
            show_rate=True,
        ) as bar:
            df = await run_many(
                urls,
                llm,
                model_name,
                max_concurrent_tasks=MAX_CONCURRENT_TASKS,
                on_after_generate=lambda _: bar.update(),
            )

        return df

    return (run_many_with_progress,)


@app.cell(hide_code=True)
def _(Analysis, BaseChatModel, asyncio, generate_analysis, pl, ty):
    async def run_many(
        urls: list[str],
        llm: BaseChatModel,
        model_name: str,
        max_concurrent_tasks: int = 10,
        on_after_generate: ty.Callable[[Analysis], None] | None = None,
    ) -> pl.DataFrame:
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def process_url(url: str):
            async with semaphore:
                try:
                    analysis = await generate_analysis(llm, url, model_name)
                    result = {"url": url, "error": None} | analysis.model_dump()
                    return result, analysis
                except Exception as e:
                    result = {
                        "url": url,
                        "error": str(e),
                    } | Analysis(
                        purpose=None,
                        encoding=[],
                        dimensionality=[],
                    ).model_dump()
                    return result, None
                finally:
                    if on_after_generate is not None:
                        on_after_generate(analysis)

        results = await asyncio.gather(
            *(process_url(url) for url in urls),
            return_exceptions=True,
        )

        # Filter out exceptions and process callbacks
        valid_results = []
        for result in results:
            if not isinstance(result, Exception):
                data, analysis = result
                valid_results.append(data)

        return pl.from_dicts(valid_results)

    return (run_many,)


@app.cell(hide_code=True)
def _(
    BaseChatModel,
    RateLimitError,
    SYSTEM_PROMPT,
    fetch_image_data,
    logger,
    tc,
):
    @tc.retry(
        wait=tc.wait_fixed(1),
        stop=tc.stop_after_attempt(10),
        before_sleep=logger.warning,
    )
    @tc.retry(
        wait=tc.wait_fixed(70),
        stop=tc.stop_after_attempt(5),
        retry=tc.retry_if_exception_type(RateLimitError),
        before_sleep=logger.warning,
    )
    async def generate_analysis(
        llm: BaseChatModel,
        image_url: str,
        model_name: str,
    ):
        image_data = fetch_image_data(image_url)
        model_name_normalized = model_name.replace(":free", "")
        system_message = {
            "role": "system",
            "content": "\n\n".join(
                [
                    # This line enables per-model cache.
                    f"Your name is `{model_name_normalized}`.",
                    SYSTEM_PROMPT,
                ]
            ),
        }
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                    },
                },
            ],
        }
        return await llm.ainvoke(
            [
                system_message,
                user_message,
            ],
            # config={"callbacks": [LangfuseCallbackHandler()]},
        )

    return (generate_analysis,)


@app.cell(hide_code=True)
def _(compute_category_options, pyd, ty, vispub_df):
    # Construct signature of structured output
    purpose_options = {"schematic", "gui", "vis"}
    PurposeType = ty.Literal[*purpose_options]

    encoding_options = (
        compute_category_options(vispub_df, "encoding_type") - purpose_options
    )
    EncodingType = ty.Literal[*encoding_options]

    dimensionality_options = compute_category_options(vispub_df, "dim_type")
    DimensionalityType = ty.Literal[*dimensionality_options]

    class Analysis(pyd.BaseModel):
        purpose: PurposeType | None = pyd.Field(
            description="Primary purpose of the visualization. None/null if unable to tell."
        )
        encoding: list[EncodingType | None] = pyd.Field(
            description="Encoding(s) appearing in the visualization. None/null if unable to tell."
        )
        dimensionality: list[DimensionalityType | None] = pyd.Field(
            description="Dimensionaliti(es) of the visualization. None/null if unable to tell."
        )

    return (Analysis,)


@app.cell(hide_code=True)
def _(BaseChatModel, init_chat_model, os):
    def init_openrouter_model(model_name: str, **kwargs) -> BaseChatModel:
        model = init_chat_model(
            "openai:gpt-dummy",
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        model.model_name = model_name
        return model

    def init_llm(model_name: str, **kwargs) -> BaseChatModel:
        if model_name.startswith("openrouter:"):
            return init_openrouter_model(
                model_name.replace("openrouter:", ""),
                **kwargs,
            )

        return init_chat_model(model_name, **kwargs)

    return (init_llm,)


@app.cell(hide_code=True)
def _(base64, client, pl):
    def compute_category_options(df: pl.DataFrame, col: str) -> set[str]:
        return set(
            (
                df.select(pl.col(col).str.split(";"))
                .explode(col)
                .unique(col)
                .filter(pl.col(col).str.strip_chars(" ").str.len_chars() != 0)
                .sort(col)
            )
            .to_numpy()
            .squeeze()
            .tolist()
        )

    def fetch_image_data(url: str) -> str:
        res = client.get(url, extensions={"force_cache": True})
        res.raise_for_status()
        image_bytes = res.content
        return base64.b64encode(image_bytes).decode("utf-8")

    return compute_category_options, fetch_image_data


@app.cell(hide_code=True)
def _(pathlib):
    # Constants
    SYSTEM_PROMPT = pathlib.Path("src/sysprompt.md").read_text()
    MODELS = [
        "google_genai:gemini-2.0-flash",
        "google_genai:gemini-2.5-flash-preview-05-20",
        "google_genai:gemini-2.5-pro-preview-05-06",
        "openai:gpt-4.1-nano",
        "openai:gpt-4.1-mini",
        "openai:gpt-4.1",
        "openai:o4-mini",
        "openrouter:meta-llama/llama-4-scout",
        "openrouter:meta-llama/llama-4-maverick",
        "openrouter:qwen/qwen2.5-vl-32b-instruct",
        "openrouter:mistralai/mistral-small-3.1-24b-instruct",
        "openrouter:mistralai/mistral-medium-3",
        "openrouter:mistralai/pixtral-large-2411",
    ]
    DEFAULT_MODEL = MODELS[0]
    MAX_CONCURRENT_TASKS = 5
    return DEFAULT_MODEL, MAX_CONCURRENT_TASKS, MODELS, SYSTEM_PROMPT


@app.cell(hide_code=True)
def _(logging):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    return (logger,)


@app.cell(hide_code=True)
def _():
    # Standard library imports
    import asyncio
    import base64
    import os
    import pathlib
    import typing as ty
    import logging

    # Third-party imports
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import pydantic as pyd
    import tenacity as tc
    from langchain.chat_models import init_chat_model
    from langchain_core.language_models import BaseChatModel
    from openai import RateLimitError

    # Cache Config
    pathlib.Path(".cache").mkdir(exist_ok=True)

    # Cache HTTP responses so that we do not fetch images from remote twice
    import hishel
    import sqlite3

    storage = hishel.SQLiteStorage(connection=sqlite3.connect(".cache/hishel.db"))
    client = hishel.CacheClient(storage=storage)

    # Cache LLM responses so that we do not run the same inference twice
    from langchain_core.globals import set_llm_cache
    from langchain_community.cache import SQLiteCache

    set_llm_cache(SQLiteCache(database_path=".cache/langchain.db"))
    return (
        BaseChatModel,
        RateLimitError,
        asyncio,
        base64,
        client,
        cs,
        init_chat_model,
        logging,
        mo,
        os,
        pathlib,
        pl,
        pyd,
        tc,
        ty,
    )


if __name__ == "__main__":
    app.run()
