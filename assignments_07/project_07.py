"""Week 7 mini-project: a World Happiness analysis agent built on smolagents.

Four tools wrap the dataset (load, summarize, correlate, rank). A CodeAgent decides
which to call, and writes its own matplotlib when no tool covers the request.

    python project_07.py
"""

import os
from pathlib import Path

import matplotlib

# Agg = non-interactive backend. The agent's generated code calls plt.show(), which
# would block the script on a GUI backend until I closed each window by hand.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("happiness_project")
OUTPUT_DIR = Path("outputs")

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")
api_key = os.getenv("OPENAI_API_KEY")

# The yearly files disagree on what the score column is called, and none of them
# carry a year at all. Everything downstream expects the renamed columns.
COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",  # the 2024 file uses this name
    "GDP per capita": "gdp_per_capita",
}

# Shared state. `df` is what the tools read and write; DATA is the same frame handed
# to the agent's sandbox (see additional_args below), because generated code runs in
# a separate namespace and cannot see this module's globals.
df = None
DATA = {"df": None}


# =====================================================================
# Task 1: the tools
# =====================================================================


@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Reads every yearly CSV in happiness_project/ (2015-2024) and merges them into one
    DataFrame. The files are semicolon-separated and use a comma as the decimal point,
    and the year is taken from each file name since no file has a year column. Column
    names are normalized to country, region, happiness_score, gdp_per_capita and year.
    Call this once before any other tool.

    Returns:
        A dict with "shape" (rows, columns) and "columns" (the column names), or an
        error dict if no files were found.
    """
    global df

    paths = sorted(DATA_DIR.glob("world_happiness_*.csv"))
    if not paths:
        return {"error": f"No world_happiness_*.csv files found in {DATA_DIR.resolve()}."}

    frames = []
    for path in paths:
        year_df = pd.read_csv(path, sep=";", decimal=",")
        year_df = year_df.rename(columns=COLUMN_NAMES)
        year_df["year"] = int(path.stem.split("_")[-1])  # "world_happiness_2020" -> 2020
        frames.append(year_df)

    df = pd.concat(frames, ignore_index=True)
    DATA["df"] = df

    return {"shape": list(df.shape), "columns": df.columns.tolist()}


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Gives count, mean, std, min, the quartiles and max for a numeric column, or
    count, unique, top and freq for a text column.

    Args:
        column: Name of the column to summarize, e.g. "happiness_score".

    Returns:
        A dict of descriptive statistics, or an error dict if no data is loaded or
        the column does not exist.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}

    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Use this for any question about correlation, linear relationship or statistical
    significance between two columns of the dataset.

    Args:
        col1: Name of the first numeric column, e.g. "gdp_per_capita".
        col2: Name of the second numeric column, e.g. "happiness_score".

    Returns:
        A dict with "col1", "col2", "pearson_r" and "p_value" rounded to 4 decimals,
        or an error dict if no data is loaded or a column is missing or not numeric.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}

    missing = [c for c in (col1, col2) if c not in df.columns]
    if missing:
        return {"error": f"Not a column: {missing}. Options: {df.columns.tolist()}"}

    # Drop NaN on the pair, not per column: pearsonr raises on NaN, and dropping
    # separately would leave the two series misaligned row for row.
    pair = df[[col1, col2]].dropna()
    if len(pair) < 2:
        return {"error": "Need at least two complete rows to correlate."}

    try:
        pearson_r, p_value = pearsonr(pair[col1], pair[col2])
    except (TypeError, ValueError) as err:
        return {"error": f"Could not correlate those columns: {err}"}

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(pearson_r), 4),
        "p_value": round(float(p_value), 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Filters the dataset to one year, sorts by the column in descending order and
    returns the leaders. Use this for "top", "highest" or "best" ranking questions.

    Args:
        column: Column to rank by, e.g. "happiness_score" or "gdp_per_capita".
        year: The year to filter to, between 2015 and 2024.
        n: How many countries to return. Defaults to 5.

    Returns:
        A dict with "column", "year" and "top_countries" (a list of dicts, each with
        "country" and the ranked value), or an error dict on bad input.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}
    if n < 1:
        return {"error": "n must be at least 1."}

    year_rows = df[df["year"] == year]
    if year_rows.empty:
        available = sorted(int(y) for y in df["year"].unique())
        return {"error": f"No rows for {year}. Years available: {available}"}

    ranked = year_rows.sort_values(column, ascending=False).head(n)
    return {
        "column": column,
        "year": year,
        "top_countries": ranked[["country", column]].to_dict(orient="records"),
    }


# =====================================================================
# Task 2: the agent
# =====================================================================

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = f"""
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).

When you write your own code, the merged DataFrame is available as data["df"] once
load_happiness_data has run. Its columns include country, region, happiness_score,
gdp_per_capita and year.

matplotlib runs on a non-interactive backend here, so plt.show() displays nothing.
Always save a figure with plt.savefig("{OUTPUT_DIR.as_posix()}/<name>.png") instead,
and say where you saved it.

Every step must be a code block. When you have the answer, end the turn by calling
final_answer("...") inside the code block. Writing the answer as prose with no code
block does not end the run, it just burns a step.

Import every library you use before you use it (for example: import pandas as pd).
Only pandas, matplotlib.pyplot and scipy.stats are available.

Be concise and student-friendly in your responses, and only claim you did something
if the tool or the code actually did it.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


def ask(query: str) -> str:
    """Run one query with the conversation kept (reset=False) and the frame shared."""
    print(f"\n--- Query: {query} ---")
    response = agent.run(query, reset=False, additional_args={"data": DATA})
    print(response)
    return response


if __name__ == "__main__":
    # The agent writes relative paths like "outputs/happiness_by_region.png" into its
    # generated code, so anchor the working directory to this file's folder.
    os.chdir(BASE_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Task 3: the five guided queries ---
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. "
        "Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        ask(query)

    plot_path = OUTPUT_DIR / "happiness_by_region.png"
    print(f"\nPlot saved: {plot_path.exists()} ({plot_path.resolve()})")

    # --- Task 4: my own queries ---

    # Query 1: no tool ranks change over time, so this one has to be code. It needs a
    # groupby on two years and a subtraction, which is exactly the gap a CodeAgent fills.
    my_query_1 = (
        "Which region's average happiness_score changed the most between 2015 and 2024, "
        "and by how much? Show me the three biggest movers in either direction."
    )
    response_1 = ask(my_query_1)
    # Comment: code generation only, no tool calls, in two steps. It grouped data["df"]
    # by region for 2015 and for 2024, subtracted, and read "biggest movers in either
    # direction" as .abs().nlargest(3) while printing the signed values. Answer: South
    # Asia -0.6843, Central and Eastern Europe +0.6627, North America and ANZ -0.3524.

    # Query 2: deliberately the opposite shape. Every part of this maps onto a tool,
    # so I expect tool calls only, and no generated code.
    my_query_2 = (
        "Summarize the gdp_per_capita column, then show me the top 5 countries by "
        "gdp_per_capita in 2018. Are the richest countries also the happiest that year?"
    )
    response_2 = ask(my_query_2)
    # Comment: both, and my prediction of "tools only" was wrong. summarize_column and
    # get_top_n_countries covered the first two parts in one step, but the "are they
    # also the happiest" part has no tool, so it wrote code to pull happiness_score for
    # those five countries. That code failed twice: get_top_n_countries returns a dict,
    # and iterating a dict yields its string keys, so country["country"] raised
    # "string indices must be integers". It recovered by hardcoding the five country
    # names it had already seen printed. Correct answer, brittle route.

    print("\nProject complete. Outputs in", OUTPUT_DIR.resolve())


# =====================================================================
# --- Reflection ---
#
# 1. In Query 3, how did the agent report the correlation?
#
#    It called compute_correlation once, got {'pearson_r': 0.6313, 'p_value': 0.0},
#    and answered: "The correlation coefficient between GDP per capita and happiness
#    score is approximately 0.6313 with a statistically significant p-value of
#    0.0000."
#
#    Half sensible. It used the tool instead of guessing, it kept the number it was
#    given, and it did answer the "is it significant" part. What it did not do is
#    interpret: no "moderate", no "positive", no "as GDP rises, happiness tends to
#    rise". It handed back the raw statistic and left the reading of it to me, which
#    for a student-facing assistant is the weaker half of the job.
#
#    Two things I would not trust as printed. The p-value of 0.0000 is an artifact of
#    my own tool: I round to 4 decimals, so any p below 0.00005 prints as zero. The
#    true value is not zero, and a reader could take "0.0000" as certainty rather
#    than as "too small to show". I would return the p-value in scientific notation,
#    or a "significant at alpha=0.05" boolean, rather than a rounded float. Second,
#    r = 0.63 is pooled over all ten years, so the same country appears up to ten
#    times. The rows are not independent, which is exactly the assumption the p-value
#    rests on. The agent had no way to know that, because nothing in my tool says it.
#
# 2. Did any of the agent's responses surprise you?
#
#    Two, in opposite directions.
#
#    More capable than expected: my own query about which region moved most between
#    2015 and 2024. No tool covers change over time, so it wrote the pandas itself in
#    one pass: filter to 2015, filter to 2024, group by region, mean, subtract. The
#    part I did not expect was that it read "biggest movers in either direction"
#    correctly, ranking by .abs().nlargest(3) while reporting the signed change. That
#    is a reading of ambiguous English, not a formula.
#
#    Less capable than expected: in the next query it had already printed the top 5
#    GDP countries for 2018, then tried to reuse that result and crashed twice on
#    country["country"], because get_top_n_countries returns a dict and iterating a
#    dict gives you its keys. It never fixed the iteration. It hardcoded the five
#    country names it could see in its own earlier output and moved on. The answer
#    was right; the method would break the moment the data changed.
#
#    The honest version of both is that the shape of my tool's return value decided
#    how well the agent did, more than the agent's reasoning did.
#
#    Worth recording from the first run, before I tightened the system prompt: on the
#    regional plot query, gpt-4o-mini produced the correct chart on step 1 and then
#    burned every remaining step rewriting it, because it kept answering in prose
#    instead of calling final_answer() in a code block. It hit max_steps on a query it
#    had already solved. Adding one line to SYSTEM_PROMPT ("end the turn by calling
#    final_answer(...) inside the code block") fixed it, and the same five queries now
#    run clean. The agent did not get better; the instructions got more specific.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#
#    compare_years(column, year_a, year_b, group_by="region"): group the data by
#    country or region, take the mean of the column in each of the two years, and
#    return the change per group sorted by size, with the count of rows behind each
#    mean.
#
#    It would answer the whole family of questions this dataset invites and none of my
#    four tools touch: which regions improved, who fell furthest, did the gap between
#    the top and bottom region widen. Right now every one of those is generated code,
#    which is where the errors above came from.
#
#    The row count matters more than it looks. This is not a fixed panel: the files
#    carry 158 countries in 2015 and 140 in 2024, and 26 countries in the 2015 file
#    are absent from the 2024 one. So part of the -0.68 I reported for South Asia
#    above is countries leaving the ranking, not people getting unhappier. A tool
#    that returns n next to the delta puts that in front of the agent instead of
#    leaving it for the reader to catch.
# =====================================================================
