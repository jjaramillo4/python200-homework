"""Week 7 warmup: tool-based agents, the ReAct loop, and smolagents.

Run this from inside assignments_07/ so that resources/bike_commute.csv is found:
    python warmup_07.py

Paths are resolved relative to this file, so running it from elsewhere also works.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

# Agg = non-interactive backend. Without it, every plt.show() in a plain script
# blocks until I close the window, which stalls the agent mid-ReAct-loop.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, ToolCallingAgent, tool

BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")

client = OpenAI()
api_key = os.getenv("OPENAI_API_KEY")
print("OpenAI client created.\n")


# =====================================================================
# --- Lesson 02: Tool definitions and the ReAct loop ---
# =====================================================================

# Q1 ------------------------------------------------------------------
print("=" * 70)
print("Q1: tool function + its JSON schema")
print("=" * 70)


def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


# The model never sees the Python above. It only sees this JSON description,
# so the "description" fields are what actually drive tool selection.
celsius_tool_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": (
            "Convert a temperature from degrees Celsius to degrees Fahrenheit. "
            "Use this whenever the user gives a Celsius value and wants it in Fahrenheit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "The temperature in degrees Celsius, e.g. 37 or -40.",
                }
            },
            "required": ["celsius"],
        },
    },
}

# Direct calls, no agent involved yet.
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))
print(json.dumps(celsius_tool_schema, indent=2))


# Q2 ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q2: agent with ONLY get_current_time, asked a conversion question")
print("=" * 70)


def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current local time as a string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]


def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for a single user prompt."""

    SYSTEM_PROMPT = """You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time."""

    # Step 1: start the conversation with system and user messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    # The lesson prints the whole ChatCompletion object here. finish_reason is the
    # only part I actually need to see which branch of the loop we took.
    print("First response finish_reason:", first_response.choices[0].finish_reason)
    first_message = first_response.choices[0].message

    messages.append(
        {
            "role": "assistant",
            "content": first_message.content,
            "tool_calls": first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            if function_name == "get_current_time":
                tool_result = get_current_time()
            else:
                tool_result = f"Error: unknown tool {function_name}."

            print("Tool called:", function_name)
            print("Tool result:", tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )
        print("Second response finish_reason:", second_response.choices[0].finish_reason)
        return second_response.choices[0].message.content or ""
    else:
        print("No tools needed....")

    return first_message.content or ""


# --- My prediction, written before running it ---
#
# Q: Will run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call?
# A: No. The only tool registered is get_current_time, and its description says
#    nothing about temperature. tool_choice='auto' means the model reads the tool
#    descriptions and reasons about whether any of them help. None do, and the
#    conversion is trivial arithmetic the model can do from its own weights, so it
#    should answer directly with 212°F.
#
#    The risk I see: the system prompt tells it to use get_current_time "whenever a
#    user asks about the time", and a temperature question is not a time question,
#    so nothing pushes it toward a wrong call.
#
# Q: How many API calls?
# A: One. The second call only happens when finish_reason == 'tool_calls'.

response_q2 = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print("Q2 response:", response_q2)
# Prediction check: see "No tools needed...." above and a single finish_reason line
# (= one API call). The model answered 212°F from its own reasoning. Prediction held.


# Q3 ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q3: agent with BOTH tools")
print("=" * 70)

# Deliberate shadowing: these two names reuse `tools` and `run_agent` from Q2.
# Python rebinds the module-level names here, so every call below uses the
# two-tool version. Q2's calls already ran above, so they saw the old version.
# (Java has no equivalent: you cannot redefine a method mid-class.)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current local time as a string.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    celsius_tool_schema,
]

# Dispatch table instead of an if/elif chain: adding a tool is now one dict entry
# plus one schema entry, and an unknown name is handled in one place.
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "celsius_to_fahrenheit": celsius_to_fahrenheit,
}


def run_agent(user_prompt: str) -> str:
    """Run a ReAct-style agent that can reach for either of two tools."""

    SYSTEM_PROMPT = (
        "You are a simple assistant. You can tell the current time and convert "
        "Celsius temperatures to Fahrenheit. Use a tool when one fits the question, "
        "and answer directly when none does."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    print("First response finish_reason:", first_response.choices[0].finish_reason)
    first_message = first_response.choices[0].message

    messages.append(
        {
            "role": "assistant",
            "content": first_message.content,
            "tool_calls": first_message.tool_calls,
        }
    )

    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments or "{}")

            function_to_call = TOOL_FUNCTIONS.get(function_name)
            if function_to_call is None:
                tool_result = f"Error: unknown tool {function_name}."
            else:
                # The model supplies arguments as JSON, so **kwargs by name, never
                # positionally. A positional call here would silently misbind args.
                tool_result = function_to_call(**function_args)

            print(f"Tool called: {function_name}({function_args})")
            print("Tool result:", tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )
        print("Second response finish_reason:", second_response.choices[0].finish_reason)
        return second_response.choices[0].message.content or ""

    print("No tools needed....")
    return first_message.content or ""


response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# A tool WAS called: celsius_to_fahrenheit with celsius=37. The question matches the
# schema description almost word for word, and the model cannot be sure of the
# arithmetic the way a function is, so it delegates.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# No tool was called. "In plain English" is a request for an explanation, not a
# conversion. Neither tool description covers it, so the model answers from its own
# knowledge. This is the same branch Q2 took, for the same reason.


# =====================================================================
# --- Lesson 03: Multi-tool agent ---
# =====================================================================


class CsvManager:
    """Holds the loaded DataFrame so each tool call does not reload the file."""

    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        series = self.df[column]
        summary = series.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.

        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"

        if x == y:
            x = None

        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            self._save_figure(f"{plot_type}_{y}_vs_index.png")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        self._save_figure(f"{plot_type}_{y}_vs_{x}.png")
        plt.show()

        return f"Plotted {y} vs {x} as a {plot_type}."

    def _save_figure(self, filename: str):
        """Write the current figure to outputs/ (plt.show() is a no-op under Agg)."""
        plt.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
        plt.close()

    # Q4 --------------------------------------------------------------

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error:
            return error

        missing = [c for c in (col1, col2) if c not in self.df.columns]
        if missing:
            return {
                "error": f"These columns are not in the data: {missing}. "
                f"Options: {self.df.columns.tolist()}"
            }

        # dropna on the pair: pearsonr raises on NaN, and dropping per column
        # would misalign the two series against each other.
        pair = self.df[[col1, col2]].dropna()
        if len(pair) < 2:
            return {"error": "Need at least two complete rows to correlate."}

        pearson_r, p_value = pearsonr(pair[col1], pair[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(pearson_r), 4),
            "p_value": round(float(p_value), 4),
        }


print("\n" + "=" * 70)
print("Q4: CsvManager with a compute_correlation tool added")
print("=" * 70)

csv_manager = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_manager.list_csv_files,
    "load_csv": csv_manager.load_csv,
    "get_columns": csv_manager.get_columns,
    "summarize_columns": csv_manager.summarize_columns,
    "describe_column": csv_manager.describe_column,
    "plot_data": csv_manager.plot_data,
    "compute_correlation": csv_manager.compute_correlation,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name to describe."}
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            # Spelling out "Pearson", "coefficient" and "p-value" is what stops the
            # flailing in the lesson: the model now has a name for what it wants.
            "description": (
                "Compute the Pearson correlation coefficient and its p-value between two "
                "numeric columns of the active CSV. Use this for any question about "
                "correlation, linear relationship, or statistical significance between "
                "two columns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "First column name."},
                    "col2": {"type": "string", "description": "Second column name."},
                },
                "required": ["col1", "col2"],
            },
        },
    },
]

print("Direct call, no agent:", csv_manager.load_csv("bike_commute.csv"))
print(
    "Direct call, no agent:",
    csv_manager.compute_correlation("avg_traffic_density", "avg_speed_kmh"),
)
# Reset so Q5 starts from a clean slate and the agent has to load the file itself.
csv_manager.df = None
csv_manager.csv_name = None


def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.

    REASON: call the model with the conversation so far.
    ACT:    if tools are requested, run the Python functions.
    OBSERVE: append each tool result back into the conversation history.

    Stops when the model answers with no tool calls, or at max_tool_rounds.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """Wrap a tool's return value as a 'tool' message the model can read."""
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    for loop_idx in range(max_tool_rounds):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}

            messages.append(observe_tool_result(tool_call.id, result))

    return "I hit the tool-round limit. Try a simpler request."


SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)


# Q5 ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q5: the query that hit the tool-round limit in the lesson")
print("=" * 70)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.",
)
print("Q5 response:", result)
# In the lesson this ran load_csv, describe_column, summarize_columns, get_columns,
# load_csv again and then gave up. Two rounds now: load_csv, then compute_correlation.
# The agent did not get smarter; the toolbox got a tool that answers the question.


# Q6 ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q6: the full conversation, role by role")
print("=" * 70)

# What each role is doing in the ReAct loop:
#   system    - the standing instructions. Sent once, read by the model on every
#               round. This is where "use the tools, do not guess" lives.
#   user      - the human turn. run_agent_cycle appends one of these per query.
#   assistant - the model's turn. Either content (a final answer, which ends the
#               loop) or a tool_calls list (the ACT step: "run this for me").
#               Note the assistant message with tool_calls has content=None.
#   tool      - the OBSERVE step: the return value of one Python call, tied back to
#               its request by tool_call_id. This is the only way a result ever
#               reaches the model; it cannot see my variables.
# Read in order, the list is the whole audit trail of the loop.
print(json.dumps(messages, indent=2, default=str))


# =====================================================================
# --- Lesson 04: smolagents ---
# =====================================================================

# Q7 ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q7: the same tool, declared the smolagents way")
print("=" * 70)


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Use this for any question about correlation, linear relationship, or statistical
    significance between two columns of the active CSV.

    Args:
        col1: Name of the first numeric column, e.g. "avg_traffic_density".
        col2: Name of the second numeric column, e.g. "avg_speed_kmh".

    Returns:
        A dict with "col1", "col2", "pearson_r" and "p_value", or an error dict.
    """
    return csv_manager.compute_correlation(col1, col2)


print(compute_correlation.description)
print("inputs:", compute_correlation.inputs)
print("output_type:", compute_correlation.output_type)

# Comparing this to the hand-written schema in Q4:
#
# smolagents builds the whole schema by introspection. The function NAME becomes the
# tool name, the TYPE HINTS become the JSON types (col1: str -> {"type": "string"},
# -> dict becomes output_type "object"), the docstring summary becomes .description,
# and each "Args:" line becomes that parameter's description. A parameter with no
# default lands in required automatically.
#
# So what it still needs from me is exactly the part a machine cannot infer: the
# natural language. A precise function name, a type hint on every argument and the
# return, a summary line saying WHEN to reach for this tool (not just what it
# computes), and one Args line per parameter. If I write a vague docstring,
# smolagents produces a vague schema and the model picks the wrong tool. The
# boilerplate is gone; the writing that makes the tool findable is not.
#
# Worth knowing: .description is the docstring with the Args and Returns sections
# stripped out, so the summary AND the "use this for..." paragraph both land in it.
# The per-argument text is carried separately in .inputs, and smolagents assembles
# both into the system prompt.


# Q8 ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q8: ToolCallingAgent vs CodeAgent on a styling request")
print("=" * 70)


@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns).

    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis.
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        A short success message string, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)


TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

TOOL_INSTRUCTIONS = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

CODE_INSTRUCTIONS = f"""
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- IMPORTANT: If the user requests plot styling (color, marker, title text, labels, grid, etc.)
  that the plot_data tool cannot control, DO NOT call plot_data.
  Instead, write matplotlib code directly so the plot matches the request.
  If code execution fails, do not fall back to plot_data when the user requested styling.
  Explain what failed and what you would need to proceed.
- matplotlib runs on a non-interactive backend, so plt.show() displays nothing.
  Always finish a plot with plt.savefig(r"{OUTPUT_DIR}\\<name>.png") instead.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""

tool_agent = ToolCallingAgent(tools=TOOLS, model=model, instructions=TOOL_INSTRUCTIONS)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

print("\n--- ToolCallingAgent ---")
response_tool = tool_agent.run(prompt)
print("Tool agent response:", response_tool)

print("\n--- CodeAgent ---")
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
print("Code agent response:", response_code)

# What each one actually produced:
#
# ToolCallingAgent: in one step it called load_csv, then plot_data(y="avg_heart_rate",
# x="duration_min", plot_type="scatter"). plot_data has no colour parameter, so the
# dots came out matplotlib default blue. It then answered:
#
#   "Scatter plot of avg_heart_rate vs duration_min has been successfully created
#    with green dots."
#
# It did not drop the colour quietly, it asserted it. That is worse than the lesson's
# version, where the agent only failed to mention the colour. The reason is that the
# agent's entire view of the world is the tool's return string, "Plotted
# avg_heart_rate vs duration_min as a scatter." Nothing in that string contradicts
# "green", so the model fills the gap from the prompt it was given. It cannot see
# the figure. There was no call available to it that would have worked.
#
# CodeAgent: used load_csv for the loading (a tool covers it), then in step 2 wrote
# its own matplotlib against csv_manager.df with color='green' and saved the file.
# Step 3 was just final_answer. Same toolbox, same model, different outcome: the gap
# in the tools got filled by generated code instead of invented prose.
#
# What this says about when each is the right choice: the ToolCallingAgent can only
# ever do the cross product of what I wrote. That is a ceiling, and it is also a
# guarantee: whatever it does, I wrote and tested. The CodeAgent has no such ceiling,
# which is what you want for open-ended analysis, and no such guarantee, which is
# why it needs the restricted interpreter and the authorized-imports allowlist.
# Fixed, known operations -> tool agent. Unpredictable one-off asks -> code agent.


# Q9 ------------------------------------------------------------------
#
# 1. When is a ToolCallingAgent the better choice?
#
#    Any task where the set of legal actions is known in advance and the cost of an
#    unexpected action is high. My day-job example: a Kintone agent that updates
#    sales records. The verbs are a closed list (look up a record, append a memo,
#    move a status). Each one is a reviewed function with validation in it, and the
#    agent's only freedom is which to call and with what arguments.
#
#    The property that makes it a fit is that the action space is closed and the
#    actions have side effects. A CodeAgent there could write a correct-looking
#    update that skips my validation and writes to the wrong app, and I would not
#    find out until the data was already wrong. The tool agent cannot: it can only
#    pick from the list. The ceiling is the feature.
#
# 2. One meaningful risk of a CodeAgent that does not apply to a ToolCallingAgent:
#
#    The CodeAgent executes text that an LLM wrote, at my privilege level. The
#    ToolCallingAgent only ever chooses a name and fills in arguments; the code that
#    runs is code I wrote. So the CodeAgent's failure modes include ones no tool
#    agent has: code that runs cleanly and returns a confidently wrong number (a
#    silent off-by-one in a filter, averaging an already-averaged column), and code
#    that touches things outside the question, like the extra loop in the lesson
#    that saved a figure nobody asked for. Scale that to a filesystem write or an
#    API call and it is no longer a tidiness problem.
#
#    smolagents' restricted interpreter and additional_authorized_imports cut down
#    the blast radius, but they are a sandbox around the arbitrary execution, not a
#    replacement for it. Wrong-but-plausible arithmetic passes every sandbox check
#    there is.

print("\nWarmup complete. Plots written to", OUTPUT_DIR)
