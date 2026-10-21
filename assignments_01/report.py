import json
from pathlib import Path

from weatherkit import DailyAggregator, WeatherResponse, to_readings

DATA_PATH = Path(__file__).parent / "weather_raw.json"


def load_response(path: Path) -> WeatherResponse:
    """Load and validate an Open-Meteo hourly response from a JSON file."""
    return WeatherResponse.model_validate(json.loads(path.read_text()))


def main() -> None:
    response = load_response(DATA_PATH)
    print(f"Latitude {response.latitude}, timezone {response.timezone}, "
          f"{len(response.hourly.time)} hourly observations\n")

    readings = to_readings(response)
    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)

    print(f"{'Date':<12}{'High':>7}{'Low':>7}{'Precip':>9}{'Range':>8}")
    for s in summaries:
        print(f"{s.date:<12}{s.temp_max:>7.1f}{s.temp_min:>7.1f}"
              f"{s.precipitation_sum:>9.1f}{s.temp_range():>8.1f}")

    dropped = aggregator.incomplete_days(readings)
    if dropped:
        print(f"\nWARNING: dropped incomplete days: {', '.join(dropped)}")
    else:
        print("\nNo incomplete days dropped.")


# Without this guard, main() would run as a side effect of
# "from report import load_response": importing the module would load the
# file, run the whole pipeline, and print the table, even though the caller
# only wanted the helper.
if __name__ == "__main__":
    main()


# --- Reflection ---
#
# 1. Rejecting the whole file over one null is right when a wrong number is
#    worse than no number, e.g. a report or model that must be built from
#    complete data, where a silent gap would skew a daily max/min. You'd rather
#    tolerate the gap for something like a live dashboard, where 167 good
#    hours are still useful and failing outright shows nothing. To tolerate it,
#    change the fields to list[float | None] and have to_readings and the
#    aggregator skip the None values.
#
# 2. At noon the current day only has about 12 hours, so with min_hours=24 it
#    gets dropped, and today never appears in the report until the day is
#    over. If the pipeline didn't drop it, today's "max" and "min" would only
#    cover the morning and look like real daily values. incomplete_days()
#    lists the dropped date, so the gap is visible and logged instead of the
#    day silently disappearing.
#
# 3. A Week 10 pipeline can import exactly the piece it needs, like
#    "from weatherkit import WeatherResponse, to_readings", without copying
#    code or running a script. Each module can also be tested and changed on
#    its own without touching the others.
