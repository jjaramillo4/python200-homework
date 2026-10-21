import pytest

from weatherkit import DailyAggregator, HourlyReading


@pytest.fixture
def readings():
    """24 full hours on 2026-04-08, plus 3 hours on 2026-04-09."""
    day_one = [
        HourlyReading(f"2026-04-08T{h:02d}:00", 10.0 + h * 0.5, 0.1 if h < 3 else 0.0)
        for h in range(24)
    ]
    day_two = [
        HourlyReading("2026-04-09T00:00", 5.0, 0.0),
        HourlyReading("2026-04-09T01:00", 4.0, 1.0),
        HourlyReading("2026-04-09T02:00", 6.0, 0.5),
    ]
    return day_one + day_two


def test_groups_into_two_dates(readings):
    summaries = DailyAggregator(min_hours=1).summarize(readings)
    assert [s.date for s in summaries] == ["2026-04-08", "2026-04-09"]


def test_max_and_min(readings):
    day_one = DailyAggregator().summarize(readings)[0]
    assert day_one.temp_max == 21.5
    assert day_one.temp_min == 10.0
    assert day_one.temp_range() == 11.5


def test_precipitation_sum(readings):
    day_one = DailyAggregator().summarize(readings)[0]
    assert day_one.precipitation_sum == pytest.approx(0.3)


def test_short_day_is_dropped(readings):
    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)
    assert [s.date for s in summaries] == ["2026-04-08"]
    assert aggregator.incomplete_days(readings) == ["2026-04-09"]


@pytest.mark.parametrize("min_hours, expected_days", [
    (24, 1),
    (3, 2),
    (1, 2),
])
def test_min_hours_is_respected(readings, min_hours, expected_days):
    summaries = DailyAggregator(min_hours=min_hours).summarize(readings)
    assert len(summaries) == expected_days


def test_lowering_min_hours_keeps_short_day(readings):
    aggregator = DailyAggregator(min_hours=3)
    summaries = aggregator.summarize(readings)
    assert summaries[1].date == "2026-04-09"
    assert summaries[1].hours_observed == 3
    assert summaries[1].temp_max == 6.0
    assert aggregator.incomplete_days(readings) == []


# Deliberate break: in DailyAggregator._group_by_date I changed
# reading.timestamp[:10] to reading.timestamp, so every hour became its own
# "day". test_groups_into_two_dates caught it first (it got 27 one-hour groups
# instead of two dates), and 7 more tests in this file failed with it.
# Restored afterwards; all tests pass.