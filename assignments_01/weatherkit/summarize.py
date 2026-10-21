from dataclasses import dataclass

from weatherkit.records import HourlyReading


@dataclass
class DailySummary:
    """Weather for one calendar day. Temperatures in C, precipitation in mm."""
    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the difference between the day's high and low."""
        return self.temp_max - self.temp_min


class DailyAggregator:
    """Groups hourly readings into daily summaries."""

    def __init__(self, min_hours: int = 24) -> None:
        """
        Args:
            min_hours: Minimum hourly observations required to report a day.
        """
        self.min_hours = min_hours

    def _group_by_date(self, readings: list[HourlyReading]) -> dict[str, list[HourlyReading]]:
        """Group readings by the date portion (first 10 characters) of the timestamp."""
        groups: dict[str, list[HourlyReading]] = {}
        for reading in readings:
            groups.setdefault(reading.timestamp[:10], []).append(reading)
        return groups

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Summarize readings per day, dropping days with fewer than min_hours.

        Args:
            readings: Hourly readings, in any order.

        Returns:
            One DailySummary per complete day, sorted by date.
        """
        summaries = []
        for date, day in sorted(self._group_by_date(readings).items()):
            if len(day) < self.min_hours:
                continue
            temps = [r.temperature_c for r in day]
            summaries.append(DailySummary(
                date=date,
                temp_max=max(temps),
                temp_min=min(temps),
                precipitation_sum=sum(r.precipitation_mm for r in day),
                hours_observed=len(day),
            ))
        return summaries

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return the dates that summarize() drops for having too few hours.

        Args:
            readings: Hourly readings, in any order.

        Returns:
            The dropped dates, sorted.
        """
        return sorted(
            date for date, day in self._group_by_date(readings).items()
            if len(day) < self.min_hours
        )
