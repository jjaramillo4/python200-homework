# Week 1 warmups -- written without AI, per the course README.
# Run the pytest section with:  pytest warmup_01.py -v

from dataclasses import dataclass

# --- Classes ---

# Q1
# Write a class Thermometer that stores a list of temperature readings in Celsius.
# - __init__ takes a location (str) and an optional list of readings that
#   defaults to an empty list.
# - add(reading) appends one reading.
# - average() returns the mean of the readings, or None if there are none.
# - hottest() returns the highest reading, or None if there are none.
# Create a Thermometer for a location of your choice, add at least four
# readings, and print the average and the hottest.
# Comment: why does average() need to handle the empty case? What would
# happen without that check?

class Thermometer:
    """
   Stores temperature readings in Celsius for one location.
    """

    def __init__(self, location, readings = None):
        self.location = location
        self.readings = readings if readings is not None else []

    def __repr__(self):
        avg = self.average()
        avg = round(avg, 1) if avg is not None else None
        return f"Thermometer(location='{self.location}', n_readings={len(self.readings)}, average={avg})"
    def add(self,reading) -> None:
        self.readings.append(reading)

    def average(self) -> float | None:
        if not self.readings: # Conditional needed to prevent division by zero error when calculating the average of an empty list.
            return None
        return sum(self.readings) / len(self.readings)
    
    def hottest(self) -> float | None:
        if not self.readings:
            return None
        return max(self.readings)

My_thermometer = Thermometer("Charlotte")
My_thermometer.add(20.0)
My_thermometer.add(18.0)
My_thermometer.add(19.0)
My_thermometer.add(18.0)

print(f"Average: {My_thermometer.average()}")
print(f"Hottest: {My_thermometer.hottest()}")



# Q2
# Add a __repr__ to Thermometer that prints something like:
#   Thermometer(location='Charlotte', n_readings=4, average=18.6)
# Print the object directly (print(my_thermometer)), and also print a list
# containing two of them.
# Comment: what does Python display when a class has no __repr__, and why is
# that unhelpful when debugging?

A_thermometer = Thermometer("Atlanta", [22.0, 21.0, 23.0])
print(My_thermometer)
print([My_thermometer, A_thermometer])

from dataclasses import dataclass, field, FrozenInstanceError

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator


class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location
        self.readings = readings if readings is not None else []

    def add(self, reading):
        self.readings.append(reading)


My_thermometer = Thermometer("Charlotte", [20.0, 18.0, 19.0, 18.0])


# --- Classes ---

# Q3
class TemperatureAlert:
    """Flags readings above a threshold."""

    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        return [r for r in thermometer.readings if r > self.threshold]


mild_alert = TemperatureAlert(18.5)
hot_alert = TemperatureAlert()
print(f"Above {mild_alert.threshold}: {mild_alert.breaches(My_thermometer)}")
print(f"Above {hot_alert.threshold}: {hot_alert.breaches(My_thermometer)}")
# The threshold is configuration that belongs to the alert, not to each check.
# Store it once and every call to breaches() reuses it. With twenty
# thermometers you loop over them calling alert.breaches(t), and the threshold
# can't drift between calls or be mistyped on one of them. You can also keep
# several alerts (mild, hot) side by side and pass them around as objects.


# --- Dataclasses, Type Hints, and Docstrings ---

# Q1
@dataclass
class Station:
    """A weather station and its location (degrees) and elevation (meters)."""
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float


station_a = Station("CLT01", "Charlotte Airport", 35.21, -80.94, 228.0)
station_b = Station("CLT01", "Charlotte Airport", 35.21, -80.94, 228.0)
print(station_a == station_b)
# True. @dataclass generates __eq__, which compares the objects field by field.
# The hand-written class has no __eq__, so == falls back to identity (like
# Java's default equals()): two separate objects would be False even with
# identical values.


# Q2
@dataclass(frozen=True)
class Station:
    """A weather station and its location (degrees) and elevation (meters)."""
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float


frozen_station = Station("CLT01", "Charlotte Airport", 35.21, -80.94, 228.0)
try:
    frozen_station.elevation = 300.0
except FrozenInstanceError as e:
    print(f"FrozenInstanceError: {e}")

stations = {
    Station("CLT01", "Charlotte Airport", 35.21, -80.94, 228.0),
    Station("CLT01", "Charlotte Airport", 35.21, -80.94, 228.0),
    Station("RDU01", "Raleigh-Durham", 35.88, -78.79, 132.0),
}
print(len(stations))
# 2. frozen=True also generates __hash__ (based on the fields), so stations can
# go in a set or be dict keys, and equal stations collapse into one entry.
# That's only safe because they're immutable: if a field could change after
# the object went into a set, its hash would change and the set would lose it.


# Q3
# First try:  stations: list[Station] = []
#   ValueError: mutable default <class 'list'> for field stations is not
#   allowed: use default_factory
# A default value is created once, when the class is defined, so every
# StationBatch would share the SAME list -- adding to one batch would add to
# all of them. Python refuses it. field(default_factory=list) calls list()
# for each new object, so every batch gets its own empty list.
@dataclass
class StationBatch:
    """A group of stations in one region."""
    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Add a station to the batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the greatest elevation, or None if empty."""
        if not self.stations:
            return None
        return max(self.stations, key=lambda s: s.elevation)


batch = StationBatch("North Carolina")
print(batch.highest())
batch.add(Station("CLT01", "Charlotte Airport", 35.21, -80.94, 228.0))
batch.add(Station("RDU01", "Raleigh-Durham", 35.88, -78.79, 132.0))
batch.add(Station("AVL01", "Asheville", 35.43, -82.54, 652.0))
print(batch.highest())


# --- Pydantic ---

# Q1
class Reading(BaseModel):
    """One sensor reading."""
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)


reading = Reading(station_id="CLT01", timestamp="2026-04-08T12:00",
                  temperature_c=18.5, humidity=55.0)
print(reading)


# Q2
try:
    Reading(station_id="CLT01", temperature_c=18.5, humidity=55.0)
except ValidationError as e:
    print(e)

try:
    Reading(station_id="CLT01", timestamp="2026-04-08T12:00",
            temperature_c=150.0, humidity=55.0)
except ValidationError as e:
    print(e)

try:
    Reading(station_id="CLT01", timestamp="2026-04-08T12:00",
            temperature_c=18.5, humidity="very humid")
except ValidationError as e:
    print(e)

coerced = Reading(station_id="CLT01", timestamp="2026-04-08T12:00",
                  temperature_c="21.5", humidity=40)
print(coerced)
print(type(coerced.temperature_c), type(coerced.humidity))
# Pydantic converts a value to the declared type when the conversion is
# unambiguous and loses nothing: "21.5" is clearly the number 21.5, and 40 is
# exactly 40.0. "very humid" has no numeric meaning, so there's nothing to
# convert it to and it's rejected.


# Q3
try:
    Reading(station_id="AB", temperature_c="hot", humidity=50.0)
except ValidationError as e:
    for err in e.errors():
        print(err["loc"], err["msg"])
# 3 errors. Pydantic checks every field before raising, so you see all the
# problems at once and can fix them in one pass, instead of fix-rerun-repeat.
# With data from an API, that also tells you everything that's wrong with a
# bad record in a single log entry.


# Q4
class Reading(BaseModel):
    """One sensor reading."""
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def check_sensor_failure(self):
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("humidity 0.0 with temperature below -40 looks like a failed sensor")
        return self


print(Reading(station_id="CLT01", timestamp="2026-04-08T12:00",
              temperature_c=-45.0, humidity=20.0))
try:
    Reading(station_id="CLT01", timestamp="2026-04-08T12:00",
            temperature_c=-45.0, humidity=0.0)
except ValidationError as e:
    print(e)
# Field constraints check one field at a time. -45 is a valid temperature and
# 0.0 is a valid humidity on their own; only the COMBINATION is bad. A
# model_validator(mode="after") runs once all fields are validated, so it can
# look at several fields together.


# --- pytest ---

# Q1
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)
# 37 * 9 / 5 + 32 gives 98.60000000000001 because floats can't store most
# decimals exactly, so a plain == fails. pytest.approx allows a tiny tolerance.


# Q2
def mean(values: list[float]) -> float:
    """Return the arithmetic mean of values. Raises ValueError if empty."""
    if not values:
        raise ValueError("cannot take the mean of an empty list")
    return sum(values) / len(values)


def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])
# pytest.raises(ValueError) alone passes for ANY ValueError, including one
# raised by a different bug in the function. match= also checks the message,
# so the test only passes when it's the error we actually meant to raise.


# Q3
@pytest.mark.parametrize("values, expected", [
    ([1.0, 2.0, 3.0], 2.0),
    ([5.0], 5.0),
    ([-2.0, -4.0], -3.0),
    ([-1.0, 1.0, 3.0], 1.0),
])
def test_mean_values(values, expected):
    assert mean(values) == pytest.approx(expected)
# Summary line:
# ============================== 6 passed in 0.03s ===============================
# One parametrized test keeps the logic in one place: fixing or changing the
# assertion fixes every case, adding a case is one line, and pytest still
# reports each case separately so you can see exactly which input failed.


# Q4
# With 9 / 5 changed to 9 / 4, the failure output was:
#     >       assert celsius_to_fahrenheit(100) == 212
#     E       assert 257.0 == 212
#     E        +  where 257.0 = celsius_to_fahrenheit(100)
#     FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 257.0 == 212
# pytest showed the actual result (257.0), the expected value (212), and the
# call that produced it (celsius_to_fahrenheit(100)). Note 0 C still passed,
# since 0 * anything + 32 is 32 -- the 100 C case is what caught the bug. A
# bare "assertion failed" wouldn't say which input broke or by how much.
