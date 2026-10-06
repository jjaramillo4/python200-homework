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
        return f"Thermometer(location='{self.location}', n_readings={len(self.readings)}, average={round(self.average(), 1)})"
   
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

# __repr__ is useful for debugging because it provides a clear and concise representation of the object, including its location, number of readings, and average temperature. Without __repr__, Python would display a default representation that includes the object's memory address, which is not informative for understanding the object's state.

# Q3
# Write a class TemperatureAlert that holds a threshold (float, default 30.0)
# and has one method:
# - breaches(thermometer) -- returns a list of every reading in that
#   Thermometer above the threshold.
# Create two TemperatureAlert objects with different thresholds and run both
# against the SAME Thermometer. Print both results.
# Comment: why is the threshold stored on TemperatureAlert rather than passed
# as an argument to breaches()? What advantage does that give you if you have
# twenty thermometers to check?


# --- Dataclasses, Type Hints, and Docstrings ---

# Q1
# Rewrite this class as a dataclass. Add type hints to every field and a
# docstring describing what it represents.
#
#   class Station:
#       def __init__(self, station_id, name, latitude, longitude, elevation):
#           self.station_id = station_id
#           self.name = name
#           self.latitude = latitude
#           self.longitude = longitude
#           self.elevation = elevation
#
# Create two Station objects with identical field values and print
# station_a == station_b.
# Comment: why is the result what it is, and what would it have been with the
# original hand-written class?


# Q2
# Make Station frozen. Then:
# 1. Show that assigning to a field now raises FrozenInstanceError (catch it
#    and print the message -- do not let the script crash).
# 2. Build a set containing three Station objects where two are identical,
#    and print the length.
# Comment: what does frozen=True give you besides immutability, and why is
# that useful here?


# Q3
# Write a dataclass StationBatch with:
# - a region field (str)
# - a stations field that is a list of Station, defaulting to empty
# - a method add(station: Station) -> None
# - a method highest(self) -> Station | None that returns the station with the
#   greatest elevation, or None if the batch is empty
# First try writing the default as  stations: list[Station] = []  , run it,
# and paste the error you get into a comment. Then fix it properly and explain
# in that comment why Python refuses the first version.
# Give every method a type-hinted signature and a docstring.


# --- Pydantic ---

# Q1
# Write a Pydantic model Reading with these fields:
#   station_id     str    at least 3 characters
#   timestamp      str    required
#   temperature_c  float  between -90 and 60
#   humidity       float  between 0 and 100
# Construct one valid Reading and print it.


# Q2
# Show three separate failures, each wrapped in try / except ValidationError
# so the script keeps running. Print the error each time.
# 1. A missing required field
# 2. A temperature_c of 150.0
# 3. A humidity of "very humid"
# Then construct a Reading where temperature_c is passed as the STRING "21.5"
# and humidity is passed as the INTEGER 40. Print the resulting object and the
# type() of both fields.
# Comment: why does Pydantic accept "21.5" but reject "very humid"? State the
# rule in your own words.


# Q3
# Trigger several errors at once. In one try block, construct a Reading with a
# too-short station_id, a missing timestamp, and a non-numeric temperature_c.
# Catch the ValidationError and loop over e.errors(), printing the loc and msg
# for each.
# Comment: how many errors were reported, and why is reporting all of them at
# once more useful than stopping at the first?


# Q4
# Add a model_validator(mode="after") to Reading that rejects any reading where
# humidity is exactly 0.0 AND temperature_c is below -40 (a failed sensor, not
# real weather).
# Show that a valid reading still constructs, and that the bad combination
# raises.
# Comment: why can't this rule be expressed with Field constraints alone?


# --- pytest ---
# Write these as real pytest tests named test_* in this file.

# Q1
# Write celsius_to_fahrenheit(celsius: float) -> float with a docstring.
# Then write test_celsius_to_fahrenheit() asserting that:
# - 0 C is 32 F
# - 100 C is 212 F
# - 37 C is approximately 98.6 F
# The third one will fail with a plain ==. Make it pass with pytest.approx.
# Comment: why was pytest.approx necessary?


# Q2
# Write mean(values: list[float]) -> float that raises a ValueError with a
# useful message when values is empty.
# Write test_mean_of_empty_raises() using pytest.raises(ValueError, match=...)
# to confirm both that the error is raised and that the message contains the
# word you expect.
# Comment: what would pytest.raises(ValueError) alone fail to catch that
# match= catches?


# Q3
# Write test_mean_values() using @pytest.mark.parametrize to check at least
# four input/output pairs for mean, including a single-element list and a
# list containing negative numbers.
# Run  pytest warmup_01.py -v  and paste the summary line into a comment.
# Comment: why is one parametrized test with four cases better than four
# nearly identical test functions?


# Q4
# Deliberately break celsius_to_fahrenheit (e.g. change 9 / 5 to 9 / 4). Run
# your test again and paste the failure output into a comment. Then fix the
# function.
# Comment: what specific values did pytest show you in the failure report, and
# why is that more useful than a bare "assertion failed"?
