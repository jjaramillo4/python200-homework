from pydantic import BaseModel, Field, model_validator


class HourlyBlock(BaseModel):
    """The hourly measurements, stored as parallel (columnar) lists."""
    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def check_equal_lengths(self):
        lengths = {len(self.time), len(self.temperature_2m), len(self.precipitation)}
        if len(lengths) != 1:
            raise ValueError(
                f"hourly lists differ in length: time={len(self.time)}, "
                f"temperature_2m={len(self.temperature_2m)}, "
                f"precipitation={len(self.precipitation)}"
            )
        return self


class WeatherResponse(BaseModel):
    """The top level of an Open-Meteo hourly forecast response."""
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock
