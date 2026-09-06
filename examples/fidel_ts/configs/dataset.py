"""Per-sub-dataset configuration for Fidel-TS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from examples.fidel_ts.data.fidel_ts_dataset import TextSource
from tsfmx.utils.yaml import load_yaml


@dataclass
class FidelTsConfig:
    """Names the series and the text streams of one Fidel-TS sub-dataset.

    Every sub-dataset ships the same directory shape but different channel names and report
    paths, so this is what a script needs in order to open one it was not written for.

    Attributes:
        name: Sub-dataset directory under data/Fidel-TS, e.g. 'Bear_room'.
        target_column: Column of the parquet files to forecast.
        timestamp_column: Column holding the timestamps.
        text_sources: Text streams to retrieve as of the prediction time.
        time_series_dir: Subdirectory holding the parquet files.
        entity_locations: Entity to weather-station name, for sub-datasets whose text is keyed by
            station rather than by series. Germany_Renewable_Energy_Grid names its series after the
            grid operator and its weather after the city, so the two need relating.
        std_floor_ratio: Lower bound on a window's standard deviation, as a fraction of the whole
            series'.
    """

    name: str = "Bear_room"
    target_column: str = "Zone Temperature"
    timestamp_column: str = "Timestamp"
    time_series_dir: str = "time_series"
    entity_locations: dict[str, str] = field(default_factory=dict)
    std_floor_ratio: float = 0.1
    text_sources: list[TextSource] = field(
        default_factory=lambda: [
            TextSource(name="Weather", path="hetero/weather/weather_report/formal_report/wm_messages_v1.json"),
            TextSource(
                name="Control",
                path="hetero/room/room_report.zip/room_report/formal_report/{entity}/all.json",
                per_entity=True,
            ),
        ]
    )

    @classmethod
    def from_yaml(cls, path: Path) -> FidelTsConfig:
        config_dict = load_yaml(path)
        sources = config_dict.pop("text_sources", None)
        config = cls(**config_dict)
        if sources is not None:
            config.text_sources = [TextSource(**source) for source in sources]
        return config
