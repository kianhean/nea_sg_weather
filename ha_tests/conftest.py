"""Fixtures for Home Assistant integration tests."""
import re
import threading
import pytest


# pytest-homeassistant-custom-component blocks custom integrations by default.
# This autouse fixture opts every test in ha_tests/ into the allow-list so HA
# can find and load the integration from the local custom_components/ directory.
@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Allow HA to load integrations from the local custom_components/ directory."""
    yield


@pytest.fixture(autouse=True)
def verify_cleanup():
    """Override PHAC's verify_cleanup to also allow aiohttp's daemon thread.

    aiohttp ≥3.9 spawns a '_run_safe_shutdown_loop' thread when the connector
    is first used (via HA's shared ClientSession). PHAC only whitelists
    _DummyThread and 'waitpid-' threads, so we extend that check here.
    """
    threads_before = {t for t in threading.enumerate() if t.is_alive()}
    yield
    threads_after = {t for t in threading.enumerate() if t.is_alive()}
    for thread in threads_after - threads_before:
        assert (
            isinstance(thread, threading._DummyThread)
            or thread.name.startswith("waitpid-")
            or "_run_safe_shutdown_loop" in thread.name
            or thread.name.startswith("SyncWorker_")
        ), f"Unexpected thread left after test: {thread.name!r}"


# ---------------------------------------------------------------------------
# Canned API responses — realistic enough that nea.py process_data() runs
# (each payload stringifies to well over 120 characters).
# ---------------------------------------------------------------------------

FORECAST_2HR = {
    "data": {
        "items": [{
            "timestamp": "2024-01-01T12:00:00+08:00",
            "forecasts": [
                {"area": "Ang Mo Kio", "forecast": "Partly Cloudy"},
                {"area": "Bedok", "forecast": "Partly Cloudy"},
                {"area": "Bishan", "forecast": "Sunny"},
            ],
        }],
        "area_metadata": [
            {"label_location": {"latitude": "1.3691", "longitude": "103.8454"}},
            {"label_location": {"latitude": "1.3236", "longitude": "103.9273"}},
            {"label_location": {"latitude": "1.3526", "longitude": "103.8352"}},
        ],
    },
}

FORECAST_24HR = {
    "data": {
        "records": [{
            "timestamp": "2024-01-01T12:00:00+08:00",
            "periods": [{
                "timePeriod": {"start": "2024-01-01T06:00:00+08:00"},
                "regions": {
                    "west":    {"text": "Partly Cloudy"},
                    "east":    {"text": "Sunny"},
                    "central": {"text": "Cloudy"},
                    "south":   {"text": "Partly Cloudy"},
                    "north":   {"text": "Sunny"},
                },
            }],
        }],
    },
}

FORECAST_4DAY = {
    "data": {
        "records": [{
            "forecasts": [{
                "timestamp": "2024-01-02T00:00:00+08:00",
                "forecast": {"text": "Partly Cloudy"},
                "temperature": {"high": 34, "low": 25},
                "wind": {"speed": {"high": 20, "low": 10}, "direction": "NE"},
            }],
        }],
    },
}

TEMPERATURE = {
    "data": {
        "readings": [{
            "timestamp": "2024-01-01T12:00:00+08:00",
            "data": [
                {"stationId": "S01", "value": 28.5},
                {"stationId": "S02", "value": 30.0},
            ],
        }],
    },
}

HUMIDITY = {
    "data": {
        "readings": [{
            "timestamp": "2024-01-01T12:00:00+08:00",
            "data": [
                {"stationId": "S01", "value": 75.0},
                {"stationId": "S02", "value": 80.0},
            ],
        }],
    },
}

WIND_DIRECTION = {
    "data": {
        "readings": [{
            "timestamp": "2024-01-01T12:00:00+08:00",
            "data": [
                {"stationId": "S01", "value": 45},
                {"stationId": "S02", "value": 90},
            ],
        }],
    },
}

WIND_SPEED = {
    "data": {
        "readings": [{
            "timestamp": "2024-01-01T12:00:00+08:00",
            "data": [
                {"stationId": "S01", "value": 10},
                {"stationId": "S02", "value": 15},
            ],
        }],
    },
}

RAINFALL = {
    "data": {
        "readings": [{
            "timestamp": "2024-01-01T12:00:00+08:00",
            "data": [{"stationId": "S117", "value": 0.0}],
        }],
        "stations": [{
            "id": "S117",
            "name": "Banyan Road",
            "location": {"latitude": 1.256, "longitude": 103.679},
        }],
    },
}

UV_INDEX = {
    "data": {
        "records": [{
            "date": "2024-01-01",
            "updatedTimestamp": "2024-01-01T11:11:04+08:00",
            "timestamp": "2024-01-01T11:00:00+08:00",
            "index": [
                {"hour": "2024-01-01T12:00:00+08:00", "value": 5},
                {"hour": "2024-01-01T11:00:00+08:00", "value": 3},
                {"hour": "2024-01-01T10:00:00+08:00", "value": 1},
                {"hour": "2024-01-01T09:00:00+08:00", "value": 1},
                {"hour": "2024-01-01T08:00:00+08:00", "value": 0},
                {"hour": "2024-01-01T07:00:00+08:00", "value": 0},
            ],
        }],
    },
}

PM25 = {
    "data": {
        "items": [{
            "timestamp": "2024-01-01T11:30:00+08:00",
            "readings": {
                "pm25_one_hourly": {
                    "west": 15, "east": 18, "central": 20, "south": 16, "north": 14,
                },
            },
        }],
    },
}

PSI = {
    "data": {
        "items": [{
            "timestamp": "2024-01-01T12:00:00+08:00",
            "readings": {
                "psi_twenty_four_hourly": {
                    "west": 55, "east": 60, "central": 52, "south": 58, "north": 50,
                },
                "pm25_twenty_four_hourly": {
                    "west": 20, "east": 22, "central": 18, "south": 21, "north": 19,
                },
                "pm10_twenty_four_hourly": {
                    "west": 30, "east": 32, "central": 28, "south": 31, "north": 29,
                },
                "so2_twenty_four_hourly": {
                    "west": 6, "east": 5, "central": 3, "south": 3, "north": 5,
                },
                "o3_eight_hour_max": {
                    "west": 36, "east": 34, "central": 48, "south": 31, "north": 65,
                },
                "co_eight_hour_max": {
                    "west": 0.6, "east": 0.5, "central": 0.7, "south": 0.4, "north": 0.5,
                },
                "no2_one_hour_max": {
                    "west": 52, "east": 46, "central": 46, "south": 42, "north": 28,
                },
                "pm25_sub_index": {
                    "west": 55, "east": 60, "central": 52, "south": 58, "north": 50,
                },
                "pm10_sub_index": {
                    "west": 30, "east": 32, "central": 28, "south": 31, "north": 29,
                },
                "so2_sub_index": {
                    "west": 5, "east": 6, "central": 4, "south": 5, "north": 5,
                },
                "co_sub_index": {
                    "west": 3, "east": 3, "central": 2, "south": 3, "north": 3,
                },
                "o3_sub_index": {
                    "west": 10, "east": 12, "central": 9, "south": 11, "north": 10,
                },
            },
        }],
    },
}

# Rain radar (weather-radar-images/{range}?date=…): the day's images, newest first.
RADAR_FRAME_URL = "https://s3.example/radar-{minute:02d}.png"
RADAR = {
    "code": 0,
    "data": {
        "projection": {"center": {"longitude": 103.972583, "latitude": 1.34911}},
        "boundaryBox": {
            "upperLeft": {"longitude": 103.342685, "latitude": 1.97854},
            "lowerRight": {"longitude": 104.602315, "latitude": 0.719515},
        },
        "records": [
            {
                "timestamp": f"2024-01-01T12:{minute:02d}:00+08:00",
                "image": {"url": RADAR_FRAME_URL.format(minute=minute)},
            }
            for minute in (10, 5, 0)
        ],
    },
}

_BASE = "https://api-open.data.gov.sg/v2/real-time/api"

_ENDPOINTS = [
    ("two-hr-forecast",        FORECAST_2HR),
    ("twenty-four-hr-forecast", FORECAST_24HR),
    ("four-day-outlook",       FORECAST_4DAY),
    ("air-temperature",        TEMPERATURE),
    ("relative-humidity",      HUMIDITY),
    ("wind-direction",         WIND_DIRECTION),
    ("wind-speed",             WIND_SPEED),
    ("rainfall",               RAINFALL),
    ("uv",                     UV_INDEX),
    ("pm25",                   PM25),
    ("psi",                    PSI),
    ("weather-radar-images",   RADAR),
]


@pytest.fixture
def mock_nea_api(aioclient_mock):
    """Intercept all NEA API calls and return canned responses.

    Uses HA's aioclient_mock, which patches the shared ClientSession the
    integration gets from async_get_clientsession(). Regex matching ignores
    query-string parameters (date_time=…), and registered responses are
    served for every matching request.
    """
    for path, payload in _ENDPOINTS:
        aioclient_mock.get(
            re.compile(rf"{re.escape(_BASE)}/{path}"),
            json=payload,
        )
    return aioclient_mock
