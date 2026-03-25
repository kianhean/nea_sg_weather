"""Boot / lifecycle tests for the nea_sg_weather integration.

These tests exercise the real async_setup_entry → async_unload_entry path
with a carefully wired mock ``hass`` object and a patched data-fetch layer
(so no real HTTP calls are made).  They verify that:

- The coordinator is created and stored in hass.data
- The correct HA platforms (weather / sensor / camera) are forwarded
- Unloading cleans up hass.data and succeeds
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.nea_sg_weather import async_setup_entry, async_unload_entry
from custom_components.nea_sg_weather.const import DOMAIN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hass():
    """Return a minimal mock hass that satisfies async_setup_entry."""
    hass = MagicMock()
    hass.data = {}
    hass.config_entries.async_forward_entry_setups = AsyncMock(return_value=True)
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)
    return hass


def _make_entry(weather=True, sensor=False, areas=None, region=False, rain=False):
    """Return a mock ConfigEntry with the given feature flags."""
    entry = MagicMock()
    entry.entry_id = "boot-test-entry"
    data = {
        "weather": weather,
        "sensor": sensor,
        "scan_interval": 15,
        "timeout": 60,
    }
    if sensor:
        data["sensors"] = {
            "prefix": "SG",
            "areas": areas if areas is not None else ["Ang Mo Kio"],
            "region": region,
            "rain": rain,
        }
    entry.data = data
    return entry


# ---------------------------------------------------------------------------
# Fixture: patch NeaWeatherData.async_update so no real HTTP is needed
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_data_fetch():
    """Replace the coordinator data-fetch with a no-op that returns a MagicMock."""
    with patch(
        "custom_components.nea_sg_weather.NeaWeatherData.async_update",
        new_callable=AsyncMock,
        return_value=MagicMock(),
    ):
        yield


# ---------------------------------------------------------------------------
# Setup tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_setup_entry_returns_true_for_weather_only():
    result = await async_setup_entry(_make_hass(), _make_entry(weather=True))
    assert result is True


@pytest.mark.asyncio
async def test_setup_entry_returns_true_for_sensor_config():
    result = await async_setup_entry(
        _make_hass(),
        _make_entry(weather=False, sensor=True, areas=["Ang Mo Kio"]),
    )
    assert result is True


@pytest.mark.asyncio
async def test_setup_entry_stores_coordinator_in_hass_data():
    hass = _make_hass()
    entry = _make_entry(weather=True)
    await async_setup_entry(hass, entry)
    assert DOMAIN in hass.data
    assert entry.entry_id in hass.data[DOMAIN]


@pytest.mark.asyncio
async def test_setup_entry_weather_only_forwards_weather_platform():
    hass = _make_hass()
    await async_setup_entry(hass, _make_entry(weather=True, sensor=False))
    platforms = hass.config_entries.async_forward_entry_setups.call_args[0][1]
    assert "weather" in platforms
    assert "sensor" not in platforms
    assert "camera" not in platforms


@pytest.mark.asyncio
async def test_setup_entry_sensor_config_forwards_sensor_platform():
    hass = _make_hass()
    await async_setup_entry(
        hass,
        _make_entry(weather=False, sensor=True, areas=["Ang Mo Kio"]),
    )
    platforms = hass.config_entries.async_forward_entry_setups.call_args[0][1]
    assert "sensor" in platforms


@pytest.mark.asyncio
async def test_setup_entry_region_sensor_forwards_sensor_platform():
    hass = _make_hass()
    await async_setup_entry(
        hass,
        _make_entry(weather=False, sensor=True, areas=["None"], region=True),
    )
    platforms = hass.config_entries.async_forward_entry_setups.call_args[0][1]
    assert "sensor" in platforms


@pytest.mark.asyncio
async def test_setup_entry_rain_config_forwards_camera_platform():
    hass = _make_hass()
    await async_setup_entry(
        hass,
        _make_entry(weather=False, sensor=True, areas=["None"], rain=True),
    )
    platforms = hass.config_entries.async_forward_entry_setups.call_args[0][1]
    assert "camera" in platforms


@pytest.mark.asyncio
async def test_setup_entry_weather_and_sensor_forwards_both_platforms():
    hass = _make_hass()
    await async_setup_entry(
        hass,
        _make_entry(weather=True, sensor=True, areas=["Bedok"], region=True),
    )
    platforms = hass.config_entries.async_forward_entry_setups.call_args[0][1]
    assert "weather" in platforms
    assert "sensor" in platforms


# ---------------------------------------------------------------------------
# Unload tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unload_entry_returns_true():
    hass = _make_hass()
    entry = _make_entry(weather=True)
    await async_setup_entry(hass, entry)
    result = await async_unload_entry(hass, entry)
    assert result is True


@pytest.mark.asyncio
async def test_unload_entry_removes_coordinator_from_hass_data():
    hass = _make_hass()
    entry = _make_entry(weather=True)
    await async_setup_entry(hass, entry)
    await async_unload_entry(hass, entry)
    assert entry.entry_id not in hass.data.get(DOMAIN, {})


@pytest.mark.asyncio
async def test_full_setup_then_unload_lifecycle():
    """Full happy-path: setup succeeds, then unload cleans up."""
    hass = _make_hass()
    entry = _make_entry(weather=True, sensor=True, areas=["All"], region=True)

    assert await async_setup_entry(hass, entry) is True
    assert entry.entry_id in hass.data[DOMAIN]

    assert await async_unload_entry(hass, entry) is True
    assert entry.entry_id not in hass.data.get(DOMAIN, {})
