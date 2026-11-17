"""Rain radar cameras (NEA base map, still + animated) in a real Home Assistant.

The existing weather.gov.sg rain map cameras are set up alongside them, unchanged.
"""
from __future__ import annotations

import io
from http import HTTPStatus

from PIL import Image
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.nea_sg_weather.const import DOMAIN
from custom_components.nea_sg_weather.radar_image import BASEMAPS

from .conftest import RADAR_FRAME_URL

WITH_RAIN = {
    "name": "Singapore Weather",
    "weather": False,
    "sensor": True,
    "scan_interval": 15,
    "timeout": 60,
    "sensors": {"prefix": "Singapore Weather", "areas": [], "region": False, "rain": True},
}

RADAR_CAMERAS = {
    "camera.singapore_weather_rain_radar_70km": "Singapore Weather Rain radar 70 km",
    "camera.singapore_weather_rain_radar_240km": "Singapore Weather Rain radar 240 km",
    "camera.singapore_weather_rain_radar_animation_70km": (
        "Singapore Weather Rain radar animation 70 km"
    ),
    "camera.singapore_weather_rain_radar_animation_240km": (
        "Singapore Weather Rain radar animation 240 km"
    ),
}


def _png(size, mode="RGBA") -> bytes:
    out = io.BytesIO()
    Image.new(mode, size).save(out, format="PNG")
    return out.getvalue()


async def _load(hass: HomeAssistant) -> MockConfigEntry:
    entry = MockConfigEntry(domain=DOMAIN, data=WITH_RAIN, title=WITH_RAIN["name"])
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def test_radar_cameras_added_next_to_rain_maps(hass: HomeAssistant, mock_nea_api):
    entry = await _load(hass)
    ent_reg = er.async_get(hass)
    ids = {
        e.entity_id
        for e in er.async_entries_for_config_entry(ent_reg, entry.entry_id)
        if e.domain == "camera"
    }
    # The existing cameras keep their IDs.
    assert {
        "camera.singapore_weather_rain_map",
        "camera.singapore_weather_animated_rain_map",
    } <= ids
    assert set(RADAR_CAMERAS) <= ids
    for entity_id, name in RADAR_CAMERAS.items():
        state = hass.states.get(entity_id)
        assert state is not None and state.state != "unavailable", entity_id
        assert state.name == name
    attrs = hass.states.get("camera.singapore_weather_rain_radar_animation_70km").attributes
    assert attrs["frame_count"] == 3
    assert attrs["image_timestamp"] == "2024-01-01T12:10:00+08:00"


async def test_radar_images_are_drawn_on_nea_basemap(
    hass: HomeAssistant, hass_client, mock_nea_api
):
    mock_nea_api.get(BASEMAPS["70km"].url, content=_png((853, 479), "RGB"))
    for minute in (10, 5, 0):
        mock_nea_api.get(RADAR_FRAME_URL.format(minute=minute), content=_png((480, 480)))
    await _load(hass)
    client = await hass_client()

    resp = await client.get("/api/camera_proxy/camera.singapore_weather_rain_radar_70km")
    assert resp.status == HTTPStatus.OK
    assert resp.content_type == "image/jpeg"
    assert Image.open(io.BytesIO(await resp.read())).size == (853, 479)

    resp = await client.get(
        "/api/camera_proxy/camera.singapore_weather_rain_radar_animation_70km"
    )
    assert resp.status == HTTPStatus.OK
    assert resp.content_type == "image/gif"
    gif = Image.open(io.BytesIO(await resp.read()))
    assert gif.n_frames == 3


async def test_radar_coordinator_stops_on_unload(hass: HomeAssistant, mock_nea_api):
    entry = await _load(hass)
    calls = mock_nea_api.call_count
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert hass.states.get("camera.singapore_weather_rain_radar_70km").state == "unavailable"
    assert mock_nea_api.call_count == calls
