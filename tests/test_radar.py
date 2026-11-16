"""Tests for the NEA rain radar coordinator, base-map rendering and camera entities.

Ported from the singapore integration's tests/test_radar.py.
"""

import io
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from PIL import Image

from custom_components.nea_sg_weather import camera as camera_module
from custom_components.nea_sg_weather.camera import (
    BASEMAP_RETRY_SECONDS,
    NeaAnimatedRainCamera,
    NeaRainCamera,
    NeaRainRadarAnimationCamera,
    NeaRainRadarCamera,
    async_setup_entry,
)
from custom_components.nea_sg_weather.radar import (
    ANIMATION_WINDOW,
    RADAR_RANGES,
    RADAR_URL,
    SGT,
    RadarCoordinator,
    RadarData,
    UpdateFailed,
    parse_radar as _parse_radar,
)
from custom_components.nea_sg_weather.radar_image import (
    BASEMAPS,
    FRAME_MS,
    HOLD_MS,
    animate_radar,
    composite_radar,
)

NOON = datetime(2026, 11, 13, 12, 0, tzinfo=SGT)
PREFIX = "Singapore Weather"


async def _refresh(coordinator) -> None:
    """What DataUpdateCoordinator.async_refresh does with the result (HA is stubbed)."""
    try:
        coordinator.data = await coordinator._async_update_data()
        coordinator.last_update_success = True
    except UpdateFailed:
        coordinator.last_update_success = False


@pytest.fixture(autouse=True)
def pinned_now():
    """Pin Singapore time to midday, so no test fetches the previous day by chance."""
    clock = [NOON]
    with patch(
        "custom_components.nea_sg_weather.radar._now",
        side_effect=lambda: clock[0],
    ):
        yield clock


def _payload(
    radar_range: str = "70km",
    url: str = "https://s3.example/rain.png?sig=1",
    timestamp: str = "2026-11-13T22:25:00+08:00",
) -> dict:
    # Trimmed from a live 2026-11-13 response.
    return {
        "code": 0,
        "data": {
            "projection": {
                "type": "Azimuthal Equidistant",
                "center": {"longitude": 103.972583, "latitude": 1.34911},
            },
            "boundaryBox": {
                "upperLeft": {"longitude": 103.342685, "latitude": 1.97854},
                "lowerRight": {"longitude": 104.602315, "latitude": 0.719515},
                "crs": "EPSG:4326",
            },
            "records": [
                {
                    "timestamp": timestamp,
                    "updatedTimestamp": "2026-11-13T22:24:13+08:00",
                    "image": {
                        "url": url,
                        "urlExpiresAt": "2026-11-13T22:49:51+08:00",
                        "label": radar_range,
                        "range": radar_range,
                        "format": "png",
                    },
                }
            ],
        },
        "errorMsg": "",
    }


def test_parse_radar_image_and_bounds():
    image = _parse_radar(_payload(), "70km")
    assert image.range == "70km"
    assert image.image_url == "https://s3.example/rain.png?sig=1"
    assert image.timestamp.isoformat() == "2026-11-13T22:25:00+08:00"
    assert image.url_expires_at.isoformat() == "2026-11-13T22:49:51+08:00"
    assert image.north == pytest.approx(1.97854)
    assert image.south == pytest.approx(0.719515)
    assert image.west == pytest.approx(103.342685)
    assert image.east == pytest.approx(104.602315)
    assert image.center_latitude == pytest.approx(1.34911)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"code": 0, "data": {"records": []}},
        {"code": 0, "data": {"records": [{"timestamp": "x", "image": {}}]}},
    ],
)
def test_parse_radar_bad_payload_raises(payload):
    with pytest.raises(UpdateFailed):
        _parse_radar(payload, "70km")


def test_parse_radar_tolerates_missing_metadata():
    payload = {"data": {"records": [{"image": {"url": "https://x.png"}}]}}
    image = _parse_radar(payload, "480km")
    assert image.image_url == "https://x.png"
    assert image.timestamp is None
    assert image.north is None


def test_radar_ranges():
    assert list(RADAR_RANGES) == ["70km", "240km"]
    # Every range is drawn on an NEA base map.
    assert set(RADAR_RANGES) == set(BASEMAPS)


def _response(status: int = 200, json_payload=None, body: bytes = b"png"):
    response = AsyncMock()
    response.headers = {}  # real mapping: the data.gov.sg limiter reads it on 429
    response.status = status
    response.json = AsyncMock(return_value=json_payload)
    response.read = AsyncMock(return_value=body)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    return response


def _session(responses: dict):
    """Session whose GET answers per-URL; values are responses or exceptions."""
    session = MagicMock()

    def _get(url, **_kwargs):
        result = responses[url]
        if isinstance(result, Exception):
            raise result
        return result

    session.get = MagicMock(side_effect=_get)
    return session


def _all_ok(**overrides) -> dict:
    responses = {
        RADAR_URL.format(range=r): _response(
            json_payload=_payload(r, url=f"https://s3.example/{r}.png")
        )
        for r in RADAR_RANGES
    }
    responses.update(overrides)
    return responses


@pytest.mark.asyncio
async def test_coordinator_fetches_every_range():
    coordinator = RadarCoordinator(MagicMock())
    with patch(
        "custom_components.nea_sg_weather.radar.async_get_clientsession",
        return_value=_session(_all_ok()),
    ):
        await _refresh(coordinator)
    assert coordinator.last_update_success is True
    assert set(coordinator.data.images) == set(RADAR_RANGES)
    assert coordinator.data.images["240km"].image_url.endswith("240km.png")


@pytest.mark.asyncio
async def test_coordinator_partial_failure_keeps_last_image():
    coordinator = RadarCoordinator(MagicMock())
    with patch(
        "custom_components.nea_sg_weather.radar.async_get_clientsession",
        return_value=_session(_all_ok()),
    ):
        await _refresh(coordinator)
    old_240 = coordinator.data.images["240km"]

    url_240 = RADAR_URL.format(range="240km")
    with patch(
        "custom_components.nea_sg_weather.radar.async_get_clientsession",
        return_value=_session(_all_ok(**{url_240: _response(status=429)})),
    ):
        await _refresh(coordinator)
    assert coordinator.last_update_success is True
    assert coordinator.data.images["240km"] is old_240


@pytest.mark.asyncio
async def test_coordinator_partial_failure_without_cache_omits_range():
    coordinator = RadarCoordinator(MagicMock())
    url_70 = RADAR_URL.format(range="70km")
    with patch(
        "custom_components.nea_sg_weather.radar.async_get_clientsession",
        return_value=_session(_all_ok(**{url_70: aiohttp.ClientError("boom")})),
    ):
        await _refresh(coordinator)
    assert coordinator.last_update_success is True
    assert set(coordinator.data.images) == {"240km"}


@pytest.mark.asyncio
async def test_coordinator_all_failed_without_cache_fails():
    coordinator = RadarCoordinator(MagicMock())
    responses = {RADAR_URL.format(range=r): _response(status=503) for r in RADAR_RANGES}
    with patch(
        "custom_components.nea_sg_weather.radar.async_get_clientsession",
        return_value=_session(responses),
    ):
        await _refresh(coordinator)
    assert coordinator.last_update_success is False


@pytest.mark.asyncio
async def test_coordinator_parse_failure_propagates():
    coordinator = RadarCoordinator(MagicMock())
    url_70 = RADAR_URL.format(range="70km")
    bad = _response(json_payload={"code": 0, "data": {"records": []}})
    with patch(
        "custom_components.nea_sg_weather.radar.async_get_clientsession",
        return_value=_session(_all_ok(**{url_70: bad})),
    ):
        await _refresh(coordinator)
    assert coordinator.last_update_success is False


def _coordinator(data):
    coordinator = MagicMock()
    coordinator.data = data
    coordinator.last_update_success = True
    return coordinator


def _data(*ranges, timestamp="2026-11-13T22:25:00+08:00", url_suffix="") -> RadarData:
    return RadarData(
        images={
            r: _parse_radar(
                _payload(r, f"https://s3.example/{r}.png{url_suffix}", timestamp), r
            )
            for r in ranges
        }
    )


def test_radar_camera_ids_attrs_and_availability():
    data = _data("70km")
    present = NeaRainRadarCamera(_coordinator(data), "entry1", PREFIX, "70km")
    missing = NeaRainRadarCamera(_coordinator(data), "entry1", PREFIX, "240km")

    # IDs from the configured prefix, like the integration's other entities.
    assert present._attr_unique_id == "Singapore Weather Rain Radar 70km"
    assert present.entity_id == "camera.singapore_weather_rain_radar_70km"
    assert present._attr_translation_key == "rain_radar"
    assert present._attr_translation_placeholders == {"range": "70 km"}
    assert present.available is True
    attrs = present.extra_state_attributes
    assert attrs["range"] == "70 km"
    assert attrs["image_timestamp"] == "2026-11-13T22:25:00+08:00"
    assert attrs["north"] == pytest.approx(1.97854)
    assert attrs["east"] == pytest.approx(104.602315)
    assert missing.available is False
    assert missing.extra_state_attributes == {"range": "240 km"}
    # On the config entry's main device, with the existing rain map cameras.
    assert present._attr_device_info["identifiers"] == {("nea_sg_weather", "entry1")}


def _png(size, pixels=None, mode="RGBA") -> bytes:
    """A tiny image: transparent (or black) with the given {(x, y): colour} pixels."""
    image = Image.new(mode, size)
    for xy, colour in (pixels or {}).items():
        image.putpixel(xy, colour)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


RED = (255, 0, 0, 255)
OVERLAY = _png((480, 480), {(83 + 108, 193 + 60): RED, (10, 10): RED})
BASE_70 = _png((853, 479), mode="RGB")
BASE_240 = _png((853, 853), mode="RGB")


def test_composite_radar_uses_nea_crop_and_scale():
    out = composite_radar(BASE_70, OVERLAY, BASEMAPS["70km"])
    image = Image.open(io.BytesIO(out))
    assert image.format == "JPEG"
    assert image.size == (853, 479)
    # Crop pixel (108, 60) of 217 px wide, scaled to 853 px: about (425, 236).
    r, g, b = image.getpixel((426, 237))
    assert r > 100 and g < 60 and b < 60
    # (10, 10) of the radar image lies outside NEA's 70 km crop.
    assert max(image.getpixel((5, 5))) < 30


def test_composite_radar_full_frame_for_240km():
    out = composite_radar(BASE_240, OVERLAY, BASEMAPS["240km"])
    image = Image.open(io.BytesIO(out))
    assert image.size == (853, 853)
    # (10, 10) * 853 / 480 is about (18, 18); one pixel blurs when scaled, so
    # look at the neighbourhood.
    reds = [image.getpixel((x, y))[0] for x in range(16, 22) for y in range(16, 22)]
    assert max(reds) > 100


def test_basemaps_cover_70_and_240_only():
    assert set(BASEMAPS) == {"70km", "240km"}
    assert BASEMAPS["70km"].url == (
        "https://www.nea.gov.sg/assets/images/map/base-853.png"
    )


def _routed_session(routes: dict):
    """Session answering GET per URL with bytes (200) or an int status."""
    session = MagicMock()

    def _get(url, **_kwargs):
        result = routes[url]
        if isinstance(result, int):
            return _response(status=result)
        return _response(body=result)

    session.get = MagicMock(side_effect=_get)
    return session


def _hass():
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda func, *args: func(*args))
    return hass


def _urls(session) -> list[str]:
    return [call.args[0] for call in session.get.call_args_list]


@pytest.mark.asyncio
async def test_radar_image_cached_by_capture_time():
    coordinator = _coordinator(RadarData())
    camera = NeaRainRadarCamera(coordinator, "entry1", PREFIX, "70km")
    camera.hass = _hass()
    routes = {BASEMAPS["70km"].url: BASE_70}
    routes.update(
        {f"https://s3.example/70km.png{sig}": OVERLAY for sig in ("", "?sig=2")}
    )
    session = _routed_session(routes)

    with patch(
        "custom_components.nea_sg_weather.camera.async_get_clientsession",
        return_value=session,
    ):
        assert await camera.async_camera_image() is None  # no data yet
        assert session.get.call_count == 0

        coordinator.data = _data("70km")
        first = await camera.async_camera_image()
        assert camera.content_type == "image/jpeg"
        # Same capture, freshly re-signed URL: no re-download.
        coordinator.data = _data("70km", url_suffix="?sig=2")
        assert await camera.async_camera_image() is first
        assert _urls(session).count("https://s3.example/70km.png") == 1

        coordinator.data = _data("70km", timestamp="2026-11-13T22:30:00+08:00")
        await camera.async_camera_image()
        assert _urls(session).count("https://s3.example/70km.png") == 2


@pytest.mark.asyncio
async def test_radar_70km_drawn_on_nea_basemap_fetched_once():
    coordinator = _coordinator(_data("70km"))
    camera = NeaRainRadarCamera(coordinator, "entry1", PREFIX, "70km")
    camera.hass = _hass()
    session = _routed_session(
        {
            BASEMAPS["70km"].url: BASE_70,
            "https://s3.example/70km.png": OVERLAY,
        }
    )

    with patch(
        "custom_components.nea_sg_weather.camera.async_get_clientsession",
        return_value=session,
    ):
        image = await camera.async_camera_image()
        assert Image.open(io.BytesIO(image)).size == (853, 479)
        assert camera.content_type == "image/jpeg"

        coordinator.data = _data("70km", timestamp="2026-11-13T22:30:00+08:00")
        await camera.async_camera_image()

    assert _urls(session).count(BASEMAPS["70km"].url) == 1
    assert _urls(session).count("https://s3.example/70km.png") == 2


@pytest.mark.asyncio
async def test_radar_basemap_failure_serves_overlay_and_retries_later():
    coordinator = _coordinator(_data("240km"))
    camera = NeaRainRadarCamera(coordinator, "entry1", PREFIX, "240km")
    camera.hass = _hass()
    routes = {BASEMAPS["240km"].url: 503, "https://s3.example/240km.png": OVERLAY}
    session = _routed_session(routes)
    clock = [1000.0]

    with (
        patch(
            "custom_components.nea_sg_weather.camera.async_get_clientsession",
            return_value=session,
        ),
        patch(
            "custom_components.nea_sg_weather.camera.time.monotonic",
            side_effect=lambda: clock[0],
        ),
    ):
        assert await camera.async_camera_image() == OVERLAY
        assert camera.content_type == "image/png"
        # Within the retry window the base map isn't requested again.
        await camera.async_camera_image()
        assert _urls(session).count(BASEMAPS["240km"].url) == 1

        routes[BASEMAPS["240km"].url] = BASE_240
        clock[0] += BASEMAP_RETRY_SECONDS + 1
        image = await camera.async_camera_image()

    assert Image.open(io.BytesIO(image)).size == (853, 853)
    assert camera.content_type == "image/jpeg"
    assert _urls(session).count(BASEMAPS["240km"].url) == 2


@pytest.mark.asyncio
async def test_radar_corrupt_basemap_falls_back_to_overlay():
    coordinator = _coordinator(_data("70km"))
    camera = NeaRainRadarCamera(coordinator, "entry1", PREFIX, "70km")
    camera.hass = _hass()
    session = _routed_session(
        {
            BASEMAPS["70km"].url: b"<html>not an image</html>",
            "https://s3.example/70km.png": OVERLAY,
        }
    )
    with patch(
        "custom_components.nea_sg_weather.camera.async_get_clientsession",
        return_value=session,
    ):
        assert await camera.async_camera_image() == OVERLAY
    assert camera.content_type == "image/png"
    assert camera._basemap is None


@pytest.mark.asyncio
async def test_radar_image_download_failure_returns_cached():
    coordinator = _coordinator(_data("70km"))
    camera = NeaRainRadarCamera(coordinator, "entry1", PREFIX, "70km")
    camera.hass = _hass()

    with patch(
        "custom_components.nea_sg_weather.camera.async_get_clientsession",
        return_value=MagicMock(get=MagicMock(return_value=_response(body=b"ok"))),
    ):
        assert await camera.async_camera_image() == b"ok"

    coordinator.data = _data("70km", timestamp="2026-11-13T22:30:00+08:00")
    with patch(
        "custom_components.nea_sg_weather.camera.async_get_clientsession",
        return_value=MagicMock(get=MagicMock(return_value=_response(status=403))),
    ):
        assert await camera.async_camera_image() == b"ok"


# --- Animation: frames from the day's list --------------------------------------


def _day_payload(times: list[str], radar_range: str = "70km", sig: str = "") -> dict:
    """A ``?date=`` response: one record per ``HH:MM`` on 2026-11-13, newest first."""
    payload = _payload(radar_range)
    payload["data"]["records"] = [
        {
            "timestamp": f"2026-11-13T{hhmm}:00+08:00",
            "image": {"url": f"https://s3.example/{radar_range}-{hhmm}.png{sig}"},
        }
        for hhmm in times
    ]
    return payload


def _times(newest: str, count: int, step: int = 5) -> list[str]:
    """``count`` HH:MM strings ``step`` minutes apart, newest first."""
    hour, minute = map(int, newest.split(":"))
    start = datetime(2026, 11, 13, hour, minute)
    return [f"{start - timedelta(minutes=step * i):%H:%M}" for i in range(count)]


def test_parse_radar_frames_are_last_hour_oldest_first():
    image = _parse_radar(_day_payload(_times("22:25", 25)), "70km")
    # 60 minutes at 5-minute steps: 12 frames, the latest image included.
    assert len(image.frames) == 12
    assert [f.timestamp.strftime("%H:%M") for f in image.frames[:2]] == [
        "21:30",
        "21:35",
    ]
    assert image.frames[-1].timestamp == image.timestamp
    assert image.frames[-1].image_url == image.image_url
    assert image.frames[-1].timestamp - image.frames[0].timestamp < ANIMATION_WINDOW


def test_parse_radar_frames_skip_records_without_url_or_timestamp():
    payload = _day_payload(["22:25", "22:20", "22:15"])
    records = payload["data"]["records"]
    del records[1]["image"]
    records[2]["timestamp"] = "not a time"
    image = _parse_radar(payload, "70km")
    assert [f.timestamp.strftime("%H:%M") for f in image.frames] == ["22:25"]


def test_parse_radar_single_record_has_one_frame():
    image = _parse_radar(_payload("70km"), "70km")
    assert len(image.frames) == 1
    assert image.frames[0].image_url == image.image_url


def test_parse_radar_merges_previous_day_and_prefers_todays_urls():
    today = _day_payload(["00:10", "00:05"], sig="?today")
    today["data"]["records"].append(
        {  # also on the previous day's page: today's URL wins
            "timestamp": "2026-11-13T00:00:00+08:00",
            "image": {"url": "https://s3.example/today-0000.png"},
        }
    )
    earlier = _payload("70km")
    earlier["data"]["records"] = [
        {
            "timestamp": f"2026-11-12T{hhmm}:00+08:00",
            "image": {"url": f"https://s3.example/earlier-{hhmm}.png"},
        }
        for hhmm in ["23:55", "23:50", "23:15", "23:10"]
    ] + [
        {
            "timestamp": "2026-11-13T00:00:00+08:00",
            "image": {"url": "https://s3.example/earlier-0000.png"},
        }
    ]
    image = _parse_radar(today, "70km", earlier)
    assert image.image_url.endswith("00:10.png?today")
    labels = [f.timestamp.strftime("%d %H:%M") for f in image.frames]
    # 23:10 is exactly an hour before 00:10: outside the window.
    assert labels == [
        "12 23:15",
        "12 23:50",
        "12 23:55",
        "13 00:00",
        "13 00:05",
        "13 00:10",
    ]
    assert image.frames[3].image_url == "https://s3.example/today-0000.png"


def test_parse_radar_empty_today_uses_previous_day():
    today = {"code": 0, "data": {"records": []}}
    earlier = _day_payload(["23:55", "23:50"])
    image = _parse_radar(today, "70km", earlier)
    assert image.timestamp.strftime("%H:%M") == "23:55"
    assert image.north == pytest.approx(1.97854)
    assert len(image.frames) == 2


def _capturing_session(responses: dict):
    """Session answering per (url, date param); records every (url, params) call."""
    session = MagicMock()
    calls: list[tuple[str, dict]] = []

    def _get(url, **kwargs):
        params = kwargs.get("params") or {}
        calls.append((url, params))
        result = responses.get((url, params.get("date")), responses.get(url))
        if isinstance(result, Exception):
            raise result
        return result

    session.get = MagicMock(side_effect=_get)
    session.calls = calls
    return session


@pytest.mark.asyncio
async def test_coordinator_asks_for_todays_list_once_per_range_at_midday():
    coordinator = RadarCoordinator(MagicMock())
    session = _capturing_session(_all_ok())
    with patch(
        "custom_components.nea_sg_weather.radar.async_get_clientsession",
        return_value=session,
    ):
        await _refresh(coordinator)
    # Still one request per range: frames cost no extra API calls.
    assert session.calls == [
        (RADAR_URL.format(range=r), {"date": "2026-11-13"}) for r in RADAR_RANGES
    ]


@pytest.mark.asyncio
async def test_coordinator_fetches_previous_day_just_after_midnight(pinned_now):
    pinned_now[0] = datetime(2026, 11, 14, 0, 20, tzinfo=SGT)
    coordinator = RadarCoordinator(MagicMock())
    url_70 = RADAR_URL.format(range="70km")
    today = _payload("70km", timestamp="2026-11-14T00:15:00+08:00")
    earlier = _day_payload(["23:55", "23:50"])
    responses = _all_ok()
    responses[(url_70, "2026-11-14")] = _response(json_payload=today)
    responses[(url_70, "2026-11-13")] = _response(json_payload=earlier)
    session = _capturing_session(responses)
    with patch(
        "custom_components.nea_sg_weather.radar.async_get_clientsession",
        return_value=session,
    ):
        await _refresh(coordinator)
    assert coordinator.last_update_success is True
    assert (url_70, {"date": "2026-11-13"}) in session.calls
    assert len(session.calls) == 2 * len(RADAR_RANGES)
    assert len(coordinator.data.images["70km"].frames) == 3


@pytest.mark.asyncio
async def test_coordinator_previous_day_failure_is_not_an_error(pinned_now):
    pinned_now[0] = datetime(2026, 11, 14, 0, 20, tzinfo=SGT)
    coordinator = RadarCoordinator(MagicMock())
    responses = _all_ok()
    for r in RADAR_RANGES:
        responses[(RADAR_URL.format(range=r), "2026-11-13")] = _response(status=429)
    with patch(
        "custom_components.nea_sg_weather.radar.async_get_clientsession",
        return_value=_capturing_session(responses),
    ):
        await _refresh(coordinator)
    assert coordinator.last_update_success is True
    assert set(coordinator.data.images) == set(RADAR_RANGES)


@pytest.mark.asyncio
async def test_coordinator_skips_previous_day_after_the_first_hour(pinned_now):
    pinned_now[0] = datetime(2026, 11, 14, 1, 30, tzinfo=SGT)
    coordinator = RadarCoordinator(MagicMock())
    session = _capturing_session(_all_ok())
    with patch(
        "custom_components.nea_sg_weather.radar.async_get_clientsession",
        return_value=session,
    ):
        await _refresh(coordinator)
    assert {params["date"] for _, params in session.calls} == {"2026-11-14"}


# --- Animation: GIF rendering ----------------------------------------------------


def _overlay_at(xy) -> bytes:
    return _png((480, 480), {xy: RED})


def test_animate_radar_loops_every_frame_with_a_hold_on_the_latest():
    frames = [("21:30", _overlay_at((10, 10))), ("21:35", _overlay_at((20, 20)))]
    frames.append(("21:40", OVERLAY))
    gif = animate_radar(BASE_70, frames, BASEMAPS["70km"])
    image = Image.open(io.BytesIO(gif))
    assert image.format == "GIF"
    assert image.size == (853, 479)
    assert image.n_frames == 3
    assert image.info["loop"] == 0
    durations = []
    for index in range(image.n_frames):
        image.seek(index)
        durations.append(image.info["duration"])
    assert durations == [FRAME_MS, FRAME_MS, HOLD_MS]


def test_animate_radar_draws_rain_and_time_label():
    base = _png((853, 853), mode="RGB")
    rain = _png(
        (480, 480), {(x, y): RED for x in range(200, 210) for y in range(200, 210)}
    )
    gif = animate_radar(base, [("22:25", rain)], BASEMAPS["240km"])
    frame = Image.open(io.BytesIO(gif)).convert("RGB")
    # 240 km uses the whole image, scaled from 480 to 853 px wide.
    centre = round(205 * 853 / 480)
    assert frame.getpixel((centre, centre))[0] > 200
    assert frame.getpixel((600, 600)) == (0, 0, 0)
    # The label is white text: some bright pixels inside the top-left box.
    box = frame.crop((8, 8, 60, 30))
    assert box.getextrema()[0][1] > 200


def test_animate_radar_without_frames_raises():
    with pytest.raises(ValueError):
        animate_radar(BASE_70, [], BASEMAPS["70km"])


def test_animate_radar_corrupt_input_raises_oserror():
    with pytest.raises(OSError):
        animate_radar(b"<html>", [("22:25", OVERLAY)], BASEMAPS["70km"])


# --- Animation: camera entity ---------------------------------------------------


def _anim_data(times: list[str], radar_range: str = "70km", sig: str = "") -> RadarData:
    payload = _day_payload(times, radar_range, sig)
    return RadarData(images={radar_range: _parse_radar(payload, radar_range)})


def _frame_routes(times: list[str], radar_range: str = "70km", sig: str = "") -> dict:
    return {f"https://s3.example/{radar_range}-{t}.png{sig}": OVERLAY for t in times}


@pytest.mark.asyncio
async def test_setup_keeps_rain_map_cameras_and_adds_radar_cameras():
    hass = MagicMock()
    hass.data = {"nea_sg_weather": {"entry1": MagicMock()}}
    entry = MagicMock()
    entry.entry_id = "entry1"
    entry.data = {"sensors": {"prefix": PREFIX}}
    add = MagicMock()
    with patch.object(camera_module, "RadarCoordinator") as radar_cls:
        await async_setup_entry(hass, entry, add)

    entities = [e for call in add.call_args_list for e in call.args[0]]
    # The existing weather.gov.sg rain map cameras are still there.
    assert sum(isinstance(e, NeaRainCamera) for e in entities) == 1
    assert sum(isinstance(e, NeaAnimatedRainCamera) for e in entities) == 1
    radar = [e for e in entities if isinstance(e, NeaRainRadarCamera)]
    assert sorted(e.entity_id for e in radar) == [
        "camera.singapore_weather_rain_radar_240km",
        "camera.singapore_weather_rain_radar_70km",
        "camera.singapore_weather_rain_radar_animation_240km",
        "camera.singapore_weather_rain_radar_animation_70km",
    ]
    # One radar coordinator, tied to the entry; first refresh in the background.
    radar_cls.assert_called_once_with(hass, entry)
    entry.async_create_background_task.assert_called_once()


def test_animation_camera_ids_attrs_and_availability():
    camera = NeaRainRadarAnimationCamera(
        _coordinator(_anim_data(_times("22:25", 3))), "entry1", PREFIX, "70km"
    )
    assert camera._attr_unique_id == "Singapore Weather Rain Radar Animation 70km"
    assert camera.entity_id == "camera.singapore_weather_rain_radar_animation_70km"
    assert camera._attr_translation_key == "rain_radar_animation"
    assert camera._attr_translation_placeholders == {"range": "70 km"}
    assert camera.content_type == "image/gif"
    assert camera.available is True
    attrs = camera.extra_state_attributes
    assert attrs["frame_count"] == 3
    assert attrs["first_frame_timestamp"] == "2026-11-13T22:15:00+08:00"
    assert attrs["image_timestamp"] == "2026-11-13T22:25:00+08:00"
    assert camera._attr_device_info["identifiers"] == {("nea_sg_weather", "entry1")}

    empty = NeaRainRadarAnimationCamera(_coordinator(RadarData()), "e", PREFIX, "70km")
    assert empty.available is False
    assert empty.extra_state_attributes["frame_count"] == 0


@pytest.mark.asyncio
async def test_animation_downloads_each_frame_once_and_rebuilds_on_new_frame():
    times = _times("22:25", 12)
    coordinator = _coordinator(_anim_data(times))
    camera = NeaRainRadarAnimationCamera(coordinator, "entry1", PREFIX, "70km")
    camera.hass = _hass()
    routes = {BASEMAPS["70km"].url: BASE_70, **_frame_routes(_times("22:30", 13))}
    session = _routed_session(routes)

    with patch(
        "custom_components.nea_sg_weather.camera.async_get_clientsession",
        return_value=session,
    ):
        gif = await camera.async_camera_image()
        assert Image.open(io.BytesIO(gif)).n_frames == 12
        assert camera.content_type == "image/gif"
        calls = session.get.call_count  # base map + 12 frames
        assert calls == 13
        # Same frames: the cached GIF, no downloads, no re-render.
        assert await camera.async_camera_image() is gif
        assert session.get.call_count == calls

        # Next poll: one new frame in, the oldest out; only the new one is fetched.
        coordinator.data = _anim_data(_times("22:30", 12))
        gif2 = await camera.async_camera_image()
    assert gif2 is not gif
    assert session.get.call_count == calls + 1
    assert _urls(session)[-1] == "https://s3.example/70km-22:30.png"
    assert len(camera._overlays) == 12


@pytest.mark.asyncio
async def test_animation_missing_frame_is_skipped_then_retried():
    times = _times("22:25", 3)
    coordinator = _coordinator(_anim_data(times))
    camera = NeaRainRadarAnimationCamera(coordinator, "entry1", PREFIX, "70km")
    camera.hass = _hass()
    routes = {BASEMAPS["70km"].url: BASE_70, **_frame_routes(times)}
    routes["https://s3.example/70km-22:20.png"] = 403
    session = _routed_session(routes)

    with patch(
        "custom_components.nea_sg_weather.camera.async_get_clientsession",
        return_value=session,
    ):
        gif = await camera.async_camera_image()
        assert Image.open(io.BytesIO(gif)).n_frames == 2
        routes["https://s3.example/70km-22:20.png"] = OVERLAY
        gif = await camera.async_camera_image()
    assert Image.open(io.BytesIO(gif)).n_frames == 3


@pytest.mark.asyncio
async def test_animation_without_basemap_serves_nothing_new():
    coordinator = _coordinator(_anim_data(_times("22:25", 2)))
    camera = NeaRainRadarAnimationCamera(coordinator, "entry1", PREFIX, "70km")
    camera.hass = _hass()
    session = _routed_session({BASEMAPS["70km"].url: 503})
    with patch(
        "custom_components.nea_sg_weather.camera.async_get_clientsession",
        return_value=session,
    ):
        assert await camera.async_camera_image() is None
    # No frames downloaded while there is no map to draw them on.
    assert _urls(session) == [BASEMAPS["70km"].url]


@pytest.mark.asyncio
async def test_animation_corrupt_basemap_keeps_last_gif_and_refetches_later():
    times = _times("22:25", 2)
    coordinator = _coordinator(_anim_data(times))
    camera = NeaRainRadarAnimationCamera(coordinator, "entry1", PREFIX, "70km")
    camera.hass = _hass()
    routes = {BASEMAPS["70km"].url: BASE_70, **_frame_routes(_times("22:30", 3))}
    session = _routed_session(routes)
    with patch(
        "custom_components.nea_sg_weather.camera.async_get_clientsession",
        return_value=session,
    ):
        gif = await camera.async_camera_image()
        # The map turns corrupt (e.g. NEA served an HTML page) and a new frame arrives.
        camera._basemap = b"<html>"
        coordinator.data = _anim_data(_times("22:30", 2))
        assert await camera.async_camera_image() is gif
    assert camera._basemap is None
    assert camera._basemap_failed_at is not None
    assert camera._overlays == {}
