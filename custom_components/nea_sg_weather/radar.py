"""Coordinator for NEA rain radar images from data.gov.sg (still + animation frames).

Ported from the ``singapore`` integration (kianhean/singapore-homeassistant). Separate
from the main weather coordinator: it polls every 5 minutes (NEA's radar cadence)
whatever the configured scan interval, and only exists while the camera platform is
loaded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Final

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

_LOGGER = logging.getLogger(__name__)

RADAR_URL = "https://api-open.data.gov.sg/v2/real-time/api/weather-radar-images/{range}"
# NEA publishes a new radar image every 5 minutes. The image URLs are presigned
# and expire ~20 minutes after the API call, so polling also keeps them fresh.
UPDATE_INTERVAL = timedelta(minutes=5)
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)

# With ``?date=YYYY-MM-DD`` (Singapore date) the same endpoint returns that day's
# images newest first, ~2 hours (25 records) on the first page, each with its own
# presigned URL. One request per range therefore gives both the latest image and the
# frames of the animation, so the animated cameras cost no extra API calls.
ANIMATION_WINDOW = timedelta(hours=1)
# Until this long after midnight today's list is too short for a full loop, so the
# previous day's images are fetched as well.
_EARLIER_DAY_UNTIL = ANIMATION_WINDOW + timedelta(minutes=10)
SGT = timezone(timedelta(hours=8))  # Singapore has no DST

# Radar range (API path segment) -> entity name suffix. Only the ranges NEA has a
# base map for (see radar_image.BASEMAPS); data.gov.sg's 480 km image would be
# coloured blobs with no map.
RADAR_RANGES: Final[dict[str, str]] = {
    "70km": "70 km",
    "240km": "240 km",
}


@dataclass(frozen=True)
class RadarFrame:
    """One radar image of the animation loop."""

    timestamp: datetime
    image_url: str


@dataclass
class RadarImage:
    """Latest radar image for one range, plus the last hour's frames."""

    range: str
    image_url: str
    timestamp: datetime | None
    url_expires_at: datetime | None = None
    # Bounding box of the image in EPSG:4326.
    north: float | None = None
    south: float | None = None
    east: float | None = None
    west: float | None = None
    center_latitude: float | None = None
    center_longitude: float | None = None
    # Images within ``ANIMATION_WINDOW`` of the latest one, oldest first (the latest
    # included). Empty when NEA gave no usable timestamps.
    frames: list[RadarFrame] = field(default_factory=list)


@dataclass
class RadarData:
    """Latest radar images keyed by range (``70km`` / ``240km``)."""

    images: dict[str, RadarImage] = field(default_factory=dict)


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _records(payload: object) -> list:
    data = payload.get("data") if isinstance(payload, dict) else None
    records = data.get("records") if isinstance(data, dict) else None
    return records if isinstance(records, list) else []


def _record_url(record: object) -> str | None:
    image = record.get("image") if isinstance(record, dict) else None
    url = image.get("url") if isinstance(image, dict) else None
    return url if isinstance(url, str) and url else None


def _frames(records: list, latest: datetime | None) -> list[RadarFrame]:
    """The records within ``ANIMATION_WINDOW`` of ``latest``, oldest first.

    Records are newest first, so for a timestamp listed twice (today's and the
    previous day's pages overlap) the first, freshest URL wins.
    """
    frames: dict[datetime, RadarFrame] = {}
    for record in records:
        url = _record_url(record)
        stamp = _parse_datetime(record.get("timestamp")) if url else None
        if url is None or stamp is None or stamp.tzinfo is None:
            continue
        frames.setdefault(stamp, RadarFrame(stamp, url))
    if not frames:
        return []
    newest = latest if latest is not None and latest.tzinfo else max(frames)
    return sorted(
        (
            f
            for f in frames.values()
            if newest - ANIMATION_WINDOW < f.timestamp <= newest
        ),
        key=lambda f: f.timestamp,
    )


def parse_radar(
    payload: dict, radar_range: str, earlier: dict | None = None
) -> RadarImage:
    """Parse one weather-radar-images response (plus, optionally, the day before's).

    Expected shape::

        {"code": 0, "data": {
            "projection": {"center": {"longitude": 103.97, "latitude": 1.35}},
            "boundaryBox": {"upperLeft": {"longitude": .., "latitude": ..},
                            "lowerRight": {"longitude": .., "latitude": ..}},
            "records": [{"timestamp": "...", "image": {"url": "https://...png",
                                                       "urlExpiresAt": "..."}}]}}

    ``records`` are newest first; the first is the latest image. ``earlier`` is the
    previous day's response, used just after midnight when today's list is short
    (or still empty).

    Raises ``UpdateFailed`` when there is no record with an image URL.
    """
    records = _records(payload)
    earlier_records = _records(earlier)
    source = payload if records else earlier
    primary = records or earlier_records
    if not primary:
        raise UpdateFailed(f"Radar {radar_range} payload has no records")
    record = primary[0]
    url = _record_url(record)
    if url is None:
        raise UpdateFailed(f"Radar {radar_range} payload has no image URL")
    image = record["image"]
    timestamp = _parse_datetime(record.get("timestamp"))

    data = source["data"]  # type: ignore[index]  # has records, so a dict
    box = data.get("boundaryBox") or {}
    upper_left = box.get("upperLeft") or {}
    lower_right = box.get("lowerRight") or {}
    center = (data.get("projection") or {}).get("center") or {}
    return RadarImage(
        range=radar_range,
        image_url=url,
        timestamp=timestamp,
        url_expires_at=_parse_datetime(image.get("urlExpiresAt")),
        north=_float(upper_left.get("latitude")),
        south=_float(lower_right.get("latitude")),
        east=_float(lower_right.get("longitude")),
        west=_float(upper_left.get("longitude")),
        center_latitude=_float(center.get("latitude")),
        center_longitude=_float(center.get("longitude")),
        frames=_frames(records + earlier_records, timestamp),
    )


def _now() -> datetime:
    """Current Singapore time (a function so tests can pin it)."""
    return datetime.now(SGT)


class RadarCoordinator(DataUpdateCoordinator[RadarData]):
    """Polls the latest NEA radar image (and the last hour's frames) per range."""

    def __init__(
        self, hass: HomeAssistant, config_entry: ConfigEntry | None = None
    ) -> None:
        super().__init__(
            hass,
            _LOGGER,
            config_entry=config_entry,
            name="NEA rain radar",
            update_interval=UPDATE_INTERVAL,
        )

    async def _fetch_range(
        self, session: aiohttp.ClientSession, radar_range: str, day: date
    ) -> dict:
        async with session.get(
            RADAR_URL.format(range=radar_range),
            params={"date": day.isoformat()},
            timeout=_REQUEST_TIMEOUT,
        ) as response:
            if response.status != 200:
                raise UpdateFailed(
                    f"data.gov.sg radar {radar_range} returned HTTP {response.status}"
                )
            return await response.json(content_type=None)

    async def _async_update_data(self) -> RadarData:
        session = async_get_clientsession(self.hass)
        previous = self.data.images if self.data is not None else {}
        images: dict[str, RadarImage] = {}
        errors: list[str] = []
        now = _now()
        today = now.date()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        early = now - midnight < _EARLIER_DAY_UNTIL

        # Sequential on purpose: data.gov.sg rate-limits bursts (HTTP 429).
        for radar_range in RADAR_RANGES:
            try:
                payload = await self._fetch_range(session, radar_range, today)
            except (aiohttp.ClientError, TimeoutError, UpdateFailed) as err:
                # Transient fetch problem (a 429 included): keep this range's last
                # image, if any.
                errors.append(f"{radar_range}: {err}")
                if radar_range in previous:
                    images[radar_range] = previous[radar_range]
                continue
            earlier = None
            if early:
                # Only lengthens the animation loop: a failure here is not an error.
                try:
                    earlier = await self._fetch_range(
                        session, radar_range, today - timedelta(days=1)
                    )
                except (aiohttp.ClientError, TimeoutError, UpdateFailed) as err:
                    _LOGGER.debug(
                        "Radar %s: previous day not fetched: %s", radar_range, err
                    )
            # Parse failures propagate so a changed payload is reported.
            images[radar_range] = parse_radar(payload, radar_range, earlier)

        if errors:
            if not images:
                raise UpdateFailed(f"Error fetching radar images: {'; '.join(errors)}")
            _LOGGER.warning(
                "Error fetching radar images (%s); using last known values",
                "; ".join(errors),
            )
        return RadarData(images=images)
