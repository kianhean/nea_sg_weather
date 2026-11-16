"""Support for retrieving weather data from NEA."""
from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any
from datetime import datetime, timezone, timedelta
import math
import time
import aiohttp
import httpx
from PIL import Image
import io

from homeassistant.components.camera import Camera, CameraEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_PREFIX, CONF_SENSORS
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from . import NeaWeatherDataUpdateCoordinator
from .entity import main_device_info
from .const import (
    DOMAIN,
    RAIN_MAP_HEADERS,
    RAIN_MAP_URL_PREFIX,
    RAIN_MAP_URL_SUFFIX,
    RAIN_MAP_GIF_URL,
)
from .radar import RADAR_RANGES, SGT, RadarCoordinator, RadarImage
from .radar_image import BASEMAPS, animate_radar, composite_radar

_LOGGER = logging.getLogger(__name__)

# How long to serve the bare radar overlay before trying NEA's base map again.
BASEMAP_RETRY_SECONDS = 30 * 60


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Add a weather camera entity from a config_entry."""
    coordinator: NeaWeatherDataUpdateCoordinator = hass.data[DOMAIN][
        config_entry.entry_id
    ]
    entry_id = config_entry.entry_id

    async_add_entities([NeaRainCamera(hass, coordinator, config_entry.data, entry_id)])
    async_add_entities([NeaAnimatedRainCamera(hass, coordinator, config_entry.data, entry_id)])

    # Rain radar on NEA's own base maps (still + last-hour animation), from
    # data.gov.sg. Its own coordinator: a 5-minute cadence whatever the scan
    # interval, shut down with the config entry. The first refresh runs in the
    # background so a data.gov.sg outage never blocks setup.
    radar = RadarCoordinator(hass, config_entry)
    config_entry.async_create_background_task(
        hass, radar.async_refresh(), f"{DOMAIN} rain radar first refresh"
    )
    prefix = config_entry.data[CONF_SENSORS][CONF_PREFIX]
    async_add_entities(
        [NeaRainRadarCamera(radar, entry_id, prefix, r) for r in RADAR_RANGES]
        + [NeaRainRadarAnimationCamera(radar, entry_id, prefix, r) for r in RADAR_RANGES]
    )


class NeaRainCamera(Camera):
    """Implementation of a camera entity for rain map overlay."""

    _attr_has_entity_name = True
    _attr_translation_key = "rain_map"

    def __init__(
        self,
        hass: HomeAssistant,
        coordinator,
        config: MappingProxyType[str, Any],
        entry_id: str,
    ) -> None:
        """Initialise area sensor with a data instance and site."""
        super().__init__()
        self.hass = hass
        self.coordinator = coordinator
        self._limit_refetch = True
        self.content_type = "image/png"
        self.verify_ssl = True
        self._last_query_time = None
        self._last_image_time = None
        self._last_image_time_pretty = None
        self._last_image = None
        self._last_url = None
        self._platform = "camera"
        self._prefix = config[CONF_SENSORS][CONF_PREFIX]
        self._entry_id = entry_id
        self._attr_device_info = main_device_info(entry_id)
        self.entity_id = (
            (self._platform + "." + self._prefix + "_rain_map")
            .lower()
            .replace(" ", "_")
        )
        self._last_state = None
        self._last_attributes = None
        self._updated_attributes = None

    @property
    def unique_id(self):
        """Return the unique ID."""
        return self._prefix + " Rain Map"

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Return a still image response from the camera."""

        async def get_image(current_image_time: int) -> bytes | None:
            # get initial set of images
            next_image_url = (
                RAIN_MAP_URL_PREFIX + str(current_image_time) + RAIN_MAP_URL_SUFFIX
            )
            try:
                _LOGGER.debug("Getting rain map image from %s", next_image_url)
                async_client = get_async_client(self.hass, verify_ssl=self.verify_ssl)
                response = await async_client.get(
                    next_image_url, headers=RAIN_MAP_HEADERS
                )
                response.raise_for_status()
                _LOGGER.debug(
                    "Rain map image successfully updated at %s, new URL is %s",
                    datetime.strptime(
                        str(current_image_time), "%Y%m%d%H%M"
                    ).isoformat(),
                    next_image_url,
                )
                self._last_image = response.content
                self._last_image_time = current_image_time
                self._last_image_time_pretty = datetime.strptime(
                    str(current_image_time), "%Y%m%d%H%M"
                ).isoformat()
                self._last_url = next_image_url
                # Update timestamp from external coordinator entity
                self._last_state = self.hass.states.get(self.entity_id).state
                self._last_attributes = self.hass.states.get(self.entity_id).attributes
                self._updated_attributes = dict(self._last_attributes)
                self._updated_attributes["Updated at"] = self._last_image_time_pretty
                self._updated_attributes["URL"] = next_image_url
                self.hass.states.async_set(
                    self.entity_id, self._last_state, self._updated_attributes
                )
                return self._last_image

            except httpx.TimeoutException:
                _LOGGER.warning(
                    "Timeout getting camera image for %s from %s",
                    self.entity_id,
                    next_image_url,
                )
                return self._last_image

            except (httpx.HTTPStatusError, httpx.RequestError) as err:
                if response.status_code == 404:
                    # Image not ready, check older image urls
                    _LOGGER.debug(
                        "%s rain map image not ready, trying previous images",
                        current_image_time,
                    )

                    # minor fix for whole hours
                    if str(current_image_time)[-2:] == "00":
                        _skip_minutes = 45
                    else:
                        _skip_minutes = 5

                    if current_image_time - _skip_minutes == self._last_image_time:
                        return self._last_image
                    else:
                        return await get_image(current_image_time - _skip_minutes)
                else:
                    _LOGGER.warning(
                        "Error getting new camera image for %s from %s: %s",
                        self.entity_id,
                        next_image_url,
                        err,
                    )
                    return self._last_image

        _current_query_time = int(
            datetime.strftime(datetime.now(timezone(timedelta(hours=8))), "%Y%m%d%H%M")
        )

        if _current_query_time != self._last_query_time:
            self._last_query_time = _current_query_time
            _current_image_time = math.floor(_current_query_time / 5) * 5
            if _current_image_time != self._last_image_time:
                return await get_image(_current_image_time)

        return self._last_image

    async def stream_source(self):
        """Return the source of the stream."""
        return None

    @property
    def extra_state_attributes(self) -> dict:
        """Return dict of additional properties to attach to sensors."""
        return {
            "Updated at": self._last_image_time_pretty,
            "URL": self._last_url,
        }


class NeaAnimatedRainCamera(Camera):
    """Implementation of a camera entity for rain map overlay."""

    _attr_has_entity_name = True
    _attr_translation_key = "animated_rain_map"

    def __init__(
        self,
        hass: HomeAssistant,
        coordinator,
        config: MappingProxyType[str, Any],
        entry_id: str,
    ) -> None:
        """Initialise area sensor with a data instance and site."""
        super().__init__()
        self.hass = hass
        self.coordinator = coordinator
        self.content_type = "image/gif"
        self._last_query_time = None
        self._last_gif_time = None
        self._last_gif_time_pretty = None
        self._last_gif = None
        self._gifs = []
        self._last_url = None
        self.verify_ssl = True
        self._platform = "camera"
        self._prefix = config[CONF_SENSORS][CONF_PREFIX]
        self._entry_id = entry_id
        self._attr_device_info = main_device_info(entry_id)
        self.entity_id = (
            (self._platform + "." + self._prefix + "_animated_rain_map")
            .lower()
            .replace(" ", "_")
        )
        self._last_state = None
        self._last_attributes = None
        self._updated_attributes = None

    @property
    def unique_id(self):
        """Return the unique ID."""
        return self._prefix + " Animated Rain Map"

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Return an animated gif response from the camera."""

        async def get_image(current_gif_time: int) -> bytes | None:
            next_image_url = (
                RAIN_MAP_URL_PREFIX + str(current_gif_time) + RAIN_MAP_URL_SUFFIX
            )
            # get initial set of images
            try:
                if self._gifs == [] or (current_gif_time - self._last_gif_time > 5):
                    _LOGGER.debug("Getting initial images from %s", RAIN_MAP_GIF_URL)
                    async_client = get_async_client(
                        self.hass, verify_ssl=self.verify_ssl
                    )
                    response = await async_client.get(
                        RAIN_MAP_GIF_URL, headers=RAIN_MAP_HEADERS
                    )
                    response.raise_for_status()
                    initial_images_urls_str = response.text[
                        response.text.find('slideshowimages("')
                        + len('slideshowimages("') :
                    ]
                    initial_images_urls_str = initial_images_urls_str[
                        : initial_images_urls_str.find(");")
                    ]
                    initial_images_urls = initial_images_urls_str.replace(
                        '"', ""
                    ).split(",")
                    for next_image_url in initial_images_urls[1:]:  # skip first image
                        response = await async_client.get(
                            next_image_url, headers=RAIN_MAP_HEADERS
                        )
                        response.raise_for_status()
                        frame = Image.open(io.BytesIO(response.content))
                        self._gifs.append(frame)
                    _LOGGER.debug(
                        "Initial rain map images successfully updated at %s, %s frames downloaded",
                        datetime.strptime(
                            str(current_gif_time), "%Y%m%d%H%M"
                        ).isoformat(),
                        len(initial_images_urls),
                    )
                else:
                    _LOGGER.debug("Getting rain map image from %s", next_image_url)
                    async_client = get_async_client(
                        self.hass, verify_ssl=self.verify_ssl
                    )
                    response = await async_client.get(
                        next_image_url, headers=RAIN_MAP_HEADERS
                    )
                    response.raise_for_status()
                    frame = Image.open(io.BytesIO(response.content))
                    self._gifs.append(frame)
                    _LOGGER.debug(
                        "Rain map image successfully updated at %s, new URL is %s",
                        datetime.strptime(
                            str(current_gif_time), "%Y%m%d%H%M"
                        ).isoformat(),
                        next_image_url,
                    )
                self._last_gif_time = current_gif_time
                self._last_gif_time_pretty = datetime.strptime(
                    str(current_gif_time), "%Y%m%d%H%M"
                ).isoformat()
                self._last_url = next_image_url
                # Update timestamp from external coordinator entity
                self._last_state = self.hass.states.get(self.entity_id).state
                self._last_attributes = self.hass.states.get(self.entity_id).attributes
                self._updated_attributes = dict(self._last_attributes)
                self._updated_attributes["Updated at"] = self._last_gif_time_pretty
                self._updated_attributes["URL"] = next_image_url
                self.hass.states.async_set(
                    self.entity_id, self._last_state, self._updated_attributes
                )

            except httpx.TimeoutException:
                _LOGGER.warning(
                    "Timeout getting camera image for %s from %s",
                    self.entity_id,
                    next_image_url,
                )
                return self._last_gif

            except (httpx.HTTPStatusError, httpx.RequestError) as err:
                if response.status_code == 404:
                    # Image not ready, check older image urls
                    _LOGGER.debug(
                        "%s rain map image not ready, trying previous images",
                        current_gif_time,
                    )

                    # minor fix for whole hours
                    if str(current_gif_time)[-2:] == "00":
                        _skip_minutes = 45
                    else:
                        _skip_minutes = 5

                    if current_gif_time - _skip_minutes == self._last_gif_time:
                        return self._last_gif
                    else:
                        return await get_image(current_gif_time - _skip_minutes)
                else:
                    _LOGGER.warning(
                        "Error getting new camera image for %s from %s: %s",
                        self.entity_id,
                        next_image_url,
                        err,
                    )
                    return self._last_gif

            # created animated gif
            try:
                if len(self._gifs) > 24:  # drop older frames when we have enough
                    self._gifs = self._gifs[-24:]
                if len(self._gifs) == 24:
                    _LOGGER.debug(
                        "Converting %s frames into animated gif", len(self._gifs)
                    )
                    buff = io.BytesIO()
                    self._gifs[0].save(
                        buff,
                        format="GIF",
                        save_all=True,
                        append_images=self._gifs[1:],
                        optimize=False,
                        duration=[100] * 23 + [1000],  # pause on last frame
                        disposal=2,
                        loop=0,
                    )
                    self._last_gif = buff.getvalue()
                return self._last_gif
            except Exception as e:
                _LOGGER.warning("Error creating animated GIF: %s", e)

        _current_query_time = int(
            datetime.strftime(datetime.now(timezone(timedelta(hours=8))), "%Y%m%d%H%M")
        )

        if _current_query_time != self._last_query_time:
            self._last_query_time = _current_query_time
            _current_gif_time = math.floor(_current_query_time / 5) * 5
            if _current_gif_time != self._last_gif_time:
                return await get_image(_current_gif_time)

        return self._last_gif

    @property
    def extra_state_attributes(self) -> dict:
        """Return dict of additional properties to attach to sensors."""
        return {
            "Updated at": self._last_gif_time_pretty,
            "URL": self._last_url,
        }


async def _download_image(hass: HomeAssistant, url: str, label: str) -> bytes | None:
    """Download an image, returning None (with a warning) on any failure."""
    session = async_get_clientsession(hass)
    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                _LOGGER.warning("%s image returned HTTP %s", label, response.status)
                return None
            return await response.read()
    except (aiohttp.ClientError, TimeoutError) as err:
        _LOGGER.warning("Error fetching %s image: %s", label, err)
        return None


class NeaRainRadarCamera(CoordinatorEntity, Camera):
    """Latest NEA rain radar image for one range, drawn on NEA's base map.

    The transparent data.gov.sg radar PNG is laid on the base map from NEA's rain
    areas page (fetched once from nea.gov.sg and kept in memory), the way that page
    draws it. While the base map can't be fetched the bare overlay is served.
    """

    _attr_has_entity_name = True
    _attr_translation_key = "rain_radar"
    _attr_attribution = "Data provided by NEA via data.gov.sg; map by NEA"
    _unique_suffix = "Rain Radar"
    _entity_suffix = "rain_radar"

    def __init__(
        self,
        coordinator: RadarCoordinator,
        entry_id: str,
        prefix: str,
        radar_range: str,
    ) -> None:
        super().__init__(coordinator)
        Camera.__init__(self)
        self.content_type = "image/jpeg"
        self._range = radar_range
        self._attr_unique_id = f"{prefix} {self._unique_suffix} {radar_range}"
        self.entity_id = (
            f"camera.{prefix}_{self._entity_suffix}_{radar_range}".lower().replace(
                " ", "_"
            )
        )
        self._attr_translation_placeholders = {"range": RADAR_RANGES[radar_range]}
        self._attr_device_info = main_device_info(entry_id)
        self._basemap_spec = BASEMAPS[radar_range]
        self._basemap: bytes | None = None
        self._basemap_failed_at: float | None = None
        self._cached_key: object = None
        self._cached_image: bytes | None = None

    @property
    def _radar(self) -> RadarImage | None:
        data = self.coordinator.data
        return data.images.get(self._range) if data is not None else None

    @property
    def available(self) -> bool:
        return super().available and self._radar is not None

    @property
    def extra_state_attributes(self) -> dict:
        attrs: dict = {"range": RADAR_RANGES[self._range]}
        radar = self._radar
        if radar is not None:
            attrs["image_timestamp"] = (
                radar.timestamp.isoformat() if radar.timestamp else None
            )
            attrs["north"] = radar.north
            attrs["south"] = radar.south
            attrs["east"] = radar.east
            attrs["west"] = radar.west
            attrs["center_latitude"] = radar.center_latitude
            attrs["center_longitude"] = radar.center_longitude
        return attrs

    async def _async_get_basemap(self) -> bytes | None:
        """Return NEA's base map for this range, fetching it at most once.

        A failed fetch is retried after ``BASEMAP_RETRY_SECONDS``, not on every
        image request.
        """
        if self._basemap is not None:
            return self._basemap
        if (
            self._basemap_failed_at is not None
            and time.monotonic() - self._basemap_failed_at < BASEMAP_RETRY_SECONDS
        ):
            return None
        self._basemap = await _download_image(
            self.hass, self._basemap_spec.url, f"Rain radar {self._range} base map"
        )
        if self._basemap is None:
            self._basemap_failed_at = time.monotonic()
        return self._basemap

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Return the latest radar image on NEA's base map.

        The presigned URL changes on every poll, so the cache is keyed by the
        capture timestamp (falling back to the URL if NEA omits it), plus whether
        the base map was available.
        """
        radar = self._radar
        if radar is None:
            return self._cached_image
        basemap = await self._async_get_basemap()
        key = (radar.timestamp or radar.image_url, basemap is not None)
        if key == self._cached_key and self._cached_image is not None:
            return self._cached_image

        overlay = await _download_image(
            self.hass, radar.image_url, f"Rain radar {self._range}"
        )
        if overlay is None:
            return self._cached_image

        image, content_type = overlay, "image/png"
        if basemap is not None:
            try:
                image = await self.hass.async_add_executor_job(
                    composite_radar, basemap, overlay, self._basemap_spec
                )
                content_type = "image/jpeg"
            except OSError as err:  # PIL.UnidentifiedImageError is an OSError
                _LOGGER.warning(
                    "Could not draw rain radar %s on its base map: %s", self._range, err
                )
                # A corrupt base map would fail every time: fetch it again later.
                self._basemap = None
                self._basemap_failed_at = time.monotonic()
                key = (key[0], False)

        self.content_type = content_type
        self._cached_key = key
        self._cached_image = image
        return image


class NeaRainRadarAnimationCamera(NeaRainRadarCamera):
    """The last hour of NEA rain radar on NEA's base map, as a looping GIF.

    Frames come from the same API call as the still image (``RadarImage.frames``).
    Each frame's overlay is downloaded once and kept while it is in the window, so a
    new poll costs one small download; the GIF is rebuilt only when the set of
    frames changes. Without the base map there is no animation (the last one built,
    if any, is served) until the map can be fetched again.
    """

    _attr_translation_key = "rain_radar_animation"
    _unique_suffix = "Rain Radar Animation"
    _entity_suffix = "rain_radar_animation"

    def __init__(
        self,
        coordinator: RadarCoordinator,
        entry_id: str,
        prefix: str,
        radar_range: str,
    ) -> None:
        super().__init__(coordinator, entry_id, prefix, radar_range)
        self.content_type = "image/gif"
        self._overlays: dict[datetime, bytes] = {}

    @property
    def available(self) -> bool:
        radar = self._radar
        return super().available and radar is not None and bool(radar.frames)

    @property
    def extra_state_attributes(self) -> dict:
        attrs = super().extra_state_attributes
        radar = self._radar
        frames = radar.frames if radar is not None else []
        attrs["frame_count"] = len(frames)
        attrs["first_frame_timestamp"] = (
            frames[0].timestamp.isoformat() if frames else None
        )
        return attrs

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Return the animated GIF of the frames currently in the window."""
        radar = self._radar
        if radar is None or not radar.frames:
            return self._cached_image
        basemap = await self._async_get_basemap()
        if basemap is None:
            return self._cached_image

        wanted = {frame.timestamp for frame in radar.frames}
        self._overlays = {t: b for t, b in self._overlays.items() if t in wanted}
        for frame in radar.frames:
            if frame.timestamp not in self._overlays:
                overlay = await _download_image(
                    self.hass,
                    frame.image_url,
                    f"Rain radar {self._range} {frame.timestamp:%H:%M}",
                )
                if overlay is not None:  # a missing frame is retried next time
                    self._overlays[frame.timestamp] = overlay

        frames = [
            (frame.timestamp, self._overlays[frame.timestamp])
            for frame in radar.frames
            if frame.timestamp in self._overlays
        ]
        if not frames:
            return self._cached_image
        key = tuple(stamp for stamp, _ in frames)
        if key == self._cached_key and self._cached_image is not None:
            return self._cached_image

        try:
            image = await self.hass.async_add_executor_job(
                animate_radar,
                basemap,
                [(f"{stamp.astimezone(SGT):%H:%M}", data) for stamp, data in frames],
                self._basemap_spec,
            )
        except OSError as err:  # PIL.UnidentifiedImageError is an OSError
            _LOGGER.warning("Could not animate rain radar %s: %s", self._range, err)
            # A corrupt base map would fail every time: fetch it again later. A
            # corrupt frame is dropped so the next request downloads it again.
            self._basemap = None
            self._basemap_failed_at = time.monotonic()
            self._overlays.clear()
            return self._cached_image

        self._cached_key = key
        self._cached_image = image
        return image
