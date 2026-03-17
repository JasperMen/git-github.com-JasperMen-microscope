"""Microscope camera service built on top of the nncam Python SDK.

This module wraps the vendor SDK so that higher level applications (such as
an Electron + Vue frontend) can control the microscope camera through a
persistent Python process. The service keeps the camera connection alive,
pulls frames via callbacks, and exposes them to the rest of the application.
"""

from __future__ import annotations

import base64
import ctypes
import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import nncam


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    """Container holding one camera frame in RGB order."""

    width: int
    height: int
    timestamp_us: int
    sequence: int
    data: bytes
    expotime_us: Optional[int] = None
    expogain: Optional[int] = None
    blacklevel: Optional[int] = None

    def as_base64(self) -> str:
        """Return the frame encoded as base64 string (useful for JSON APIs)."""

        return base64.b64encode(self.data).decode("ascii")


class MicroscopeCameraService:
    """Manage a persistent connection to the microscope camera."""

    _BYTES_PER_PIXEL = 3  # RGB24

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._camera: Optional[nncam.Nncam] = None
        self._device: Optional[nncam.NncamDeviceV2] = None
        self._buffer: Optional[ctypes.Array] = None
        self._buffer_size: int = 0
        self._row_pitch: int = -1  # zero padding disabled
        self._latest_frame: Optional[Frame] = None
        self._frame_event = threading.Event()
        self._running = False
        self._last_error: Optional[str] = None
        self._gige_ready = False
        self._available_resolutions: List[Dict[str, int]] = []
        self._current_resolution_index: Optional[int] = None
        self._frame_rate: Optional[float] = None
        self._selected_device_id: Optional[str] = None

    # --- Public API -------------------------------------------------

    def start(
        self,
        camera_identifier: Optional[str] = None,
        resolution_index: Optional[int] = None,
    ) -> dict:
        """Open the camera and start pulling frames in callback mode."""

        with self._lock:
            if self._running:
                logger.debug("Camera service already running")
                return self.status()

            devices = nncam.Nncam.EnumV2()
            if not devices:
                raise RuntimeError("No microscope camera detected")

            device = self._select_device(devices, camera_identifier)

            if not self._gige_ready:
                try:
                    nncam.Nncam.GigeEnable(None, None)
                except Exception as exc:  # pragma: no cover - optional feature
                    logger.debug("Failed to enable GigE support: %s", exc)
                else:
                    self._gige_ready = True

            camera = nncam.Nncam.Open(device.id)
            if not camera:
                raise RuntimeError(f"Unable to open camera '{device.displayname}'")

            try:
                if resolution_index is not None:
                    if resolution_index < 0 or resolution_index >= device.model.preview:
                        raise ValueError(
                            f"Invalid resolution index {resolution_index}; "
                            f"supported range: 0-{device.model.preview - 1}"
                        )
                    camera.put_eSize(resolution_index)

                width, height = camera.get_Size()
                self._allocate_buffer(width, height)

                camera.put_Option(nncam.NNCAM_OPTION_BYTEORDER, 0)  # RGB order
                camera.put_AutoExpoEnable(1)

                self._camera = camera
                self._device = device
                self._selected_device_id = device.id
                self._latest_frame = None
                self._frame_event.clear()
                self._running = True
                self._last_error = None
                self._available_resolutions = [
                    {
                        "index": i,
                        "width": device.model.res[i].width,
                        "height": device.model.res[i].height,
                    }
                    for i in range(device.model.preview)
                ]
                self._current_resolution_index = camera.get_eSize()
                self._frame_rate = None

                camera.StartPullModeWithCallback(self._native_callback, self)
                logger.info(
                    "Camera '%s' started at %dx%d",
                    device.displayname,
                    width,
                    height,
                )
            except Exception:
                camera.Close()
                self._cleanup_locked()
                raise

        return self.status()

    def stop(self) -> None:
        """Stop the camera and release resources."""

        with self._lock:
            self._stop_locked()
            self._cleanup_locked()
            logger.info("Camera service stopped")

    def status(self) -> dict:
        """Return current service state suitable for JSON serialization."""

        with self._lock:
            frame = self._latest_frame
            if self._device and self._current_resolution_index is not None:
                res_model = self._device.model.res[self._current_resolution_index]
                resolution = (res_model.width, res_model.height)
            elif frame:
                resolution = (frame.width, frame.height)
            else:
                resolution = None
            return {
                "running": self._running,
                "camera": self._device.displayname if self._device else None,
                "device_id": self._selected_device_id,
                "resolution": resolution,
                "resolution_index": self._current_resolution_index,
                "available_resolutions": list(self._available_resolutions),
                "sequence": frame.sequence if frame else None,
                "frame_rate": self._frame_rate,
                "last_error": self._last_error,
            }

    def enumerate_devices(self) -> List[Dict[str, str]]:
        devices = nncam.Nncam.EnumV2()
        return [
            {"id": dev.id, "display": dev.displayname}
            for dev in devices
        ]

    def is_running(self) -> bool:
        return self._running

    def get_latest_frame(self) -> Optional[Frame]:
        with self._lock:
            return self._latest_frame

    def wait_for_frame(self, timeout: Optional[float] = None) -> Optional[Frame]:
        """Block until a new frame is available or timeout expires."""

        if not self._frame_event.wait(timeout):
            return None
        with self._lock:
            frame = self._latest_frame
            self._frame_event.clear()
            return frame

    def fetch_recent_frame(self, timeout: Optional[float] = None) -> Frame:
        """Return the most recent frame, waiting briefly if necessary."""

        frame = self.wait_for_frame(timeout)
        if frame is not None:
            return frame

        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame

        raise RuntimeError("No frame available from camera")

    def get_frame_rate(self) -> Optional[float]:
        with self._lock:
            return self._frame_rate

    def set_resolution(self, index: int) -> Dict[str, int]:
        with self._lock:
            if not self._camera or not self._device:
                raise RuntimeError("Camera is not running")

            if index < 0 or index >= len(self._available_resolutions):
                raise ValueError("resolution index out of range")

            if self._current_resolution_index == index:
                res = self._device.model.res[index]
                return {"index": index, "width": res.width, "height": res.height}

            try:
                self._camera.Stop()
            except Exception:
                pass

            try:
                self._camera.put_eSize(index)
                width, height = self._camera.get_Size()
                self._allocate_buffer(width, height)
                self._camera.put_Option(nncam.NNCAM_OPTION_BYTEORDER, 0)
                self._camera.put_AutoExpoEnable(1)
                self._latest_frame = None
                self._frame_event.clear()
                self._frame_rate = None
                self._camera.StartPullModeWithCallback(self._native_callback, self)
                self._current_resolution_index = index
                self._available_resolutions = [
                    {
                        "index": i,
                        "width": self._device.model.res[i].width,
                        "height": self._device.model.res[i].height,
                    }
                    for i in range(self._device.model.preview)
                ]
                return {"index": index, "width": width, "height": height}
            except Exception as exc:
                self._last_error = "Failed to change resolution"
                raise RuntimeError("Failed to change resolution") from exc

    def get_color_controls(self) -> Dict[str, int]:
        with self._lock:
            if not self._camera:
                raise RuntimeError("Camera is not running")

            return {
                "brightness": self._camera.get_Brightness(),
                "hue": self._camera.get_Hue(),
                "saturation": self._camera.get_Saturation(),
            }

    def set_color_controls(
        self,
        brightness: Optional[int] = None,
        hue: Optional[int] = None,
        saturation: Optional[int] = None,
    ) -> Dict[str, int]:
        with self._lock:
            if not self._camera:
                raise RuntimeError("Camera is not running")

            if brightness is not None:
                if not (nncam.NNCAM_BRIGHTNESS_MIN <= brightness <= nncam.NNCAM_BRIGHTNESS_MAX):
                    raise ValueError("brightness out of range")
                self._camera.put_Brightness(brightness)

            if hue is not None:
                if not (nncam.NNCAM_HUE_MIN <= hue <= nncam.NNCAM_HUE_MAX):
                    raise ValueError("hue out of range")
                self._camera.put_Hue(hue)

            if saturation is not None:
                if not (nncam.NNCAM_SATURATION_MIN <= saturation <= nncam.NNCAM_SATURATION_MAX):
                    raise ValueError("saturation out of range")
                self._camera.put_Saturation(saturation)

            return {
                "brightness": self._camera.get_Brightness(),
                "hue": self._camera.get_Hue(),
                "saturation": self._camera.get_Saturation(),
            }

    def reset_color_controls(self) -> Dict[str, int]:
        return self.set_color_controls(
            brightness=nncam.NNCAM_BRIGHTNESS_DEF,
            hue=nncam.NNCAM_HUE_DEF,
            saturation=nncam.NNCAM_SATURATION_DEF,
        )

    def auto_color_balance(self) -> Dict[str, int]:
        with self._lock:
            if not self._camera:
                raise RuntimeError("Camera is not running")

            try:
                # 重置颜色参数到默认值
                self._camera.put_Brightness(nncam.NNCAM_BRIGHTNESS_DEF)
                self._camera.put_Hue(nncam.NNCAM_HUE_DEF)
                self._camera.put_Saturation(nncam.NNCAM_SATURATION_DEF)
                
                # 检查是否为单色相机（单色相机不支持白平衡）
                if self._device and (self._device.model.flag & nncam.NNCAM_FLAG_MONO) == 0:
                    # 彩色相机：执行自动白平衡
                    self._camera.AwbOnce()
                    # 等待白平衡完成（通过等待几帧来确保白平衡计算完成）
                    time.sleep(0.5)  # 等待白平衡计算完成
                
                # 执行自动电平范围调整
                try:
                    self._camera.LevelRangeAuto()
                except nncam.HRESULTException:
                    # LevelRangeAuto可能不支持，忽略错误
                    pass
            except nncam.HRESULTException as exc:
                raise RuntimeError(f"Auto color adjustment failed: 0x{exc.hr & 0xffffffff:x}") from exc

            return {
                "brightness": self._camera.get_Brightness(),
                "hue": self._camera.get_Hue(),
                "saturation": self._camera.get_Saturation(),
            }

    def get_exposure_controls(self) -> Dict[str, int]:
        """获取曝光和增益控制参数"""
        with self._lock:
            if not self._camera:
                raise RuntimeError("Camera is not running")

            auto_expo = self._camera.get_AutoExpoEnable()
            expo_time = self._camera.get_ExpoTime()
            expo_gain = self._camera.get_ExpoAGain()
            auto_expo_target = self._camera.get_AutoExpoTarget()
            
            # 获取增益范围用于UI显示
            gain_min, gain_max = 100, 1000
            try:
                gain_range = self._camera.get_ExpoAGainRange()
                gain_min, gain_max = gain_range[0], gain_range[1]
            except Exception:
                pass

            return {
                "auto_exposure": int(auto_expo),
                "exposure_time": int(expo_time),
                "gain": int(expo_gain),
                "exposure_target": int(auto_expo_target),
                "gain_min": int(gain_min),
                "gain_max": int(gain_max),
            }

    def set_exposure_controls(
        self,
        auto_exposure: Optional[int] = None,
        exposure_time: Optional[int] = None,
        gain: Optional[int] = None,
        exposure_target: Optional[int] = None,
    ) -> Dict[str, int]:
        """设置曝光和增益控制参数"""
        with self._lock:
            if not self._camera:
                raise RuntimeError("Camera is not running")

            if auto_exposure is not None:
                self._camera.put_AutoExpoEnable(auto_exposure)

            if exposure_target is not None:
                if not (nncam.NNCAM_AETARGET_MIN <= exposure_target <= nncam.NNCAM_AETARGET_MAX):
                    raise ValueError("exposure_target out of range")
                self._camera.put_AutoExpoTarget(exposure_target)

            if exposure_time is not None:
                self._camera.put_ExpoTime(exposure_time)

            if gain is not None:
                try:
                    gain_range = self._camera.get_ExpoAGainRange()
                    if not (gain_range[0] <= gain <= gain_range[1]):
                        raise ValueError("gain out of range")
                except Exception:
                    # 如果获取范围失败，使用默认范围
                    if not (100 <= gain <= 1000):
                        raise ValueError("gain out of range")
                self._camera.put_ExpoAGain(gain)

            return self.get_exposure_controls()

    def auto_exposure_once(self) -> Dict[str, int]:
        """执行一次自动曝光"""
        with self._lock:
            if not self._camera:
                raise RuntimeError("Camera is not running")

            try:
                # 先启用自动曝光
                self._camera.put_AutoExpoEnable(1)
                # 等待自动曝光完成
                time.sleep(0.5)
            except nncam.HRESULTException as exc:
                raise RuntimeError(f"Auto exposure failed: 0x{exc.hr & 0xffffffff:x}") from exc

            return self.get_exposure_controls()

    def get_sharpening_controls(self) -> Dict[str, int]:
        """获取锐化控制参数"""
        with self._lock:
            if not self._camera:
                raise RuntimeError("Camera is not running")

            sharpening = self._camera.get_Option(nncam.NNCAM_OPTION_SHARPENING)
            strength = sharpening & 0xFFFF
            radius = (sharpening >> 16) & 0xFF
            threshold = (sharpening >> 24) & 0xFF

            return {
                "strength": int(strength),
                "radius": int(radius),
                "threshold": int(threshold),
            }

    def set_sharpening_controls(
        self,
        strength: Optional[int] = None,
        radius: Optional[int] = None,
        threshold: Optional[int] = None,
    ) -> Dict[str, int]:
        """设置锐化控制参数"""
        with self._lock:
            if not self._camera:
                raise RuntimeError("Camera is not running")

            current = self.get_sharpening_controls()
            if strength is None:
                strength = current["strength"]
            if radius is None:
                radius = current["radius"]
            if threshold is None:
                threshold = current["threshold"]

            if not (nncam.NNCAM_SHARPENING_STRENGTH_MIN <= strength <= nncam.NNCAM_SHARPENING_STRENGTH_MAX):
                raise ValueError("strength out of range")
            if not (nncam.NNCAM_SHARPENING_RADIUS_MIN <= radius <= nncam.NNCAM_SHARPENING_RADIUS_MAX):
                raise ValueError("radius out of range")
            if not (nncam.NNCAM_SHARPENING_THRESHOLD_MIN <= threshold <= nncam.NNCAM_SHARPENING_THRESHOLD_MAX):
                raise ValueError("threshold out of range")

            sharpening_value = (threshold << 24) | (radius << 16) | strength
            self._camera.put_Option(nncam.NNCAM_OPTION_SHARPENING, sharpening_value)

            return self.get_sharpening_controls()

    def get_misc_controls(self) -> Dict[str, int]:
        """获取杂项控制参数"""
        with self._lock:
            if not self._camera:
                raise RuntimeError("Camera is not running")

            negative = self._camera.get_Negative()
            demosaic = self._camera.get_Option(nncam.NNCAM_OPTION_DEMOSAIC)
            curve = self._camera.get_Option(nncam.NNCAM_OPTION_CURVE)
            
            # 检查是否支持低噪声模式
            low_noise = 0
            try:
                if self._device and (self._device.model.flag & nncam.NNCAM_FLAG_LOW_NOISE) != 0:
                    # 低噪声模式通过OPTION设置，需要查看具体实现
                    # 这里假设通过某个OPTION控制
                    pass
            except:
                pass

            return {
                "negative": int(negative),
                "demosaic": int(demosaic),
                "tone_mapping": int(curve),
                "low_noise": int(low_noise),
            }

    def set_misc_controls(
        self,
        negative: Optional[int] = None,
        demosaic: Optional[int] = None,
        tone_mapping: Optional[int] = None,
        low_noise: Optional[int] = None,
    ) -> Dict[str, int]:
        """设置杂项控制参数"""
        with self._lock:
            if not self._camera:
                raise RuntimeError("Camera is not running")

            if negative is not None:
                self._camera.put_Negative(negative)

            if demosaic is not None:
                if not (0 <= demosaic <= 4):
                    raise ValueError("demosaic out of range (0-4)")
                self._camera.put_Option(nncam.NNCAM_OPTION_DEMOSAIC, demosaic)

            if tone_mapping is not None:
                if not (0 <= tone_mapping <= 2):
                    raise ValueError("tone_mapping out of range (0-2)")
                self._camera.put_Option(nncam.NNCAM_OPTION_CURVE, tone_mapping)

            # 低噪声模式需要根据具体相机支持情况实现
            # 这里暂时跳过

            return self.get_misc_controls()

    def get_speed_controls(self) -> Dict[str, int]:
        """获取速度控制参数"""
        with self._lock:
            if not self._camera:
                raise RuntimeError("Camera is not running")

            current_speed = self._camera.get_Speed()
            max_speed = self._camera.MaxSpeed()

            return {
                "speed": int(current_speed),
                "max_speed": int(max_speed),
            }

    def set_speed(self, speed: Optional[int] = None) -> Dict[str, int]:
        """设置相机速度（帧速率级别）
        
        Args:
            speed: 速度级别，范围 [0, max_speed]。值越大，帧速率越高。
                  如果为 None，则不修改当前速度。
        
        Returns:
            包含当前速度设置的字典
        """
        with self._lock:
            if not self._camera:
                raise RuntimeError("Camera is not running")

            if speed is not None:
                max_speed = self._camera.MaxSpeed()
                if not (0 <= speed <= max_speed):
                    raise ValueError(f"speed must be in range [0, {max_speed}], got {speed}")
                self._camera.put_Speed(speed)

            return self.get_speed_controls()

    # --- Internal helpers ------------------------------------------

    def _stop_locked(self) -> None:
        if self._camera is not None:
            try:
                self._camera.Stop()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            try:
                self._camera.Close()
            finally:
                self._camera = None
        self._running = False

    def _cleanup_locked(self) -> None:
        self._device = None
        self._buffer = None
        self._buffer_size = 0
        self._latest_frame = None
        self._frame_event.clear()
        self._available_resolutions = []
        self._current_resolution_index = None
        self._frame_rate = None
        self._selected_device_id = None

    def _allocate_buffer(self, width: int, height: int) -> None:
        self._buffer_size = width * height * self._BYTES_PER_PIXEL
        self._buffer = ctypes.create_string_buffer(self._buffer_size)

    @staticmethod
    def _select_device(
        devices: Tuple[nncam.NncamDeviceV2, ...], camera_identifier: Optional[str]
    ) -> nncam.NncamDeviceV2:
        if camera_identifier is None:
            return devices[0]

        for device in devices:
            if device.id == camera_identifier or device.displayname == camera_identifier:
                return device
        raise RuntimeError(f"Camera '{camera_identifier}' not found")

    # --- Callbacks --------------------------------------------------

    @staticmethod
    def _native_callback(event: int, ctx: "MicroscopeCameraService") -> None:
        ctx._handle_camera_event(event)

    def _handle_camera_event(self, event: int) -> None:
        if event == nncam.NNCAM_EVENT_IMAGE:
            self._handle_image_event()
        elif event == nncam.NNCAM_EVENT_ERROR:
            self._last_error = "Camera reported generic error"
            logger.error("Received generic error event from camera")
        elif event == nncam.NNCAM_EVENT_DISCONNECTED:
            self._last_error = "Camera disconnected"
            logger.warning("Camera disconnected; stopping service")
            self.stop()
        elif event == nncam.NNCAM_EVENT_NOFRAMETIMEOUT:
            logger.warning("No frame timeout reported by camera")

    def _handle_image_event(self) -> None:
        with self._lock:
            if not self._camera or not self._buffer:
                return

            info = nncam.NncamFrameInfoV3()
            try:
                self._camera.PullImageV3(
                    self._buffer,
                    0,
                    24,
                    self._row_pitch,
                    info,
                )
            except nncam.HRESULTException as exc:
                self._last_error = f"PullImageV3 failed: 0x{exc.hr & 0xffffffff:x}"
                logger.exception("Failed to pull frame from camera")
                return

            frame_len = info.width * info.height * self._BYTES_PER_PIXEL
            if frame_len > self._buffer_size:
                logger.error(
                    "Calculated frame length %d exceeds buffer size %d",
                    frame_len,
                    self._buffer_size,
                )
                return

            frame_bytes = self._buffer.raw[:frame_len]
            self._latest_frame = Frame(
                width=info.width,
                height=info.height,
                timestamp_us=info.timestamp,
                sequence=info.seq,
                data=frame_bytes,
                expotime_us=info.expotime if info.expotime else None,
                expogain=info.expogain if info.expogain else None,
                blacklevel=info.blacklevel if info.blacklevel else None,
            )
            try:
                frame, ntime, _ = self._camera.get_FrameRate()
                if ntime:
                    self._frame_rate = frame * 1000.0 / ntime
            except Exception:
                self._frame_rate = None
            try:
                self._current_resolution_index = self._camera.get_eSize()
            except Exception:
                pass
            self._frame_event.set()


def ensure_logging_configured() -> None:
    """Configure a default logging handler if none is present."""

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        )


__all__ = ["MicroscopeCameraService", "Frame", "ensure_logging_configured"]

