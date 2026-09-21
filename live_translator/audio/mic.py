from __future__ import annotations

from .base import AudioCallback, AudioSource, AudioSourceError, to_mono


def list_input_devices() -> list[str]:
    """Names of capture-capable devices (loopback/monitor pseudo-devices excluded), default first."""
    try:
        import sounddevice as sd
    except (ImportError, OSError):                  # OSError: PortAudio shared library missing (Linux: libportaudio2)
        return []
    names: list[str] = []
    try:
        default = sd.default.device[0]
        devices = sd.query_devices()
    except Exception:
        return []
    for i, d in enumerate(devices):
        if d["max_input_channels"] <= 0:
            continue
        # PortAudio lists every device once per host API (MME/DirectSound/WASAPI/...); keep one entry per name.
        if d["name"] in names:
            continue
        if i == default:
            names.insert(0, d["name"])
        else:
            names.append(d["name"])
    return names


class MicSource(AudioSource):
    def __init__(self, device: str = ""):
        self.device = device or None
        self.name = f"麦克风{f'（{device}）' if device else ''}"
        self._stream = None

    def start(self, on_audio: AudioCallback) -> None:
        try:
            import sounddevice as sd
        except ImportError as e:
            raise AudioSourceError("缺少依赖 sounddevice，请执行: pip install sounddevice") from e
        except OSError as e:
            raise AudioSourceError("找不到 PortAudio 库，无法使用麦克风。Linux 请执行: sudo apt install libportaudio2") from e
        try:
            info = sd.query_devices(self.device, "input")
            rate = int(info["default_samplerate"])

            def _cb(indata, frames, time_info, status):
                on_audio(to_mono(indata), rate)

            self._stream = sd.InputStream(device=self.device, channels=1, samplerate=rate,
                                          dtype="float32", blocksize=rate // 20, callback=_cb)
            self._stream.start()
        except Exception as e:
            raise AudioSourceError(f"无法打开麦克风：{e}") from e

    def stop(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            s.stop()
            s.close()
