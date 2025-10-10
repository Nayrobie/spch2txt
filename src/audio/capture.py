"""
Audio capture functionality using pyaudiowpatch for Windows WASAPI support.
"""

import os
import time
import wave
import queue
import threading
import numpy as np
import pyaudiowpatch as pyaudio
from datetime import datetime
from typing import Optional, Dict, List, Tuple


class AudioCapture:
    """Handle audio recording from various input devices."""

    def __init__(self, frames_per_buffer: int = 1024):
        """
        Initialize audio capture.

        Args:
            frames_per_buffer: Buffer size for audio chunks
        """
        self.frames_per_buffer = frames_per_buffer
        self.pa = None
        self.stream = None
        
    def list_devices(self) -> List[Dict]:
        """
        List all available audio devices.

        Returns:
            List of device information dictionaries
        """
        pa = pyaudio.PyAudio()
        devices = []

        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            devices.append({
                'index': i,
                'name': info['name'],
                'hostApi': info['hostApi'],
                'maxInputChannels': info['maxInputChannels'],
                'maxOutputChannels': info['maxOutputChannels'],
                'defaultSampleRate': info['defaultSampleRate'],
                'isLoopback': 'loopback' in info['name'].lower()
            })

        pa.terminate()
        return devices
    
    def get_default_input_device(self) -> Dict:
        """Get the default input device information."""
        pa = pyaudio.PyAudio()
        device_info = pa.get_default_input_device_info()
        pa.terminate()
        return device_info

    def get_audio_level(self, device_index: int,
                        duration: float = 0.5) -> float:
        """
        Get the current audio level for a device.

        Args:
            device_index: Audio device index
            duration: Duration to sample in seconds

        Returns:
            Audio level (RMS) as float between 0 and 1
        """
        pa = pyaudio.PyAudio()

        try:
            device_info = pa.get_device_info_by_index(device_index)
            channels = min(device_info['maxInputChannels'], 2)
            rate = int(device_info['defaultSampleRate'])

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=self.frames_per_buffer,
                input_device_index=device_index
            )

            num_chunks = int(rate / self.frames_per_buffer * duration)
            frames = []

            for _ in range(num_chunks):
                try:
                    data = stream.read(self.frames_per_buffer,
                                       exception_on_overflow=False)
                    frames.append(data)
                except Exception:
                    break

            stream.close()

            if frames:
                audio_data = b"".join(frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
                level = np.sqrt(np.mean(audio_float ** 2))
                return float(level)

            return 0.0

        except Exception:
            return 0.0
        finally:
            pa.terminate()
    
    
    def record_multi_device(self, device_indices: List[int],
                            device_names: List[str],
                            channels_list: List[int],
                            rates: List[int],
                            duration: int,
                            output_dir: str = "src/saved_audio") -> List[str]:
        """
        Record audio from multiple devices simultaneously.

        Args:
            device_indices: List of device indices
            device_names: List of device names
            channels_list: List of channel counts
            rates: List of sample rates
            duration: Recording duration in seconds
            output_dir: Directory to save recordings

        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_files = []

        for i, name in enumerate(device_names):
            clean_name = "".join(
                c if c.isalnum() or c in (' ', '-', '_') else '_'
                for c in name
            )
            clean_name = clean_name.replace(' ', '_')
            if len(clean_name) > 50:
                clean_name = clean_name[:50]
            filename = f"recording_{timestamp}_dev{i+1}_{clean_name}.wav"
            output_files.append(os.path.join(output_dir, filename))

        pa = pyaudio.PyAudio()

        try:
            streams = []

            for i, (device_index, channels, rate) in enumerate(
                zip(device_indices, channels_list, rates)
            ):
                try:
                    stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=self.frames_per_buffer,
                        input_device_index=device_index
                    )
                    streams.append(stream)
                except Exception as e:
                    for s in streams:
                        s.close()
                    pa.terminate()
                    raise RuntimeError(
                        f"Failed to open stream for device {i+1}: {e}"
                    )

            num_chunks_per_device = [
                int(rate * duration / self.frames_per_buffer)
                for rate in rates
            ]

            queues = [queue.Queue() for _ in device_indices]
            stop_event = threading.Event()
            threads = []

            is_loopback = [
                'loopback' in name.lower() for name in device_names
            ]

            for i, (stream, name, is_loop, channels, num_chunks) in enumerate(
                zip(streams, device_names, is_loopback,
                    channels_list, num_chunks_per_device)
            ):
                thread = threading.Thread(
                    target=self._record_stream_thread,
                    args=(stream, name, is_loop, channels,
                          num_chunks, queues[i], stop_event)
                )
                thread.daemon = True
                thread.start()
                threads.append(thread)

            start_time = time.time()

            try:
                while any(t.is_alive() for t in threads):
                    elapsed = time.time() - start_time
                    if elapsed >= duration:
                        stop_event.set()
                        break
                    time.sleep(0.1)
            except KeyboardInterrupt:
                stop_event.set()

            for thread in threads:
                thread.join(timeout=2.0)

            stop_event.set()

            for i, (output_file, q) in enumerate(zip(output_files, queues)):
                frames = []
                while not q.empty():
                    try:
                        frames.append(q.get_nowait())
                    except Exception:
                        break

                if len(frames) == 0:
                    num_samples = int(
                        rates[i] * duration * channels_list[i]
                    )
                    silence_chunk = b'\x00' * (
                        self.frames_per_buffer * 2 * channels_list[i]
                    )
                    num_silence_chunks = int(
                        num_samples / (
                            self.frames_per_buffer * channels_list[i]
                        )
                    )
                    frames = [silence_chunk] * num_silence_chunks

                with wave.open(output_file, 'wb') as wf:
                    wf.setnchannels(channels_list[i])
                    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(rates[i])
                    wf.writeframes(b"".join(frames))

            for stream in streams:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

            return output_files

        except Exception as e:
            raise RuntimeError(f"Error recording: {e}")
        finally:
            pa.terminate()

    def _record_stream_thread(self, stream, device_name: str,
                              is_loopback: bool, channels: int,
                              num_chunks: int, frames_queue: queue.Queue,
                              stop_event: threading.Event):
        """
        Thread function to record from a single stream.

        Args:
            stream: PyAudio stream object
            device_name: Name of the device
            is_loopback: Whether device is loopback
            channels: Number of channels
            num_chunks: Number of chunks to record
            frames_queue: Queue to store frames
            stop_event: Event to signal stop
        """
        chunk_count = 0
        silence = b'\x00' * (self.frames_per_buffer * 2 * channels)
        consecutive_errors = 0
        max_consecutive_errors = 100

        while chunk_count < num_chunks and not stop_event.is_set():
            try:
                data = stream.read(self.frames_per_buffer,
                                   exception_on_overflow=False)
                frames_queue.put(data)
                chunk_count += 1
                consecutive_errors = 0

            except Exception:
                consecutive_errors += 1
                frames_queue.put(silence)
                chunk_count += 1

                if consecutive_errors >= max_consecutive_errors:
                    break

                time.sleep(0.01)
    
        
    def cleanup(self):
        """Clean up audio resources."""
        if self.stream:
            self.stream.close()
        if self.pa:
            self.pa.terminate()
