import wave
import pyaudiowpatch as pyaudio

"""
Quick test: 10 seconds record from the default output (WASAPI loopback)

How to run:
    poetry run python src/probe_record.py
"""

DUR = 10
RATE = 48000
CH = 2
FRAMES = 4096

pa = pyaudio.PyAudio()
loopback = pa.get_default_wasapi_loopback()

wf = wave.open("out.wav", "wb")
wf.setnchannels(CH)
wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
wf.setframerate(RATE)

stream = pa.open(format=pyaudio.paInt16, channels=CH, rate=RATE,
                 input=True, frames_per_buffer=FRAMES,
                 input_device_index=loopback["index"])
frames = [stream.read(FRAMES) for _ in range(int(RATE / FRAMES * DUR))]
wf.writeframes(b"".join(frames))
stream.close(); pa.terminate(); wf.close()
print("Wrote out.wav")