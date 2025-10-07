import wave
import pyaudiowpatch as pyaudio

"""
Quick test: 10 seconds record from the default microphone

How to run:
    poetry run python tests/test_record.py
"""

DUR = 10
RATE = 16000  # Lower rate for better compatibility with Whisper
CH = 1  # Mono for microphone
FRAMES = 1024

print("Initializing audio...")
pa = pyaudio.PyAudio()

# Get default input device (microphone)
default_input = pa.get_default_input_device_info()
print(f"Using device: {default_input['name']}")
print(f"Recording for {DUR} seconds...")

wf = wave.open("out.wav", "wb")
wf.setnchannels(CH)
wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
wf.setframerate(RATE)

stream = pa.open(format=pyaudio.paInt16, 
                 channels=CH, 
                 rate=RATE,
                 input=True, 
                 frames_per_buffer=FRAMES,
                 input_device_index=default_input["index"])

# Record in chunks with progress indicator
num_chunks = int(RATE / FRAMES * DUR)
frames = []
for i in range(num_chunks):
    data = stream.read(FRAMES)
    frames.append(data)
    if i % 10 == 0:  # Print progress every ~0.6 seconds
        progress = (i / num_chunks) * 100
        print(f"Recording... {progress:.0f}%")

print("Recording complete. Saving file...")
wf.writeframes(b"".join(frames))
stream.close()
pa.terminate()
wf.close()
print("Saved to out.wav")
