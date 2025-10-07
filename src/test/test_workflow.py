import wave
import pyaudiowpatch as pyaudio
import whisper
import numpy as np
import os
from datetime import datetime

"""
Complete test workflow: Record 10 seconds of audio and transcribe it.

How to run:
    poetry run python src/audio/test_workflow.py
"""

# Configuration
DURATION = 10  # seconds
RATE = 16000  # 16kHz for Whisper compatibility
CHANNELS = 1  # Mono
FRAMES_PER_BUFFER = 1024
OUTPUT_FILE = "test_recording.wav"

def record_audio():
    """Record audio from the default microphone."""
    print("="*60)
    print("STEP 1: RECORDING AUDIO")
    print("="*60)
    
    pa = pyaudio.PyAudio()
    
    # Get default input device
    default_input = pa.get_default_input_device_info()
    print(f"Using device: {default_input['name']}")
    print(f"Recording for {DURATION} seconds...")
    print("🎤 Please speak now!")
    
    # Open wave file
    wf = wave.open(OUTPUT_FILE, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    
    # Open audio stream
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
        input_device_index=default_input["index"]
    )
    
    # Record in chunks
    num_chunks = int(RATE / FRAMES_PER_BUFFER * DURATION)
    frames = []
    
    for i in range(num_chunks):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)
        
        # Progress indicator
        if i % 15 == 0:
            progress = (i / num_chunks) * 100
            bars = int(progress / 5)
            print(f"[{'█' * bars}{' ' * (20-bars)}] {progress:.0f}%", end='\r')
    
    print(f"[{'█' * 20}] 100%")
    
    # Save and cleanup
    wf.writeframes(b"".join(frames))
    stream.close()
    pa.terminate()
    wf.close()
    
    print(f"✓ Recording saved to {OUTPUT_FILE}")
    return OUTPUT_FILE

def load_wav_file(filepath):
    """Load a WAV file and convert to format expected by Whisper."""
    with wave.open(filepath, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        
        audio_data = wf.readframes(n_frames)
        
        if sampwidth == 2:  # 16-bit audio
            audio = np.frombuffer(audio_data, dtype=np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        
        # Convert to float32 and normalize
        audio = audio.astype(np.float32) / 32768.0
        
        # Convert stereo to mono if needed
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        
        # Resample if needed
        if framerate != 16000:
            duration = len(audio) / framerate
            target_length = int(duration * 16000)
            audio = np.interp(
                np.linspace(0, len(audio), target_length),
                np.arange(len(audio)),
                audio
            )
        
        return audio

def transcribe_audio(filepath):
    """Transcribe audio using Whisper."""
    print("\n" + "="*60)
    print("STEP 2: TRANSCRIBING AUDIO")
    print("="*60)
    
    print("Loading audio file...")
    audio = load_wav_file(filepath)
    duration = len(audio) / 16000
    print(f"✓ Audio loaded: {duration:.1f} seconds")
    
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("✓ Model loaded")
    
    print("Transcribing...")
    result = model.transcribe(audio, language="en", verbose=False)
    
    return result

def display_results(result):
    """Display transcription results."""
    print("\n" + "="*60)
    print("TRANSCRIPTION RESULTS")
    print("="*60)
    
    # Full transcription
    print("\n📝 Full Transcription:")
    print("-" * 60)
    print(result["text"])
    print("-" * 60)
    
    # Segments (with timestamps)
    if "segments" in result and len(result["segments"]) > 0:
        print("\n⏱️  Timestamped Segments:")
        print("-" * 60)
        for i, segment in enumerate(result["segments"], 1):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            print(f"[{start:05.1f}s - {end:05.1f}s] {text}")
        print("-" * 60)
    
    # Language detection
    if "language" in result:
        print(f"\n🌍 Detected Language: {result['language']}")
    
    print("\n" + "="*60)

def main():
    """Main workflow."""
    print("\n" + "="*60)
    print("AUDIO RECORDING & TRANSCRIPTION TEST")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        # Step 1: Record
        audio_file = record_audio()
        
        # Step 2: Transcribe
        result = transcribe_audio(audio_file)
        
        # Step 3: Display
        display_results(result)
        
        print("\n✅ Test completed successfully!")
        print(f"📁 Audio file saved as: {audio_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
