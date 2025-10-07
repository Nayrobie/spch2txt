import wave
import pyaudiowpatch as pyaudio
import whisper
import numpy as np
import os
from datetime import datetime

"""
Complete test workflow: Interactive device selection, recording, and transcription.

How to run:
    poetry run python tests/test_full_workflow.py
"""

# Configuration
DURATION = 15  # seconds
RATE = 48000  # High quality for loopback devices
CHANNELS = 2  # Stereo for loopback
FRAMES_PER_BUFFER = 1024
OUTPUT_DIR = "src/saved_audio"

def list_all_devices():
    """List all audio devices with categorization and guidance."""
    print("="*80)
    print("AVAILABLE AUDIO DEVICES")
    print("="*80)
    
    pa = pyaudio.PyAudio()
    
    input_devices = []
    output_devices = []
    loopback_devices = []
    
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        
        device_info = {
            'index': i,
            'name': info['name'],
            'hostApi': info['hostApi'],
            'maxInputChannels': info['maxInputChannels'],
            'maxOutputChannels': info['maxOutputChannels'],
            'defaultSampleRate': info['defaultSampleRate']
        }
        
        # Categorize devices
        if 'loopback' in info['name'].lower():
            loopback_devices.append(device_info)
        elif info['maxInputChannels'] > 0:
            input_devices.append(device_info)
        elif info['maxOutputChannels'] > 0:
            output_devices.append(device_info)
    
    # Display input devices (microphones)
    print("\nINPUT DEVICES")
    print("-" * 80)
    if input_devices:
        for dev in input_devices:
            print(f"  [{dev['index']:2d}] {dev['name']}")
            print(f"       Channels: {dev['maxInputChannels']}, Rate: {dev['defaultSampleRate']:.0f}Hz")
    else:
        print("  No input devices found")
    
    # Display loopback devices (system audio)
    print("\nLOOPBACK DEVICES (System Audio - captures speaker & Teams audio):")
    print("-" * 80)
    if loopback_devices:
        for dev in loopback_devices:
            print(f"  [{dev['index']:2d}] {dev['name']}")
            print(f"       Channels: {dev['maxInputChannels']}, Rate: {dev['defaultSampleRate']:.0f}Hz")
    else:
        print("  No loopback devices found")
    
    # Display output devices (for reference)
    print("\nOUTPUT DEVICES:")
    print("-" * 80)
    for dev in output_devices[:3]:  # Show only first 3 to avoid clutter
        print(f"  [{dev['index']:2d}] {dev['name']}")
    if len(output_devices) > 3:
        print(f"  ... and {len(output_devices) - 3} more")
    
    pa.terminate()
    
    # Guidance
    print("\n" + "="*80)
    print("DEVICE SELECTION GUIDE:")
    print("Select a microphone (input device) to only record your voice.")
    print("Select a loopback device to record the whole audio system (e.g., speakers, Teams calls).")
    print("="*80)
    
    return input_devices, loopback_devices, output_devices

def record_audio(device_index, device_name, channels=2, rate=48000):
    """Record audio from a specific device."""
    print("\n" + "="*80)
    print("RECORDING AUDIO")
    print("="*80)
    print(f"Device: {device_name}")
    print(f"Duration: {DURATION} seconds")
    print(f"Settings: {rate}Hz, {channels} channel(s)")
    print("-" * 80)
    
    # Create output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f"recording_{timestamp}.wav")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pa = pyaudio.PyAudio()
    
    # Open wave file
    wf = wave.open(output_file, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    
    try:
        # Open audio stream
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
            input_device_index=device_index
        )
        
        print("Recording... (speak or play audio now)")
        
        # Record in chunks
        num_chunks = int(rate / FRAMES_PER_BUFFER * DURATION)
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
        
        print(f"✓ Recording saved to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Error recording: {e}")
        return None
    finally:
        pa.terminate()
        wf.close()

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
        
        # Resample if needed (Whisper expects 16kHz)
        if framerate != 16000:
            duration = len(audio) / framerate
            target_length = int(duration * 16000)
            # Ensure consistent dtype throughout interpolation
            audio = np.interp(
                np.linspace(0, len(audio), target_length, dtype=np.float32),
                np.arange(len(audio), dtype=np.float32),
                audio.astype(np.float32)
            )
        
        # Final ensure float32 type
        audio = audio.astype(np.float32)
        
        return audio

def transcribe_audio(filepath):
    """Transcribe audio using Whisper."""
    print("\n" + "="*80)
    print("TRANSCRIBING AUDIO")
    print("="*80)
    
    print("Loading audio file...")
    audio = load_wav_file(filepath)
    duration = len(audio) / 16000
    print(f"✓ Audio loaded: {duration:.1f} seconds")
    
    # Check audio level
    audio_level = np.abs(audio).mean()
    print(f"Audio level: {audio_level:.6f}")
    
    if audio_level < 0.001:
        print("❌  Warning: Audio is very quiet or silent")
    
    print("Loading Whisper model (base)...")
    model = whisper.load_model("base")
    print("✓ Model loaded")
    
    print("Transcribing...")
    # Ensure audio data is float32
    audio = audio.astype(np.float32)
    result = model.transcribe(audio, language="en", verbose=False)
    
    return result

def display_results(result):
    """Display transcription results."""
    print("\n" + "="*80)
    print("TRANSCRIPTION RESULTS")
    print("="*80)
    
    # Full transcription
    print("\n=Full Transcription:")
    print("-" * 80)
    if result["text"].strip():
        print(result["text"])
    else:
        print("(No speech detected)")
    print("-" * 80)

    # Language detection
    if "language" in result:
        print(f"\nDetected Language: {result['language']}")
    
    print("\n" + "="*80)

def interactive_menu():
    """Interactive menu for device selection and recording."""
    print("\n" + "="*80)
    print("FULL WORKFLOW TEST - RECORD & TRANSCRIBE")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # List all devices with guidance
    input_devs, loopback_devs, output_devs = list_all_devices()
    
    # Get user choice
    print("\n" + "="*80)
    print("SELECT DEVICE:")
    print("="*80)
    
    device_index = input("Enter device index number: ").strip()
    
    try:
        device_index = int(device_index)
        
        # Get device info
        pa = pyaudio.PyAudio()
        device_info = pa.get_device_info_by_index(device_index)
        pa.terminate()
        
        # Determine if it's a loopback device
        is_loopback = 'loopback' in device_info['name'].lower()
        
        # Set appropriate settings
        if is_loopback:
            channels = 2
            rate = int(device_info['defaultSampleRate'])
        else:
            channels = min(device_info['maxInputChannels'], 2)
            rate = int(device_info['defaultSampleRate'])
        
        print(f"\n✓ Selected: {device_info['name']}")
        print(f"  Type: {'LOOPBACK (system audio)' if is_loopback else 'INPUT (microphone)'}")
        print(f"  Settings: {rate}Hz, {channels} channel(s)")
        
        # Confirm
        confirm = input("\nProceed with recording? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Recording cancelled.")
            return
        
        # Record
        audio_file = record_audio(device_index, device_info['name'], channels, rate)
        
        if not audio_file:
            print("Recording failed.")
            return
        
        # Ask about transcription
        transcribe = input("\nTranscribe this recording? (y/n): ").strip().lower()
        if transcribe == 'y':
            result = transcribe_audio(audio_file)
            display_results(result)
        
        print("\n" + "="*80)
        print("✓ TEST COMPLETE")
        print("="*80)
        print(f"Audio saved: {audio_file}")
        
    except ValueError:
        print("\n❌ Invalid device index. Please enter a number.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point."""
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
