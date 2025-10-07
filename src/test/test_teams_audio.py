import wave
import pyaudiowpatch as pyaudio
import whisper
import numpy as np
from datetime import datetime

"""
Test script for Teams audio capture.

This script helps you:
1. Identify Teams audio devices
2. Test recording from Teams
3. Prepare for speaker diarization

How to run:
    poetry run python src/audio/test_teams_audio.py
"""

DURATION = 10
RATE = 16000
CHANNELS = 1
FRAMES_PER_BUFFER = 1024

def list_all_devices():
    """List all audio devices with detailed information."""
    print("="*70)
    print("AVAILABLE AUDIO DEVICES")
    print("="*70)
    
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
    
    # Display input devices
    print("\n📥 INPUT DEVICES (Microphones):")
    print("-" * 70)
    for dev in input_devices:
        print(f"  [{dev['index']}] {dev['name']}")
        print(f"      Channels: {dev['maxInputChannels']}, Rate: {dev['defaultSampleRate']:.0f}Hz")
    
    # Display loopback devices
    print("\n🔄 LOOPBACK DEVICES (System Audio Capture):")
    print("-" * 70)
    if loopback_devices:
        for dev in loopback_devices:
            print(f"  [{dev['index']}] {dev['name']}")
            print(f"      Channels: {dev['maxInputChannels']}, Rate: {dev['defaultSampleRate']:.0f}Hz")
    else:
        print("  No loopback devices found")
    
    # Display output devices
    print("\n📤 OUTPUT DEVICES (Speakers):")
    print("-" * 70)
    for dev in output_devices:
        print(f"  [{dev['index']}] {dev['name']}")
        print(f"      Channels: {dev['maxOutputChannels']}, Rate: {dev['defaultSampleRate']:.0f}Hz")
    
    pa.terminate()
    
    print("\n" + "="*70)
    print("💡 TIPS FOR TEAMS AUDIO:")
    print("="*70)
    print("1. During a Teams call, look for devices with 'Teams' in the name")
    print("2. Loopback devices capture ALL system audio (including Teams)")
    print("3. For best results, use a device that captures only Teams audio")
    print("="*70)
    
    return input_devices, loopback_devices, output_devices

def record_from_device(device_index, device_name, output_file="teams_test.wav"):
    """Record audio from a specific device."""
    print("\n" + "="*70)
    print(f"RECORDING FROM: {device_name}")
    print("="*70)
    print(f"Device Index: {device_index}")
    print(f"Duration: {DURATION} seconds")
    print(f"Output: {output_file}")
    print("-" * 70)
    
    pa = pyaudio.PyAudio()
    
    # Open wave file
    wf = wave.open(output_file, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    
    try:
        # Open audio stream
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
            input_device_index=device_index
        )
        
        print("🎤 Recording... (speak or play Teams audio)")
        
        # Record
        num_chunks = int(RATE / FRAMES_PER_BUFFER * DURATION)
        frames = []
        
        for i in range(num_chunks):
            data = stream.read(FRAMES_PER_BUFFER)
            frames.append(data)
            
            if i % 15 == 0:
                progress = (i / num_chunks) * 100
                bars = int(progress / 5)
                print(f"[{'█' * bars}{' ' * (20-bars)}] {progress:.0f}%", end='\r')
        
        print(f"[{'█' * 20}] 100%")
        
        # Save
        wf.writeframes(b"".join(frames))
        stream.close()
        
        print(f"✓ Recording saved to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Error recording from device: {e}")
        return None
    finally:
        pa.terminate()
        wf.close()

def quick_transcribe(audio_file):
    """Quick transcription without full workflow."""
    print("\n" + "="*70)
    print("QUICK TRANSCRIPTION")
    print("="*70)
    
    try:
        # Load audio
        with wave.open(audio_file, 'rb') as wf:
            audio_data = wf.readframes(wf.getnframes())
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        
        print("Transcribing...")
        result = model.transcribe(audio, language="en", verbose=False)
        
        print("\n📝 Transcription:")
        print("-" * 70)
        print(result["text"])
        print("-" * 70)
        
        return result
        
    except Exception as e:
        print(f"❌ Error transcribing: {e}")
        return None

def interactive_menu():
    """Interactive menu for testing."""
    print("\n" + "="*70)
    print("TEAMS AUDIO TESTING - INTERACTIVE MODE")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # List devices
    input_devs, loopback_devs, output_devs = list_all_devices()
    
    print("\n" + "="*70)
    print("WHAT WOULD YOU LIKE TO DO?")
    print("="*70)
    print("1. Record from default microphone")
    print("2. Record from loopback device (system audio)")
    print("3. Record from specific device (enter index)")
    print("4. Just list devices and exit")
    print("="*70)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        # Default microphone
        pa = pyaudio.PyAudio()
        default_input = pa.get_default_input_device_info()
        pa.terminate()
        
        print(f"\n✓ Using default microphone: {default_input['name']}")
        audio_file = record_from_device(default_input['index'], default_input['name'])
        
        if audio_file:
            transcribe = input("\nTranscribe this recording? (y/n): ").strip().lower()
            if transcribe == 'y':
                quick_transcribe(audio_file)
    
    elif choice == "2":
        # Loopback device
        if loopback_devs:
            device = loopback_devs[0]
            print(f"\n✓ Using loopback device: {device['name']}")
            audio_file = record_from_device(device['index'], device['name'])
            
            if audio_file:
                transcribe = input("\nTranscribe this recording? (y/n): ").strip().lower()
                if transcribe == 'y':
                    quick_transcribe(audio_file)
        else:
            print("\n❌ No loopback devices found on your system")
    
    elif choice == "3":
        # Specific device
        device_index = input("\nEnter device index: ").strip()
        try:
            device_index = int(device_index)
            pa = pyaudio.PyAudio()
            device_info = pa.get_device_info_by_index(device_index)
            pa.terminate()
            
            print(f"\n✓ Using device: {device_info['name']}")
            audio_file = record_from_device(device_index, device_info['name'])
            
            if audio_file:
                transcribe = input("\nTranscribe this recording? (y/n): ").strip().lower()
                if transcribe == 'y':
                    quick_transcribe(audio_file)
        except (ValueError, Exception) as e:
            print(f"\n❌ Invalid device index: {e}")
    
    elif choice == "4":
        print("\n✓ Device list complete. Exiting.")
    
    else:
        print("\n❌ Invalid choice")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
