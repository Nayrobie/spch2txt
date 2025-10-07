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
    print("\nAVAILABLE AUDIO DEVICES")
    print("-" * 50)
    
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
    print("\nINPUT DEVICES:")
    if input_devices:
        for dev in input_devices:
            print(f"  [{dev['index']:2d}] {dev['name']}")
            print(f"       Channels: {dev['maxInputChannels']}, Rate: {dev['defaultSampleRate']:.0f}Hz")
    else:
        print("  No input devices found")
    
    # Display loopback devices (system audio)
    print("\nLOOPBACK DEVICES (System Audio):")
    if loopback_devices:
        for dev in loopback_devices:
            print(f"  [{dev['index']:2d}] {dev['name']}")
            print(f"       Channels: {dev['maxInputChannels']}, Rate: {dev['defaultSampleRate']:.0f}Hz")
    else:
        print("  No loopback devices found")
    
    # Display output devices (for reference)
    print("\nOUTPUT DEVICES:")
    for dev in output_devices:
        print(f"  [{dev['index']:2d}] {dev['name']}")
        print(f"       Channels: {dev['maxOutputChannels']}, Rate: {dev['defaultSampleRate']:.0f}Hz")
    
    pa.terminate()
    
    # Guidance
    print("\nGUIDE:")
    print("- Select microphone for voice recording")
    print("- Select loopback device for system audio (Teams/speakers)")
    
    return input_devices, loopback_devices, output_devices

def record_audio(device_indices, device_names, channels_list, rates):
    """Record audio from multiple devices simultaneously."""
    print("\nRECORDING AUDIO")
    print("-" * 50)
    for i, name in enumerate(device_names):
        print(f"Device {i+1}: {name}")
        print(f"Settings: {rates[i]}Hz, {channels_list[i]} channel(s)")
    print(f"Duration: {DURATION} seconds")
    print("-" * 80)
    
    # Create output filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_files = [
        os.path.join(OUTPUT_DIR, f"recording_{timestamp}_device{i+1}.wav")
        for i in range(len(device_indices))
    ]
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pa = pyaudio.PyAudio()
    
    # Open wave files
    wfs = []
    for i, output_file in enumerate(output_files):
        wf = wave.open(output_file, "wb")
        wf.setnchannels(channels_list[i])
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rates[i])
        wfs.append(wf)
    
    try:
        # Open audio streams
        streams = []
        frames_list = [[] for _ in device_indices]
        
        for device_index, channels, rate in zip(device_indices, channels_list, rates):
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=FRAMES_PER_BUFFER,
                input_device_index=device_index
            )
            streams.append(stream)
        
        print("Recording... (speak or play audio now)")
        
        # Record in chunks
        num_chunks = int(min(rates) / FRAMES_PER_BUFFER * DURATION)
        
        for i in range(num_chunks):
            for j, stream in enumerate(streams):
                data = stream.read(FRAMES_PER_BUFFER)
                frames_list[j].append(data)
            
            # Progress indicator
            if i % 15 == 0:
                progress = (i / num_chunks) * 100
                bars = int(progress / 5)
                print(f"[{'█' * bars}{' ' * (20-bars)}] {progress:.0f}%", end='\r')
        
        print(f"[{'█' * 20}] 100%")
        
        # Save and cleanup
        for frames, wf in zip(frames_list, wfs):
            wf.writeframes(b"".join(frames))
        
        for stream in streams:
            stream.close()
        
        for wf in wfs:
            wf.close()
        
        print("✓ Recordings saved:")
        for file in output_files:
            print(f"  - {file}")
        return output_files
        
    except Exception as e:
        print(f"❌ Error recording: {e}")
        return None
    finally:
        pa.terminate()

def mix_wav_files(filepaths):
    """Mix multiple WAV files into a single audio stream."""
    audio_data = []
    
    # Load all audio files
    for filepath in filepaths:
        with wave.open(filepath, 'rb') as wf:
            n_channels = wf.getnchannels()
            rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            audio = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
            
            # Convert to float32 and normalize
            audio = audio.astype(np.float32) / 32768.0
            
            # Resample if needed (using numpy for compatibility)
            if rate != 48000:
                duration = len(audio) / rate
                target_length = int(duration * 48000)
                audio = np.interp(
                    np.linspace(0, len(audio), target_length, dtype=np.float32),
                    np.arange(len(audio), dtype=np.float32),
                    audio
                )
            
            audio_data.append(audio)
    
    # Find the shortest length
    min_length = min(len(audio) for audio in audio_data)
    
    # Trim all audio to the same length
    audio_data = [audio[:min_length] for audio in audio_data]
    
    # Mix the audio streams (average them)
    mixed_audio = np.mean(audio_data, axis=0)
    
    # Normalize the mixed audio
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0:
        mixed_audio = mixed_audio / max_val
    
    return mixed_audio

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
    print("\nTRANSCRIPTION RESULTS")
    print("-" * 50)
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
    print("\nSPEECH TO TEXT - RECORD & TRANSCRIBE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List all devices with guidance
    input_devs, loopback_devs, output_devs = list_all_devices()
    
    # Get user choices
    print("\nSELECT DEVICES:")
    print("-" * 50)
    print("First, select your microphone for voice input:")
    mic_index = input("Enter microphone device index: ").strip()
    print("\nNow, select the speakers/output device for Teams audio:")
    speaker_index = input("Enter speaker device index: ").strip()
    
    try:
        mic_index = int(mic_index)
        speaker_index = int(speaker_index)
        
        # Get device info
        pa = pyaudio.PyAudio()
        mic_info = pa.get_device_info_by_index(mic_index)
        speaker_info = pa.get_device_info_by_index(speaker_index)
        pa.terminate()
        
        # Validate microphone
        if mic_info['maxInputChannels'] <= 0:
            print("\n❌ Error: Selected microphone is not an input device")
            print("Please select a device from the INPUT DEVICES section")
            return
        
        # Set up device configurations
        device_indices = [mic_index, speaker_index]
        device_names = [mic_info['name'], speaker_info['name']]
        channels_list = [min(mic_info['maxInputChannels'], 2), 2]
        rates = [int(mic_info['defaultSampleRate']), int(speaker_info['defaultSampleRate'])]
        
        print("\n✓ Selected devices:")
        print(f"  Microphone: {mic_info['name']}")
        print(f"  Speaker: {speaker_info['name']}")
        
        # Confirm
        confirm = input("\nProceed with recording? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Recording cancelled.")
            return
        
        # Record
        audio_files = record_audio(device_indices, device_names, channels_list, rates)
        
        if not audio_files:
            print("Recording failed.")
            return
        
        # Ask about transcription
        transcribe = input("\nTranscribe recordings? (y/n): ").strip().lower()
        if transcribe == 'y':
            print("\nMixing and transcribing audio...")
            # Mix the audio files
            mixed_audio = mix_wav_files(audio_files)
            
            # Create a temporary WAV file for the mixed audio
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            mixed_file = os.path.join(OUTPUT_DIR, f"recording_{timestamp}_mixed.wav")
            
            # Save mixed audio
            with wave.open(mixed_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                # Convert back to int16
                mixed_audio_int = (mixed_audio * 32767).astype(np.int16)
                wf.writeframes(mixed_audio_int.tobytes())
            
            print("\nTranscribing mixed audio:")
            result = transcribe_audio(mixed_file)
            display_results(result)
            
            # Clean up temporary file
            os.remove(mixed_file)
        
        print("\n✓ TEST COMPLETE")
        print("-" * 50)
        print("Audio files saved:")
        for file in audio_files:
            print(f"  - {file}")
        
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
