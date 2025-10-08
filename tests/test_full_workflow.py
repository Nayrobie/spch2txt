import wave
import pyaudiowpatch as pyaudio
import whisper
import numpy as np
import os
import time
import threading
import queue
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

def record_stream_thread(stream, device_name, is_loopback, channels, num_chunks, frames_queue, stop_event):
    """Thread function to record from a single stream."""
    chunk_count = 0
    silence = b'\x00' * (FRAMES_PER_BUFFER * 2 * channels)
    consecutive_errors = 0
    max_consecutive_errors = 100  # Allow more errors for loopback devices
    
    while chunk_count < num_chunks and not stop_event.is_set():
        try:
            # Always try to read audio data
            # For loopback devices, this will capture whatever is playing
            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            frames_queue.put(data)
            chunk_count += 1
            consecutive_errors = 0
            
        except Exception as e:
            # On error, append silence and continue
            consecutive_errors += 1
            frames_queue.put(silence)
            chunk_count += 1
            
            # If too many consecutive errors, something is wrong
            if consecutive_errors >= max_consecutive_errors:
                print(f"\n  ⚠ Warning: Too many errors reading from {device_name}, stopping thread")
                break
            
            time.sleep(0.01)

def record_audio(device_indices, device_names, channels_list, rates, duration=None, stop_flag=None):
    """Record audio from multiple devices simultaneously using threads.
    
    Args:
        device_indices: List of device indices
        device_names: List of device names
        channels_list: List of channel counts
        rates: List of sample rates
        duration: Recording duration in seconds, or None for unlimited
        stop_flag: Optional streamlit session state flag for stopping unlimited recording
    """
    print("\nRECORDING AUDIO")
    print("-" * 50)
    for i, name in enumerate(device_names):
        print(f"Device {i+1}: {name}")
        print(f"Settings: {rates[i]}Hz, {channels_list[i]} channel(s)")
    if duration:
        print(f"Duration: {duration} seconds")
    else:
        print("Duration: Unlimited (stop manually)")
    
    # Determine which devices are loopback
    is_loopback = ['loopback' in name.lower() for name in device_names]
    for i, name in enumerate(device_names):
        if is_loopback[i]:
            print(f"  ⚠ Device {i+1} is loopback - will only capture if audio is playing!")
    
    print("-" * 80)
    
    # Create output filenames (with device names, timestamp, and index)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_files = []
    for i, name in enumerate(device_names):
        # Clean device name for filename (remove special characters)
        clean_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
        clean_name = clean_name.replace(' ', '_')
        # Truncate if too long
        if len(clean_name) > 50:
            clean_name = clean_name[:50]
        filename = f"recording_{timestamp}_dev{i+1}_{clean_name}.wav"
        output_files.append(os.path.join(OUTPUT_DIR, filename))
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pa = pyaudio.PyAudio()
    
    try:
        # Open audio streams
        streams = []
        
        for i, (device_index, channels, rate) in enumerate(zip(device_indices, channels_list, rates)):
            try:
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=FRAMES_PER_BUFFER,
                    input_device_index=device_index
                )
                streams.append(stream)
                print(f"✓ Opened stream for device {i+1}: {device_names[i]}")
            except Exception as e:
                print(f"❌ Failed to open stream for device {i+1}: {e}")
                # Close any opened streams
                for s in streams:
                    s.close()
                pa.terminate()
                return None
        
        print("\nRecording... (speak or play audio now)")
        
        # Calculate number of chunks needed for each device
        num_chunks_per_device = [int(rate * duration / FRAMES_PER_BUFFER) for rate in rates]
        
        # Create queues and threads for each stream
        queues = [queue.Queue() for _ in device_indices]
        stop_event = threading.Event()
        threads = []
        
        for i, (stream, name, is_loop, channels, num_chunks) in enumerate(
            zip(streams, device_names, is_loopback, channels_list, num_chunks_per_device)
        ):
            thread = threading.Thread(
                target=record_stream_thread,
                args=(stream, name, is_loop, channels, num_chunks, queues[i], stop_event)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Monitor progress
        start_time = time.time()
        max_chunks = max(num_chunks_per_device)
        
        try:
            while any(t.is_alive() for t in threads):
                elapsed = time.time() - start_time
                progress = min(100, (elapsed / duration) * 100) if duration else 0
                
                if duration:
                    bars = int(progress / 5)
                    print(f"[{'█' * bars}{' ' * (20-bars)}] {progress:.0f}%", end='\r')
                
                time.sleep(0.1)
                
                # Check if duration exceeded
                if duration and elapsed >= duration:
                    stop_event.set()
                    break
        except KeyboardInterrupt:
            print("\nRecording stopped by user")
            stop_event.set()
        
        # Wait for all threads to finish
        print("\nWaiting for recording threads to complete...")
        for i, thread in enumerate(threads):
            thread.join(timeout=2.0)
            if thread.is_alive():
                print(f"  Warning: Thread {i+1} still running after timeout (likely loopback with no audio)")
        
        # Force stop any remaining threads
        stop_event.set()
        
        print(f"[{'█' * 20}] 100%")
        print("\nSaving recordings...")
        
        # Collect frames from queues and save to files
        for i, (output_file, q) in enumerate(zip(output_files, queues)):
            print(f"  Processing device {i+1} queue (size: {q.qsize()})...")
            frames = []
            while not q.empty():
                try:
                    frames.append(q.get_nowait())
                except:
                    break
            
            print(f"  Collected {len(frames)} chunks from device {i+1}")
            
            # If no frames collected, create a silent audio file
            if len(frames) == 0:
                print(f"  Warning: No audio data collected from device {i+1}, creating silent file...")
                # Create silent audio for the duration
                num_samples = int(rates[i] * duration * channels_list[i])
                frames = [b'\x00' * (FRAMES_PER_BUFFER * 2 * channels_list[i])] * int(num_samples / (FRAMES_PER_BUFFER * channels_list[i]))
            
            # Save to WAV file
            print(f"  Saving to {output_file}...")
            try:
                with wave.open(output_file, 'wb') as wf:
                    wf.setnchannels(channels_list[i])
                    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(rates[i])
                    wf.writeframes(b"".join(frames))
                print(f"  ✓ Saved device {i+1}")
            except Exception as e:
                print(f"  ❌ Error saving device {i+1}: {e}")
        
        # Close streams
        print("\nClosing audio streams...")
        for i, stream in enumerate(streams):
            try:
                # Stop the stream first to unblock any pending reads
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
                print(f"  ✓ Closed stream {i+1}")
            except Exception as e:
                print(f"  ⚠ Error closing stream {i+1}: {e}")
        
        print("\n✓ Recordings saved:")
        for file in output_files:
            print(f"  - {file}")
        
        return output_files
        
    except Exception as e:
        print(f"❌ Error recording: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        pa.terminate()

def mix_wav_files(filepaths):
    """Mix multiple WAV files into a single audio stream."""
    import sys
    audio_data = []
    
    # Load all audio files
    for i, filepath in enumerate(filepaths):
        # Normalize path to avoid double backslash issues
        filepath = os.path.normpath(filepath)
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"  ⚠ Warning: File not found: {filepath}", flush=True)
            print(f"  Skipping file {i+1}", flush=True)
            continue
        
        print(f"  Loading file {i+1}/{len(filepaths)}: {filepath}", flush=True)
        sys.stdout.flush()
        
        with wave.open(filepath, 'rb') as wf:
            n_channels = wf.getnchannels()
            rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            print(f"    Channels: {n_channels}, Rate: {rate}Hz, Frames: {n_frames}", flush=True)
            sys.stdout.flush()
            
            audio = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
            print(f"    Loaded {len(audio)} samples", flush=True)
            sys.stdout.flush()
            
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
                print(f"    Converted stereo to mono: {len(audio)} samples", flush=True)
                sys.stdout.flush()
            
            # Convert to float32 and normalize
            audio = audio.astype(np.float32) / 32768.0
            
            # Resample if needed (using numpy for compatibility)
            if rate != 48000:
                print(f"    Resampling from {rate}Hz to 48000Hz...", flush=True)
                sys.stdout.flush()
                duration = len(audio) / rate
                target_length = int(duration * 48000)
                audio = np.interp(
                    np.linspace(0, len(audio), target_length, dtype=np.float32),
                    np.arange(len(audio), dtype=np.float32),
                    audio
                )
                print(f"    Resampled to {len(audio)} samples", flush=True)
                sys.stdout.flush()
            
            audio_data.append(audio)
            print(f"  ✓ File {i+1} loaded", flush=True)
            sys.stdout.flush()
    
    # Check if we have any audio data
    if not audio_data:
        print("  ❌ No audio files could be loaded!", flush=True)
        sys.stdout.flush()
        # Return empty audio
        return np.zeros(48000, dtype=np.float32)
    
    print("  Mixing audio streams...", flush=True)
    sys.stdout.flush()
    
    # Find the longest length
    max_length = max(len(audio) for audio in audio_data)
    print(f"  Max length: {max_length} samples ({max_length/48000:.1f}s)", flush=True)
    sys.stdout.flush()
    
    # Pad shorter audio with zeros to match the longest length
    padded_audio = []
    for i, audio in enumerate(audio_data):
        if len(audio) < max_length:
            # Pad with zeros (silence) to match longest file
            padding = np.zeros(max_length - len(audio), dtype=np.float32)
            padded_audio.append(np.concatenate([audio, padding]))
            print(f"  Padded audio {i+1} from {len(audio)} to {max_length} samples", flush=True)
            sys.stdout.flush()
        else:
            padded_audio.append(audio)
    
    # Mix the audio streams (average them where both exist, keep single stream otherwise)
    print("  Averaging audio streams...", flush=True)
    sys.stdout.flush()
    mixed_audio = np.zeros(max_length, dtype=np.float32)
    for audio in padded_audio:
        mixed_audio += audio
    mixed_audio /= len(audio_data)  # Average where streams overlap
    
    # Normalize the mixed audio
    print("  Normalizing mixed audio...", flush=True)
    sys.stdout.flush()
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0:
        mixed_audio = mixed_audio / max_val
    
    print("  ✓ Mixing complete", flush=True)
    sys.stdout.flush()
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
    print("\nNow, select the LOOPBACK device for Teams/system audio (from the LOOPBACK DEVICES section above):")
    loopback_index = input("Enter loopback device index: ").strip()
    
    try:
        mic_index = int(mic_index)
        loopback_index = int(loopback_index)
        
        # Get device info
        pa = pyaudio.PyAudio()
        mic_info = pa.get_device_info_by_index(mic_index)
        loopback_info = pa.get_device_info_by_index(loopback_index)
        pa.terminate()
        
        # Validate microphone
        if mic_info['maxInputChannels'] <= 0:
            print("\n❌ Error: Selected microphone is not an input device")
            print("Please select a device from the INPUT DEVICES section")
            return
        
        # Validate loopback (must be input-capable for capture)
        if loopback_info['maxInputChannels'] <= 0:
            print("\n❌ Error: Selected device is not an input (loopback) device")
            print("Please select a device from the LOOPBACK DEVICES section")
            return
        
        # Set up device configurations
        device_indices = [mic_index, loopback_index]
        device_names = [mic_info['name'], loopback_info['name']]
        channels_list = [min(mic_info['maxInputChannels'], 2), min(loopback_info['maxInputChannels'], 2)]
        rates = [int(mic_info['defaultSampleRate']), int(loopback_info['defaultSampleRate'])]
        
        print("\n✓ Selected devices:")
        print(f"  Microphone: {mic_info['name']}")
        print(f"  Loopback:   {loopback_info['name']}")
        
        # Confirm
        confirm = input("\nProceed with recording? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Recording cancelled.")
            return
        
        # Record for configured duration
        audio_files = record_audio(device_indices, device_names, channels_list, rates, duration=DURATION)
        
        if not audio_files:
            print("Recording failed.")
            return
        
        # Ask about transcription
        import sys
        print("", flush=True)
        sys.stdout.flush()
        transcribe = input("\nTranscribe recordings? (y/n): ").strip().lower()
        if transcribe == 'y':
            print("\n" + "="*80)
            print("MIXING AND TRANSCRIBING AUDIO")
            print("="*80)
            
            try:
                # Mix the audio files
                print("Mixing audio files...")
                mixed_audio = mix_wav_files(audio_files)
                print(f"✓ Mixed audio length: {len(mixed_audio)/48000:.1f} seconds")
                
                # Create a temporary WAV file for the mixed audio
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                mixed_file = os.path.join(OUTPUT_DIR, f"recording_{timestamp}_mixed.wav")
                
                # Save mixed audio
                print(f"Saving mixed audio to: {mixed_file}")
                with wave.open(mixed_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(48000)
                    # Convert back to int16
                    mixed_audio_int = (mixed_audio * 32767).astype(np.int16)
                    wf.writeframes(mixed_audio_int.tobytes())
                print("✓ Mixed audio saved")
                
                # Transcribe
                result = transcribe_audio(mixed_file)
                display_results(result)
                
                # Clean up temporary file
                print("\nCleaning up temporary files...")
                os.remove(mixed_file)
                print("✓ Cleanup complete")
                
            except Exception as e:
                print(f"\n❌ Error during transcription: {e}")
                import traceback
                traceback.print_exc()
        
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