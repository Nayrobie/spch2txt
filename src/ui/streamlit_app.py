"""
Streamlit POC for Teams Audio Transcription
Captures system audio (WASAPI loopback) and transcribes using OpenAI Whisper

How to run:
    poetry run streamlit run src/ui/streamlit_app.py
"""

import streamlit as st
import pyaudiowpatch as pyaudio
import wave
import whisper
import tempfile
import os
import time

# Page configuration
st.set_page_config(
    page_title="Speech to Text POC",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = []
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'stream' not in st.session_state:
    st.session_state.stream = None
if 'pa' not in st.session_state:
    st.session_state.pa = None

# Audio recording parameters
RATE = 48000
CHANNELS = 2
FRAMES_PER_BUFFER = 4096
SAMPLE_FORMAT = pyaudio.paInt16

def get_audio_devices():
    """Get all available audio devices with their details."""
    pa = pyaudio.PyAudio()
    devices = []
    
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        # Focus on input devices and WASAPI loopback devices
        if info['maxInputChannels'] > 0:
            devices.append({
                'index': i,
                'name': info['name'],
                'hostApi': info['hostApi'],
                'maxInputChannels': info['maxInputChannels'],
                'defaultSampleRate': info['defaultSampleRate']
            })
    
    pa.terminate()
    return devices

def get_default_loopback_device():
    """Get the default WASAPI loopback device."""
    try:
        pa = pyaudio.PyAudio()
        loopback = pa.get_default_wasapi_loopback()
        pa.terminate()
        return loopback
    except Exception as e:
        st.error(f"Error getting default loopback device: {e}")
        return None

def load_whisper_model(model_size="base"):
    """Load Whisper model."""
    with st.spinner(f"Loading Whisper {model_size} model..."):
        model = whisper.load_model(model_size)
        st.session_state.whisper_model = model
        st.session_state.model_loaded = True
    return model

def start_recording(device_index, duration=None):
    """Start recording audio from the selected device."""
    try:
        st.session_state.pa = pyaudio.PyAudio()
        st.session_state.audio_data = []
        
        # Open stream
        st.session_state.stream = st.session_state.pa.open(
            format=SAMPLE_FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
            input_device_index=device_index
        )
        
        st.session_state.recording = True
        return True
    except Exception as e:
        st.error(f"Error starting recording: {e}")
        return False

def stop_recording():
    """Stop recording and save audio data."""
    try:
        if st.session_state.stream:
            st.session_state.stream.stop_stream()
            st.session_state.stream.close()
        if st.session_state.pa:
            st.session_state.pa.terminate()
        
        st.session_state.recording = False
        return True
    except Exception as e:
        st.error(f"Error stopping recording: {e}")
        return False

def record_audio_chunk():
    """Record a single chunk of audio."""
    if st.session_state.stream and st.session_state.recording:
        try:
            data = st.session_state.stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            st.session_state.audio_data.append(data)
            return True
        except Exception as e:
            st.error(f"Error recording chunk: {e}")
            return False
    return False

def save_audio_to_file(filename):
    """Save recorded audio data to a WAV file."""
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(SAMPLE_FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(st.session_state.audio_data))
        return True
    except Exception as e:
        st.error(f"Error saving audio: {e}")
        return False

def transcribe_audio(audio_file, model):
    """Transcribe audio using Whisper."""
    try:
        with st.spinner("Transcribing audio..."):
            result = model.transcribe(audio_file)
            return result
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

# Main UI
st.title("🎤 Speech to Text POC")
st.markdown("### Real-time Audio Capture & Transcription")
st.markdown("Capture system audio (Teams/Zoom) and transcribe using OpenAI Whisper")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Whisper model selection
    model_size = st.selectbox(
        "Whisper Model Size",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Larger models are more accurate but slower"
    )
    
    if st.button("Load Model"):
        load_whisper_model(model_size)
    
    if st.session_state.model_loaded:
        st.success(f"✅ Model loaded: {model_size}")
    
    st.divider()
    
    # Recording duration
    recording_duration = st.number_input(
        "Recording Duration (seconds)",
        min_value=5,
        max_value=300,
        value=10,
        step=5,
        help="How long to record audio"
    )
    
    st.divider()
    
    # Device information
    st.header("🔊 Audio Devices")
    if st.button("Refresh Devices"):
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Audio Source")
    
    # Get available devices
    devices = get_audio_devices()
    default_loopback = get_default_loopback_device()
    
    if not devices:
        st.error("No audio devices found!")
    else:
        # Create device selection dropdown
        device_names = [f"{d['index']}: {d['name']}" for d in devices]
        
        # Try to pre-select WASAPI loopback device
        default_index = 0
        if default_loopback:
            for idx, d in enumerate(devices):
                if d['index'] == default_loopback['index']:
                    default_index = idx
                    break
        
        selected_device_str = st.selectbox(
            "Select Audio Device",
            device_names,
            index=default_index,
            help="Select the audio source (WASAPI loopback for system audio)"
        )
        
        try:
            selected_device_index = int(selected_device_str.split(":")[0])
            # Find the device in the devices list that matches the selected index
            selected_device = next((device for device in devices if device['index'] == selected_device_index), None)
            if selected_device is None:
                st.error(f"Could not find device with index {selected_device_index}")
                st.stop()
        except (ValueError, IndexError) as e:
            st.error(f"Error selecting audio device: {e}")
            st.stop()
        
        # Display device info
        with st.expander("Device Details"):
            st.json(selected_device)
        
        # Recording controls
        st.divider()
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            if not st.session_state.recording:
                if st.button("🔴 Start Recording", use_container_width=True, type="primary"):
                    if not st.session_state.model_loaded:
                        st.warning("Please load a Whisper model first!")
                    else:
                        if start_recording(selected_device['index']):
                            st.success("Recording started!")
                            st.rerun()
        
        with col_rec2:
            if st.session_state.recording:
                if st.button("⏹️ Stop Recording", use_container_width=True, type="secondary"):
                    stop_recording()
                    st.success("Recording stopped!")
                    st.rerun()
        
        # Recording status
        if st.session_state.recording:
            st.info(f"🎙️ Recording in progress... ({len(st.session_state.audio_data)} chunks captured)")
            
            # Auto-stop after duration
            if len(st.session_state.audio_data) >= int(RATE / FRAMES_PER_BUFFER * recording_duration):
                stop_recording()
                st.success(f"Recording completed ({recording_duration} seconds)")
                st.rerun()
            else:
                # Continue recording
                record_audio_chunk()
                time.sleep(0.01)
                st.rerun()

with col2:
    st.subheader("Transcription")
    
    # Transcribe button
    if st.session_state.audio_data and not st.session_state.recording:
        if st.button("📝 Transcribe Audio", use_container_width=True, type="primary"):
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_filename = tmp_file.name
            
            if save_audio_to_file(tmp_filename):
                st.success("Audio saved successfully!")
                
                # Transcribe
                if st.session_state.whisper_model:
                    result = transcribe_audio(tmp_filename, st.session_state.whisper_model)
                    
                    if result:
                        st.session_state.transcription = result['text']
                        
                        # Display transcription
                        st.success("✅ Transcription complete!")
                        
                        # Show segments if available
                        if 'segments' in result:
                            st.markdown("#### Segments:")
                            for segment in result['segments']:
                                st.markdown(f"**[{segment['start']:.2f}s - {segment['end']:.2f}s]** {segment['text']}")
                else:
                    st.error("Whisper model not loaded!")
                
                # Clean up temp file
                try:
                    os.unlink(tmp_filename)
                except Exception:
                    pass
    
    # Display transcription
    if st.session_state.transcription:
        st.divider()
        st.markdown("#### Full Transcription:")
        st.text_area(
            "Transcribed Text",
            st.session_state.transcription,
            height=300,
            label_visibility="collapsed"
        )
        
        # Download button
        st.download_button(
            "💾 Download Transcription",
            st.session_state.transcription,
            file_name="transcription.txt",
            mime="text/plain"
        )
    else:
        st.info("Record audio and click 'Transcribe Audio' to see results here.")

# Footer
st.divider()
st.markdown("""
### 📋 Instructions:
1. **Load Model**: Select a Whisper model size and click 'Load Model' (first time only)
2. **Select Device**: Choose the audio source (WASAPI loopback for system audio like Teams/Zoom)
3. **Record**: Click 'Start Recording' and let it capture audio for the specified duration
4. **Transcribe**: After recording stops, click 'Transcribe Audio' to get the text

**Note**: For capturing Teams/Zoom audio, select a WASAPI loopback device that corresponds to your system audio output.
""")
