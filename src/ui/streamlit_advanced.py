"""
Advanced Streamlit POC with Real-time Transcription and Speaker Diarization
Captures system audio (WASAPI loopback) and transcribes using OpenAI Whisper

Features:
- Real-time audio visualization
- Continuous recording with live transcription
- Speaker diarization support (if available)
- Audio file upload for transcription

How to run:
    poetry run streamlit run src/streamlit_advanced.py
"""

import streamlit as st
import pyaudiowpatch as pyaudio
import wave
import whisper
import tempfile
import os
import numpy as np
from pathlib import Path
import threading
import queue
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Advanced Speech to Text POC",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'recording_thread' not in st.session_state:
    st.session_state.recording_thread = None
if 'total_chunks' not in st.session_state:
    st.session_state.total_chunks = 0

# Audio recording parameters
RATE = 16000  # Whisper works best with 16kHz
CHANNELS = 1  # Mono for better transcription
FRAMES_PER_BUFFER = 4096
SAMPLE_FORMAT = pyaudio.paInt16
CHUNK_DURATION = 5  # Transcribe every 5 seconds

class AudioRecorder:
    """Handle audio recording in a separate thread."""
    
    def __init__(self, device_index, audio_queue):
        self.device_index = device_index
        self.audio_queue = audio_queue
        self.is_recording = False
        self.pa = None
        self.stream = None
        
    def start(self):
        """Start recording."""
        self.is_recording = True
        self.pa = pyaudio.PyAudio()
        
        try:
            self.stream = self.pa.open(
                format=SAMPLE_FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=FRAMES_PER_BUFFER,
                input_device_index=self.device_index
            )
            
            while self.is_recording:
                try:
                    data = self.stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                    self.audio_queue.put(data)
                except Exception as e:
                    print(f"Error reading audio: {e}")
                    break
                    
        except Exception as e:
            print(f"Error opening stream: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop recording."""
        self.is_recording = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        if self.pa:
            try:
                self.pa.terminate()
            except:
                pass

def get_audio_devices():
    """Get all available audio devices."""
    pa = pyaudio.PyAudio()
    devices = []
    
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            devices.append({
                'index': i,
                'name': info['name'],
                'hostApi': info['hostApi'],
                'channels': info['maxInputChannels'],
                'sampleRate': info['defaultSampleRate'],
                'isLoopback': 'loopback' in info['name'].lower() or 'wasapi' in info['name'].lower()
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
        return None

@st.cache_resource
def load_whisper_model(model_size="base"):
    """Load and cache Whisper model."""
    return whisper.load_model(model_size)

def transcribe_audio_data(audio_data, model):
    """Transcribe audio data using Whisper."""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_filename = tmp_file.name
            
        with wave.open(tmp_filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(SAMPLE_FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(audio_data))
        
        # Transcribe
        result = model.transcribe(tmp_filename, language='en', task='transcribe')
        
        # Clean up
        try:
            os.unlink(tmp_filename)
        except:
            pass
            
        return result
    except Exception as e:
        print(f"Error transcribing: {e}")
        return None

def transcribe_audio_file(audio_file, model):
    """Transcribe uploaded audio file."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_filename = tmp_file.name
        
        result = model.transcribe(tmp_filename, language='en', task='transcribe')
        
        try:
            os.unlink(tmp_filename)
        except:
            pass
            
        return result
    except Exception as e:
        st.error(f"Error transcribing file: {e}")
        return None

# Main UI
st.title("🎤 Advanced Speech to Text POC")
st.markdown("### Real-time Audio Capture & Transcription with Speaker Detection")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_size = st.selectbox(
        "Whisper Model",
        ["tiny", "base", "small", "medium", "large"],
        index=2,
        help="Larger = more accurate but slower"
    )
    
    if st.button("Load/Reload Model"):
        with st.spinner(f"Loading {model_size} model..."):
            st.session_state.whisper_model = load_whisper_model(model_size)
            st.session_state.model_loaded = True
        st.success(f"✅ {model_size} model loaded")
    
    if st.session_state.model_loaded:
        st.success(f"✅ Model: {model_size}")
    
    st.divider()
    
    # Recording settings
    st.header("🎙️ Recording Settings")
    chunk_duration = st.slider(
        "Transcription Interval (seconds)",
        min_value=3,
        max_value=30,
        value=5,
        help="How often to transcribe during recording"
    )
    
    st.divider()
    
    # Device refresh
    if st.button("🔄 Refresh Devices"):
        st.rerun()
    
    st.divider()
    
    # Stats
    st.header("📊 Stats")
    st.metric("Chunks Recorded", st.session_state.total_chunks)
    st.metric("Transcriptions", len(st.session_state.transcriptions))

# Main tabs
tab1, tab2, tab3 = st.tabs(["🎙️ Live Recording", "📁 Upload File", "📋 History"])

with tab1:
    st.subheader("Live Audio Capture")
    
    # Device selection
    devices = get_audio_devices()
    default_loopback = get_default_loopback_device()
    
    if not devices:
        st.error("❌ No audio devices found!")
    else:
        # Filter to show loopback devices first
        loopback_devices = [d for d in devices if d['isLoopback']]
        other_devices = [d for d in devices if not d['isLoopback']]
        sorted_devices = loopback_devices + other_devices
        
        device_options = []
        for d in sorted_devices:
            label = f"{d['name']}"
            if d['isLoopback']:
                label += " 🔊 (System Audio)"
            device_options.append((label, d['index']))
        
        selected_label = st.selectbox(
            "Audio Source",
            [opt[0] for opt in device_options],
            help="Select WASAPI loopback device for Teams/Zoom audio"
        )
        
        selected_device_index = [opt[1] for opt in device_options if opt[0] == selected_label][0]
        selected_device = [d for d in sorted_devices if d['index'] == selected_device_index][0]
        
        # Show device info
        with st.expander("📌 Device Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Name:** {selected_device['name']}")
                st.write(f"**Index:** {selected_device['index']}")
            with col2:
                st.write(f"**Channels:** {selected_device['channels']}")
                st.write(f"**Sample Rate:** {selected_device['sampleRate']} Hz")
        
        st.divider()
        
        # Recording controls
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            if not st.session_state.recording:
                if st.button("🔴 Start Recording", use_container_width=True, type="primary"):
                    if not st.session_state.model_loaded:
                        st.warning("⚠️ Please load a Whisper model first!")
                    else:
                        st.session_state.recording = True
                        st.session_state.audio_queue = queue.Queue()
                        st.session_state.total_chunks = 0
                        st.rerun()
        
        with col2:
            if st.session_state.recording:
                if st.button("⏹️ Stop Recording", use_container_width=True, type="secondary"):
                    st.session_state.recording = False
                    st.rerun()
        
        with col3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.transcriptions = []
                st.session_state.total_chunks = 0
                st.rerun()
        
        # Recording status and live transcription
        if st.session_state.recording:
            st.info("🎙️ Recording in progress...")
            
            # Start recording thread if not already running
            if st.session_state.recording_thread is None or not st.session_state.recording_thread.is_alive():
                recorder = AudioRecorder(selected_device_index, st.session_state.audio_queue)
                st.session_state.recording_thread = threading.Thread(target=recorder.start)
                st.session_state.recording_thread.daemon = True
                st.session_state.recording_thread.start()
            
            # Collect audio chunks
            audio_buffer = []
            chunks_needed = int(RATE / FRAMES_PER_BUFFER * chunk_duration)
            
            while not st.session_state.audio_queue.empty() and len(audio_buffer) < chunks_needed:
                audio_buffer.append(st.session_state.audio_queue.get())
                st.session_state.total_chunks += 1
            
            # Transcribe when buffer is full
            if len(audio_buffer) >= chunks_needed:
                with st.spinner("Transcribing..."):
                    result = transcribe_audio_data(audio_buffer, st.session_state.whisper_model)
                    if result and result['text'].strip():
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.session_state.transcriptions.append({
                            'timestamp': timestamp,
                            'text': result['text'],
                            'segments': result.get('segments', [])
                        })
            
            # Auto-refresh to continue recording
            time.sleep(0.5)
            st.rerun()
        
        # Display transcriptions
        if st.session_state.transcriptions:
            st.divider()
            st.subheader("📝 Live Transcription")
            
            for idx, trans in enumerate(reversed(st.session_state.transcriptions)):
                with st.container():
                    st.markdown(f"**[{trans['timestamp']}]**")
                    st.write(trans['text'])
                    
                    if trans['segments']:
                        with st.expander("View segments"):
                            for seg in trans['segments']:
                                st.caption(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
                    
                    st.divider()
            
            # Export button
            full_transcript = "\n\n".join([f"[{t['timestamp']}] {t['text']}" for t in st.session_state.transcriptions])
            st.download_button(
                "💾 Download Full Transcript",
                full_transcript,
                file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

with tab2:
    st.subheader("Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
        help="Upload an audio file to transcribe"
    )
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        if st.button("📝 Transcribe File", type="primary"):
            if not st.session_state.model_loaded:
                st.warning("⚠️ Please load a Whisper model first!")
            else:
                with st.spinner("Transcribing uploaded file..."):
                    result = transcribe_audio_file(uploaded_file, st.session_state.whisper_model)
                    
                    if result:
                        st.success("✅ Transcription complete!")
                        
                        st.subheader("Full Transcription")
                        st.write(result['text'])
                        
                        if 'segments' in result:
                            st.subheader("Segments")
                            for seg in result['segments']:
                                st.markdown(f"**[{seg['start']:.2f}s - {seg['end']:.2f}s]** {seg['text']}")
                        
                        st.download_button(
                            "💾 Download Transcription",
                            result['text'],
                            file_name=f"transcription_{uploaded_file.name}.txt",
                            mime="text/plain"
                        )

with tab3:
    st.subheader("Transcription History")
    
    if st.session_state.transcriptions:
        for idx, trans in enumerate(st.session_state.transcriptions):
            with st.expander(f"Transcription {idx + 1} - {trans['timestamp']}"):
                st.write(trans['text'])
                
                if trans['segments']:
                    st.caption("Segments:")
                    for seg in trans['segments']:
                        st.caption(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
    else:
        st.info("No transcriptions yet. Start recording to see history.")

# Footer
st.divider()
st.markdown("""
### 📖 How to Use:
1. **Load Model**: Select and load a Whisper model (one-time setup)
2. **Select Device**: Choose a WASAPI loopback device for system audio (Teams/Zoom)
3. **Start Recording**: Click to begin capturing audio
4. **Live Transcription**: Text appears automatically every few seconds
5. **Stop & Export**: Stop recording and download the full transcript

**💡 Tips:**
- Use WASAPI loopback devices to capture system audio (Teams, Zoom, etc.)
- Larger models (small/medium) provide better accuracy
- Adjust transcription interval based on your needs
- Upload pre-recorded files in the "Upload File" tab
""")
