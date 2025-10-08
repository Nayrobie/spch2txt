import streamlit as st
import sys
import os
import wave
import numpy as np
import pyaudiowpatch as pyaudio
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tests.test_full_workflow import list_all_devices, record_audio, mix_wav_files, transcribe_audio

st.title("Audio Recording Test")

# Select duration
duration_type = st.radio(
    "Recording Duration",
    ["30 seconds", "Unlimited (press Stop)"]
)

# Get and display available devices
input_devices, loopback_devices, _ = list_all_devices()

# Select devices
# TODO: make the selection automatic and detecting headphones vs normal loopback device, the microphone should be the default one
st.subheader("Select Devices")
mic_options = {f"{dev['index']}: {dev['name']}": dev['index'] for dev in input_devices}
loopback_options = {f"{dev['index']}: {dev['name']}": dev['index'] for dev in loopback_devices}

# Find default devices
default_mic = next((opt for opt in mic_options.keys() if "Microphone Array (Intel" in opt), list(mic_options.keys())[0])
default_speaker = next((opt for opt in loopback_options.keys() if "Speakers (Realtek(R) Audio) [Loopback]" in opt), list(loopback_options.keys())[0])

# Display device selection with defaults
mic_index = st.selectbox("Microphone", list(mic_options.keys()), index=list(mic_options.keys()).index(default_mic))
speaker_index = st.selectbox("Loopback Device", list(loopback_options.keys()), index=list(loopback_options.keys()).index(default_speaker))

# Initialize session state for recording control
if 'recording' not in st.session_state:
    st.session_state.recording = False
    st.session_state.stop_flag = type('Flag', (), {'value': False})()

if not st.session_state.recording and st.button("Start Recording"):
    st.session_state.recording = True
    st.session_state.stop_flag.value = False
    
    # Convert selection to actual indices
    mic_idx = mic_options[mic_index]
    speaker_idx = loopback_options[speaker_index]
    
    # Configure devices
    pa = pyaudio.PyAudio()
    mic_info = pa.get_device_info_by_index(mic_idx)
    speaker_info = pa.get_device_info_by_index(speaker_idx)
    pa.terminate()
    
    device_indices = [mic_idx, speaker_idx]
    device_names = [mic_info['name'], speaker_info['name']]
    channels_list = [min(mic_info['maxInputChannels'], 2), 2]
    rates = [int(mic_info['defaultSampleRate']), int(speaker_info['defaultSampleRate'])]
    
    # Set up recording parameters
    duration = 30 if duration_type == "30 seconds" else None
    stop_flag = st.session_state.stop_flag if duration is None else None
    
    # Record
    status_placeholder = st.empty()
    
    if duration:
        # For timed recording, show simple message
        status_placeholder.info("Recording for 30 seconds...")
    else:
        # For unlimited recording, show stop button
        status_placeholder.info("Recording in progress... (Press Stop when done)")
        st.button("Stop Recording", on_click=lambda: setattr(st.session_state.stop_flag, 'value', True))
    
    # Do the actual recording
    audio_files = record_audio(
        device_indices, device_names, channels_list, rates,
        duration=duration, stop_flag=stop_flag
    )
    
    # Clear status message
    status_placeholder.empty()
    
    st.session_state.recording = False
    if audio_files:
        st.success("Recording completed!")
        
        if st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                # Mix and transcribe
                mixed_audio = mix_wav_files(audio_files)
                
                # Save mixed audio
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                mixed_file = os.path.join("src/saved_audio", f"recording_{timestamp}_mixed.wav")
                
                with wave.open(mixed_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(48000)
                    mixed_audio_int = (mixed_audio * 32767).astype(np.int16)
                    wf.writeframes(mixed_audio_int.tobytes())
                
                # Transcribe and display
                result = transcribe_audio(mixed_file)
                st.text("Transcription Results:")
                st.write(result["text"])
                
                # Cleanup
                os.remove(mixed_file)
        
        # Show saved files
        st.text("Recorded files:")
        for file in audio_files:
            st.text(f"- {file}")
    else:
        st.error("Recording failed")
        
        st.session_state.recording = False
        if audio_files:
            st.success("Recording completed!")
            
            if st.button("Transcribe"):
                with st.spinner("Transcribing..."):
                    # Mix and transcribe
                    mixed_audio = mix_wav_files(audio_files)
                    
                    # Save mixed audio
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    mixed_file = os.path.join("src/saved_audio", f"recording_{timestamp}_mixed.wav")
                    
                    with wave.open(mixed_file, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(48000)
                        mixed_audio_int = (mixed_audio * 32767).astype(np.int16)
                        wf.writeframes(mixed_audio_int.tobytes())
                    
                    # Transcribe and display
                    result = transcribe_audio(mixed_file)
                    st.text("Transcription Results:")
                    st.write(result["text"])
                    
                    # Cleanup
                    os.remove(mixed_file)
            
            # Show saved files
            st.text("Recorded files:")
            for file in audio_files:
                st.text(f"- {file}")
        else:
            st.error("Recording failed")