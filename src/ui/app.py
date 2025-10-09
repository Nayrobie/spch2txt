"""
Simple Streamlit UI for speech-to-text recording and transcription.
poetry run streamlit run src/ui/app.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.audio import (
    AudioCapture,
    AudioTranscriber,
    categorize_devices,
    mix_wav_files,
    save_audio_array,
    get_audio_level
)


OUTPUT_DIR = "src/saved_audio"


def initialize_session_state():
    """Initialize session state variables."""
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    if 'audio_files' not in st.session_state:
        st.session_state.audio_files = []


def main():
    """Main Streamlit application."""
    st.title("Speech to Text - AB POC")
    st.markdown("Record audio from multiple devices and transcribe")

    initialize_session_state()

    capture = AudioCapture(frames_per_buffer=1024)

    st.header("Device Selection")

    devices = capture.list_devices()
    categorized = categorize_devices(devices)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Microphone")
        input_options = {
            f"[{dev['index']}] {dev['name']}": dev['index']
            for dev in categorized['input']
        }
        if input_options:
            mic_selection = st.selectbox(
                "Select microphone device:",
                options=list(input_options.keys()),
                key="mic_select"
            )
            mic_index = input_options[mic_selection]
        else:
            st.warning("No input devices found")
            mic_index = None

    with col2:
        st.subheader("System Audio (Loopback)")
        loopback_options = {
            f"[{dev['index']}] {dev['name']}": dev['index']
            for dev in categorized['loopback']
        }
        if loopback_options:
            loopback_selection = st.selectbox(
                "Select loopback device:",
                options=list(loopback_options.keys()),
                key="loopback_select"
            )
            loopback_index = loopback_options[loopback_selection]
        else:
            st.warning("No loopback devices found")
            loopback_index = None

    st.markdown("---")

    st.header("Recording")

    duration = st.slider(
        "Recording duration (seconds):",
        min_value=5,
        max_value=60,
        value=15,
        step=5
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        if not st.session_state.recording:
            if st.button("🔴 Start Recording", type="primary", use_container_width=True):
                if mic_index is not None and loopback_index is not None:
                    st.session_state.recording = True
                    st.session_state.transcript = ""
                    st.rerun()
                else:
                    st.error("Please select both microphone and loopback devices")
        else:
            if st.button("⏹️ Stop & Reset", type="secondary", use_container_width=True):
                st.session_state.recording = False
                st.session_state.transcript = ""
                st.session_state.audio_files = []
                st.rerun()

    with col2:
        if st.session_state.recording:
            st.info("🎙️ Recording in progress... Check terminal for details")

    if st.session_state.recording and mic_index is not None and loopback_index is not None:
        with st.spinner("Recording audio..."):
            print("\n" + "="*80)
            print("RECORDING STARTED")
            print("="*80)

            mic_info = devices[mic_index]
            loopback_info = devices[loopback_index]

            device_indices = [mic_index, loopback_index]
            device_names = [mic_info['name'], loopback_info['name']]
            channels_list = [
                min(mic_info['maxInputChannels'], 2),
                min(loopback_info['maxInputChannels'], 2)
            ]
            rates = [
                int(mic_info['defaultSampleRate']),
                int(loopback_info['defaultSampleRate'])
            ]

            print(f"Device 1: {mic_info['name']}")
            print(f"Settings: {rates[0]}Hz, {channels_list[0]} channel(s)")
            print(f"Device 2: {loopback_info['name']}")
            print(f"Settings: {rates[1]}Hz, {channels_list[1]} channel(s)")
            print(f"Duration: {duration} seconds")
            print("-"*80)

            try:
                audio_files = capture.record_multi_device(
                    device_indices=device_indices,
                    device_names=device_names,
                    channels_list=channels_list,
                    rates=rates,
                    duration=duration,
                    output_dir=OUTPUT_DIR
                )

                st.session_state.audio_files = audio_files

                print("\n✓ Recordings saved:")
                for file in audio_files:
                    print(f"  - {file}")

            except Exception as e:
                st.error(f"Recording failed: {e}")
                print(f"\n❌ Error: {e}")
                st.session_state.recording = False
                st.rerun()

        with st.spinner("Mixing and transcribing audio..."):
            print("\n" + "="*80)
            print("MIXING AND TRANSCRIBING AUDIO")
            print("="*80)

            try:
                print("Mixing audio files...")
                mixed_audio = mix_wav_files(st.session_state.audio_files)
                print(f"✓ Mixed audio length: {len(mixed_audio)/48000:.1f} seconds")

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                mixed_file = os.path.join(
                    OUTPUT_DIR, f"recording_{timestamp}_mixed.wav"
                )

                print(f"Saving mixed audio to: {mixed_file}")
                save_audio_array(mixed_audio, mixed_file, rate=48000)
                print("✓ Mixed audio saved")

                audio_level = get_audio_level(mixed_audio)
                print(f"Audio level: {audio_level:.6f}")

                if audio_level < 0.001:
                    print("⚠ Warning: Audio is very quiet or silent")
                    st.warning("Audio is very quiet or silent")

                print("\nLoading Whisper model (base)...")
                transcriber = AudioTranscriber(model_name="base")
                transcriber.load_model()
                print("✓ Model loaded")

                print("Transcribing...")
                result = transcriber.transcribe(mixed_file, verbose=False)

                st.session_state.transcript = result["text"].strip()

                print("\nTRANSCRIPTION RESULTS")
                print("-" * 50)
                if st.session_state.transcript:
                    print(st.session_state.transcript)
                else:
                    print("(No speech detected)")
                print("-" * 50)

                if "language" in result:
                    print(f"\nDetected Language: {result['language']}")

                print("\nCleaning up temporary files...")
                os.remove(mixed_file)
                print("✓ Cleanup complete")
                print("="*80)

            except Exception as e:
                st.error(f"Transcription failed: {e}")
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

        st.session_state.recording = False
        st.rerun()

    st.markdown("---")

    st.header("Transcript")

    if st.session_state.transcript:
        st.success("✅ Transcription complete!")
        st.text_area(
            "Transcribed text:",
            value=st.session_state.transcript,
            height=200,
            key="transcript_display"
        )

        if st.button("📋 Copy to Clipboard"):
            st.code(st.session_state.transcript, language=None)
            st.info("Select and copy the text above")

    else:
        st.info("No transcript yet. Start recording to generate a transcript.")

    st.markdown("---")
    st.caption("💡 Tip: Check the terminal for detailed progress and debug information")


if __name__ == "__main__":
    main()
