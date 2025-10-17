# Speech to Text POC

Real-time audio capture and transcription for Teams/Zoom meetings using OpenAI Whisper.

## Features

- **System Audio Capture**: Capture audio from Teams, Zoom, or any system audio using WASAPI loopback
- **Real-time Transcription**: Live transcription using OpenAI Whisper
- **Multiple Whisper Models**: Choose from tiny, base, small, medium, or large models
- **Speaker Segments**: View transcription broken down by time segments
- **File Upload**: Transcribe pre-recorded audio files
- **Export Transcripts**: Download transcriptions as text files
- **Simple UI**: Built with Streamlit for easy use

## Quick Start (Windows)

### Prerequisites

1. **Python 3.13** (recommended) or Python 3.10+
2. **Poetry** for dependency management

### Installation

```bash
# Create conda environment (optional but recommended)
conda create -n spch2txt "python=3.13"
conda activate spch2txt

# Install Poetry
pip install poetry

# Install dependencies
poetry install --no-root
```

### Running the Application

```bash
poetry run streamlit run src/ui/app.py
```

## How to Use

pyannote model is gated - you need to:
- Accept pyannote/segmentation-3.0 user conditions, link: https://huggingface.co/pyannote/segmentation-3.0
- Accept pyannote/speaker-diarization-3.1 user conditions, link: https://huggingface.co/pyannote/speaker-diarization-3.1
- Accept pyannote/speaker-diarization-community-1 user conditions, link: https://huggingface.co/pyannote/speaker-diarization-community-1 
- Create access token at hf.co/settings/tokens (READ only should be enough)

### Via the stramlit website (`app.py`)
1. Load a Whisper model (start with "base" for good balance)
2. Select your audio device (look for WASAPI loopback devices)
3. Set recording duration
4. Click "Start Recording"
5. Wait for recording to complete
6. Click "Transcribe Audio"
7. View and download the transcription

**Additional Features:**
- Upload pre-recorded audio files in the "Upload File" tab
- View all transcriptions in the "History" tab
- See detailed segments with timestamps

## Testing & Development

### Quick Test (Recommended)
Test the complete workflow - records 10 seconds and transcribes automatically:
```bash
poetry run python tests/test_full_workflow.py
```

### Available Test Scripts

| Test Script | Purpose | Command |
|------------|---------|----------|
| `test_full_workflow.py` | Complete record + transcribe workflow | `poetry run python tests/test_full_workflow.py` |
| `test_audio_devices.py` | List all audio devices | `poetry run python tests/test_audio_devices.py` |
| `test_record.py` | Simple 10-second recording | `poetry run python tests/test_record.py` |
| `test_transcribe.py` | Transcribe existing WAV file | `poetry run python tests/test_transcribe.py` |
| `test_teams_audio.py` | Interactive Teams audio testing | `poetry run python tests/test_teams_audio.py` |

### Individual Test Details

#### List Available Audio Devices
```bash
poetry run python tests/test_audio_devices.py
```
Shows all available microphones, speakers, and loopback devices. Use this to find your device index.

#### Test Recording Only
```bash
poetry run python tests/test_record.py
```
Records 10 seconds of audio and saves to `out.wav`.

#### Test Transcription Only
```bash
poetry run python tests/test_transcribe.py
```
Transcribes `out.wav` (run `test_record.py` first to create the audio file).

#### Test Teams Audio (Interactive)
```bash
poetry run python tests/test_teams_audio.py
```
Interactive menu for testing different audio sources including Teams. Join a meeting first before running this test.

### Testing Tips
1. Start with `test_full_workflow.py` as it tests everything at once
2. Use `test_audio_devices.py` to find your device index
3. For Teams testing, join a meeting first, then run `test_teams_audio.py`
4. All test scripts save audio files to the project root directory

### Code Formatting
```bash
poetry run ruff check . --fix
poetry run ruff format .
```

## Capturing Teams/Zoom Audio

To capture audio from Teams or Zoom:

1. **Look for WASAPI Loopback Devices**: These appear in the device list with names like:
   - "Speakers (Loopback)"
   - "Headphones (Loopback)"
   - Device names containing "WASAPI"

2. **Select the Correct Device**: Choose the loopback device that corresponds to your audio output

3. **Start Your Meeting**: Begin your Teams/Zoom call

4. **Start Recording**: The app will capture all system audio including the meeting

**Note**: WASAPI loopback captures all system audio, not just Teams/Zoom. Make sure to mute other applications if needed.

## Whisper Model Comparison

| Model  | Size     | Speed     | Accuracy | Recommended For |
| ------ | -------- | --------- | -------- | --------------- |
| tiny   | ~39 MB   | Very Fast | Basic    | Quick tests     |
| base   | ~74 MB   | Fast      | Good     | General use     |
| small  | ~244 MB  | Medium    | Better   | Quality results |
| medium | ~769 MB  | Slow      | Great    | High accuracy   |
| large  | ~1550 MB | Very Slow | Best     | Maximum quality |

## Limitations

- **Windows Only**: Uses `pyaudiowpatch` for WASAPI support
- **No Docker**: Runs directly on Windows
- **System Audio**: Captures all system audio, not isolated to specific apps
- **Speaker Diarization**: Basic segment detection only (no advanced speaker identification)
- **Permissions**: May require admin rights depending on audio device configuration

## Troubleshooting

### No Audio Devices Found
- Ensure you're running on Windows
- Check that audio devices are properly configured in Windows Sound settings
- Try running as Administrator

### Recording Fails
- Verify the selected device supports input
- Check Windows audio permissions
- Ensure no other application is exclusively using the device

### Transcription is Slow
- Use a smaller Whisper model (tiny or base)
- Reduce recording duration
- Close other applications to free up resources

### Poor Transcription Quality
- Use a larger model (small or medium)
- Ensure audio quality is good (check volume levels)
- Reduce background noise
- Use a better microphone/audio source