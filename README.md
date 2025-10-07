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
poetry run streamlit run src/ui/streamlit_app.py
```

## How to Use

### Basic Version (`streamlit_app.py`)
1. Load a Whisper model (start with "base" for good balance)
2. Select your audio device (look for WASAPI loopback devices)
3. Set recording duration
4. Click "Start Recording"
25. Wait for recording to complete
6. Click "Transcribe Audio"
7. View and download the transcription

### Advanced Version (`streamlit_advanced.py`)
1. Load a Whisper model in the sidebar
2. Select audio source (WASAPI loopback for system audio)
3. Adjust transcription interval (how often to transcribe)
4. Click "Start Recording"
5. Watch live transcriptions appear automatically
6. Click "Stop Recording" when done
7. Download the full transcript

**Additional Features:**
- Upload pre-recorded audio files in the "Upload File" tab
- View all transcriptions in the "History" tab
- See detailed segments with timestamps

## Testing & Development

### Quick Test (Recommended)
Test the complete workflow - records 10 seconds and transcribes:
```bash
poetry run python tests/test_full_workflow.py
```

### List Available Audio Devices
```bash
poetry run python tests/test_audio_devices.py
```

### Test Teams Audio (Interactive)
```bash
poetry run python tests/test_teams_audio.py
```

### Individual Component Tests

**Test Recording Only (10 seconds)**:
```bash
poetry run python tests/test_record.py
```

**Test Transcription Only**:
```bash
poetry run python tests/test_transcribe.py
```

### Code Formatting
```bash
poetry run ruff check . --fix
poetry run ruff format .
```

### Documentation
See `tests/README.md` for detailed testing instructions.

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