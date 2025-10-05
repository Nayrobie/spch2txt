# Windows Setup Guide

Quick setup guide for running the Speech to Text POC on Windows.

## Prerequisites

1. **Windows 10 or 11**
2. **Python 3.10+** (Python 3.13 recommended)
3. **Git** (optional, for cloning the repository)

## Installation Steps

### Step 1: Install Python

1. Download Python from [python.org](https://www.python.org/downloads/)
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```cmd
   python --version
   ```

### Step 2: Install Poetry

Open Command Prompt or PowerShell and run:
```cmd
pip install poetry
```

Verify installation:
```cmd
poetry --version
```

### Step 3: Install Project Dependencies

Navigate to the project directory and run:
```cmd
poetry install --no-root
```

This will install:
- `streamlit` - Web UI framework
- `pyaudiowpatch` - Windows audio capture (WASAPI)
- `openai-whisper` - Speech recognition
- `ffmpeg` - Audio processing
- Other dependencies

### Step 4: Verify Audio Devices

Test that your audio devices are detected:
```cmd
poetry run python src/audio_devices.py
```

You should see a list of audio devices including WASAPI loopback devices.

## Running the Application

### Method 1: Double-Click Launcher (Easiest)

Simply double-click `run_app.bat` in the project folder.

### Method 2: Command Line

Open Command Prompt in the project directory:

**Basic Version:**
```cmd
poetry run streamlit run src/streamlit_app.py
```

**Advanced Version (Recommended):**
```cmd
poetry run streamlit run src/streamlit_advanced.py
```

The application will open in your default web browser at `http://localhost:8501`

## Capturing Teams/Zoom Audio

### Finding the Right Audio Device

1. **Open the app** and look at the device list
2. **Look for devices with "Loopback"** in the name:
   - "Speakers (Loopback)"
   - "Headphones (Loopback)"
   - "Realtek Audio (Loopback)"
   - Any device with "WASAPI" in the name

3. **Select the loopback device** that matches your audio output

### Testing Audio Capture

1. **Play a YouTube video** or any audio
2. **Start recording** in the app
3. **Stop and transcribe** - you should see the audio content transcribed

### For Teams/Zoom Meetings

1. **Join a Teams/Zoom meeting**
2. **Select the WASAPI loopback device** in the app
3. **Start recording** before or during the meeting
4. **Stop when done** and download the transcript

**Important Notes:**
- WASAPI loopback captures **all system audio**, not just Teams/Zoom
- Mute other applications (music, videos) during recording
- Ensure your speakers/headphones are the default audio output
- The app captures what you hear, not what you say (unless you enable speaker output)

## Troubleshooting

### "No audio devices found"

**Solution:**
1. Check Windows Sound settings (Right-click speaker icon → Sounds)
2. Ensure audio devices are enabled
3. Try running Command Prompt as Administrator
4. Restart the application

### "Poetry not found"

**Solution:**
1. Reinstall Poetry: `pip install poetry`
2. Close and reopen Command Prompt
3. Check PATH environment variable includes Python Scripts folder

### "Module not found" errors

**Solution:**
```cmd
poetry install --no-root
```

### Recording produces no audio

**Solution:**
1. Verify the correct device is selected
2. Check Windows audio is not muted
3. Test with `poetry run python src/probe_record.py`
4. Ensure the device is not in use by another application

### Transcription is very slow

**Solution:**
1. Use a smaller Whisper model (tiny or base)
2. Close other applications
3. Reduce recording duration
4. Check CPU usage in Task Manager

### "Access Denied" or permission errors

**Solution:**
1. Run Command Prompt as Administrator
2. Check Windows audio permissions
3. Ensure antivirus is not blocking the application

## Performance Tips

### For Best Performance:
- **Use "base" or "small" Whisper model** for real-time transcription
- **Close unnecessary applications** to free up CPU/RAM
- **Use shorter transcription intervals** (3-5 seconds)
- **Ensure good audio quality** from the source

### For Best Accuracy:
- **Use "small" or "medium" Whisper model**
- **Ensure clear audio** with minimal background noise
- **Use longer transcription intervals** (10-15 seconds)
- **Adjust microphone/speaker volume** appropriately

## First-Time User Guide

### Quick Test (5 minutes):

1. **Open Command Prompt** in the project folder
2. **Run**: `run_app.bat`
3. **Choose option 2** (Advanced version)
4. **Load the "base" model** in the sidebar
5. **Play a YouTube video** with clear speech
6. **Select a loopback device** from the dropdown
7. **Click "Start Recording"**
8. **Watch the transcription** appear in real-time
9. **Click "Stop Recording"** after 30 seconds
10. **Download the transcript**

### For Teams/Zoom Meeting:

1. **Start the app** before your meeting
2. **Load the Whisper model** (do this once)
3. **Join your Teams/Zoom meeting**
4. **Select the loopback device** matching your audio output
5. **Click "Start Recording"** when the meeting begins
6. **Let it run** throughout the meeting
7. **Click "Stop Recording"** when done
8. **Download the full transcript**

## Getting Help

If you encounter issues:

1. **Check the main README.md** for detailed documentation
2. **Run the test scripts** to isolate the problem:
   - `poetry run python src/audio_devices.py` - List devices
   - `poetry run python src/probe_record.py` - Test recording
   - `poetry run python src/probe_transcribe.py` - Test transcription
3. **Check Windows Event Viewer** for system errors
4. **Verify all dependencies** are installed: `poetry install --no-root`

## System Requirements

**Minimum:**
- Windows 10/11
- 4 GB RAM
- 2 GHz dual-core processor
- 2 GB free disk space

**Recommended:**
- Windows 11
- 8 GB+ RAM
- 4+ core processor
- 5 GB free disk space
- SSD for faster model loading

## Privacy & Security

- **All processing is local** - no data sent to external servers
- **Audio is not saved** unless you explicitly save it
- **Transcripts are temporary** - download them before closing the app
- **WASAPI loopback** captures system audio only, not microphone input (unless configured)

## Checklist

Before your first meeting transcription:

- [ ] Python installed and in PATH
- [ ] Poetry installed
- [ ] Dependencies installed (`poetry install --no-root`)
- [ ] Audio devices detected (`poetry run python src/audio_devices.py`)
- [ ] Test recording successful (`poetry run python src/probe_record.py`)
- [ ] Whisper model loaded in the app
- [ ] WASAPI loopback device identified
- [ ] Test transcription with YouTube video successful

You're ready to transcribe your first meeting! 
