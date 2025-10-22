# Speech to Text POC

A meeting capture solution that records both system audio (from Teams, Zoom, Youtube etc.) and microphone input simultaneously. The application uses OpenAI Whisper for speech-to-text transcription and PyAnnote for speaker diarization, enabling detailed meeting transcripts with speaker identification and timestamps. The transcript is then used to generate a meeting minutes using OpenAI API with GPT-4.

## Table of Contents
- [Speech to Text POC](#speech-to-text-poc)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Configure Speaker Diarization](#configure-speaker-diarization)
    - [Configure Meeting Summarization](#configure-meeting-summarization)
    - [Running the Application](#running-the-application)
  - [Speech to Text Workflow](#speech-to-text-workflow)
    - [How It Works](#how-it-works)
  - [Testing \& Development](#testing--development)
    - [Available Test Scripts](#available-test-scripts)
    - [Code Formatting](#code-formatting)
  - [Whisper Model Comparison](#whisper-model-comparison)
  - [Limitations](#limitations)
  - [Development Challenges](#development-challenges)
    - [Audio Device Detection](#audio-device-detection)
    - [ASR Methodology](#asr-methodology)
    - [PyAnnote Dependencies](#pyannote-dependencies)
    - [Hallucination Filtering](#hallucination-filtering)
    - [Combining Transcription and Diarization](#combining-transcription-and-diarization)
    - [Audio Corruption when adding Diarization](#audio-corruption-when-adding-diarization)
  - [Future Improvements](#future-improvements)
    - [Core: spch2txt](#core-spch2txt)
    - [UI/UX Enhancements](#uiux-enhancements)
    - [Post-Processing with LLM](#post-processing-with-llm)
    - [Packaging and Deployment](#packaging-and-deployment)
    - [Audio and Timestamp Logic](#audio-and-timestamp-logic)
    - [Transcription Quality](#transcription-quality)

## Features

- **System Audio Recording**: Capture audio from Teams, Zoom, Youtube, or any system audio playing on your device using WASAPI loopback
- **Microphone Recording**: Simultaneous microphone and system audio capture
- **Transcription**: Transcription from speech to text using OpenAI Whisper
- **Speaker Diarization**: Identify different speakers using pyannote.audio to view meeting transcription with timestamps and speaker labels
- **Meeting Summarization**: Automatic generation of meeting minutes using OpenAI GPT-4
- **Whisper Model**: The Whisper AI model used is the base one (all model config: tiny, base, small, medium, or large)
- **Export Transcripts**: Save transcriptions as JSON files with metadata and summaries as text files
- **Simple UI**: Built with Streamlit UI for easy use

## Quick Start
Warning: This app is only meant for Windows operating systems.

### Prerequisites

1. **Python 3.13** (recommended)
2. **Poetry** for dependency management
3. **Conda** for environment management

### Installation

```bash
# Create conda environment
conda create -n spch2txt "python=3.13"
conda activate spch2txt

# Install Poetry
pip install poetry

# Install dependencies
poetry install --no-root
```

### Configure Speaker Diarization
   
   To enable speaker identification, set up pyannote.audio:
   - Accept user conditions for pyannote models:
     - https://huggingface.co/pyannote/segmentation-3.0
     - https://huggingface.co/pyannote/speaker-diarization-3.1
   - Create a Hugging Face access token (READ only) at https://hf.co/settings/tokens
   - Add to your `.env` file:
     ```
     HUGGINGFACE_TOKEN=your_token_here
     ```

### Configure Meeting Summarization

   To enable automatic meeting minutes generation:
   - Create an OpenAI API key at https://platform.openai.com/api-keys
   - Add to your `.env` file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - The app uses GPT-4 by default for cost-effective summarization

### Running the Application

1. **Start the Streamlit UI**
   ```bash
   poetry run streamlit run src/ui/app.py
   ```

2. **User Mode (Default)**
   - Automatically detects default microphone and all loopback devices
   - Click "Start Recording" to begin unlimited recording
   - Click "Stop Recording" when finished
   - Transcription and summarization start automatically after recording stops
   - View timestamped transcripts with speaker labels and meeting summaries

3. **Dev Mode**
   - Enable "Dev Mode" in the sidebar
   - Manually select specific microphone and loopback devices
   - Set fixed recording duration
   - Useful for testing specific device configurations

## Speech to Text Workflow

<img src="src/image/spch2txt-diagram-2025-10-17-155440.png" alt="Speech-to-Text Flow" width="60%">

### How It Works

1. **Audio Capture**
   - Captures audio from microphone and system loopback devices simultaneously
   - Uses WASAPI loopback to record system audio (Teams, Zoom, etc.)
   - Records each device to separate WAV files

2. **Transcription**
   - Loads Whisper model (cached at initiation for performance)
   - Transcribes each audio stream separately with timestamps
   - Filters out Whisper hallucinations (like false transcriptions from silence)

3. **Speaker Diarization**
   - Analyzes audio to identify different speakers
   - Assigns speaker labels to transcription segments
   - Combines diarization with transcription using timestamp overlap

4. **Meeting Summarization**
   - Generates meeting minutes using OpenAI GPT-4
   - Creates structured summaries organized by topic
   - Highlights key decisions and action items
   - Uses lazy loading to avoid unnecessary API calls

5. **Output**
   - Combines transcripts from all devices in chronological order
   - Formats output with timestamps and speaker labels
   - Saves transcript to JSON file with metadata in `src/saved_transcripts/`
   - Saves summary to text file in `src/saved_summary/`
   - Example transcript output:
     ```
     [00:05] [Microphone SPEAKER_00]: Hello everyone
     [00:08] [System Audio SPEAKER_01]: Hi, thanks for joining
     ```
   - Example summary output:
     ```
     **Project Update**
     - Beta release scheduled for next Monday
     - Data pipeline refactor completed, reducing latency by 30%
     
     **Action Items**
     - Finalize event schema by end of week
     - Demo drift dashboard during stakeholder sync
     ```

## Testing & Development

### Available Test Scripts

| Test Script | Purpose | Command |
|------------|---------|----------|
| `test_full_workflow.py` | Complete record + transcribe workflow | `poetry run python tests/test_full_workflow.py` |
| `test_audio_devices.py` | List all audio devices | `poetry run python tests/test_audio_devices.py` |
| `test_record.py` | Simple 10-second recording | `poetry run python tests/test_record.py` |
| `test_transcribe.py` | Transcribe existing WAV file | `poetry run python tests/test_transcribe.py` |
| `test_teams_audio.py` | Interactive Teams audio testing | `poetry run python tests/test_teams_audio.py` |
| `test_summarizer.py` | Summarize from a test transcript JSON | `poetry run python tests/test_summarizer.py` |


### Code Formatting
```bash
poetry run ruff check . --fix
poetry run ruff format .
```

## Whisper Model Comparison

| Model  | Size     | Speed     | Accuracy | Recommended For |
| ------ | -------- | --------- | -------- | --------------- |
| tiny   | ~39 MB   | Very Fast | Basic    | Quick tests     |
| base   | ~74 MB   | Fast      | Good     | General use     |
| small  | ~244 MB  | Medium    | Better   | Quality results |
| medium | ~769 MB  | Slow      | Great    | High accuracy   |
| large  | ~1550 MB | Very Slow | Best     | Maximum quality |

## Limitations

- ⚠️ **Windows Only** ⚠️: Uses `pyaudiowpatch` for WASAPI support
- **No Docker**: Runs directly on Windows
- **System Audio**: Captures all system audio, not isolated to specific apps
- **Permissions**: May require admin rights depending on audio device configuration

## Development Challenges

### Audio Device Detection
- Windows provides a long list of audio devices (microphones, loopback, outputs)
- Challenge: Identifying which loopback device is actually in use
- Solution: Detect all loopback devices and filter by checking for actual audio data during recording

### ASR Methodology
- Understanding different Automatic Speech Recognition approaches
- Learning Whisper's capabilities and limitations
- Balancing model size vs. accuracy vs. speed

### PyAnnote Dependencies
- Complex dependency chain with unclear documentation
- Gated models requiring Hugging Face authentication
- Solution: Implemented lazy loading to avoid conflicts

### Hallucination Filtering
- Microphone transcribes false text ("1.5%", "...", etc.) during silence
- Caused by low audio levels triggering Whisper's pattern recognition
- Solution: Filter segments based on `no_speech_prob` threshold and known hallucination patterns
- Example filtered output:
  ```
  ⚠ Filtered: '1.5%' (no_speech_prob=0.58)
  ⚠ Filtered 14 hallucination(s)
  ✓ Kept 0 valid segment(s)
  ```

### Combining Transcription and Diarization
- Challenge: Merging speaker labels from diarization with text from transcription
- Both systems produce time-based segments with different boundaries
- Solution: Match segments using timestamp interval overlap, assigning speakers based on maximum overlap duration

### Audio Corruption when adding Diarization
- Initial implementation caused corrupted/cut WAV files during recording
- Symptoms: Audio quality degraded, making transcripts unusable
- **Root cause**: `from pyannote.audio import Pipeline` at module import time loaded torchaudio and set a global audio backend that interfered with PyAudio's recording
- **Solution**: Moved all pyannote imports inside methods (`_load_pipeline()` and `diarize()`), ensuring they only load after recording completes, eliminating the conflict

## Future Improvements

### Core: spch2txt
- Improve signal processing and merging logic when multiple speakers overlap  
- Analyze impact of audio volume on transcription quality  
- Optimize performance (CPU usage, latency, I/O)  
- Add unit tests and basic security checks
- Add an evaluation system and log every evaluation to compare the results after optimizing and improving the app

### UI/UX Enhancements
- Add pause/resume button for recording to handle meeting breaks
- Implement visual indicators for recording status (recording, paused, processing)
- Add progress bars for transcription and summarization steps

### Post-Processing with LLM
- Implement structured prompt templates for meeting minutes with customizable sections:
  - Meeting metadata (date, attendees, duration)
  - Executive summary
  - Discussion topics with timestamps
  - Decisions made
  - Action items with assigned owners
  - Follow-up items and next steps
- Add prompt engineering options for different meeting types (standup, planning, retrospective, etc.)
- Allow users to customize summary format and detail level
- Allow flexible endpoint selection (OpenAI API or on-prem VLLM)

### Packaging and Deployment
- Package the entire application and dependencies in a portable ZIP  

### Audio and Timestamp Logic
- Fix: audio devices are detected only at application startup (connecting another device after starting the app won't show in the loopback device list)
- Adjust timestamp format to show ranges (e.g., `[00:00 → 00:23] [Speaker 01] [System Audio]`)  

### Transcription Quality
- Test WhisperX instead of Whisper for improved alignment and reduced computation time  
- Experiment with larger Whisper models (from `base` to `medium`)
- Apply Whisper optimization parameters for better accuracy and speed
- Allow users to optionally specify the recording language for higher precision in single-language sessions