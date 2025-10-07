# Test Scripts

Quick test scripts for validating audio capture and transcription functionality.

## Quick Start

### Test Everything (Recommended)
```bash
poetry run python tests/test_full_workflow.py
```
Records 10 seconds and transcribes automatically.

---

## Individual Tests

### List Audio Devices
```bash
poetry run python tests/test_audio_devices.py
```
Shows all available microphones, speakers, and loopback devices.

### Test Recording
```bash
poetry run python tests/test_record.py
```
Records 10 seconds to `out.wav`.

### Test Transcription
```bash
poetry run python tests/test_transcribe.py
```
Transcribes `out.wav` (run `test_record.py` first).

### Test Teams Audio
```bash
poetry run python tests/test_teams_audio.py
```
Interactive menu for testing different audio sources including Teams.

---

## Test Files

| File | Purpose |
|------|---------|
| `test_audio_devices.py` | List all audio devices |
| `test_record.py` | Simple 10-second recording |
| `test_transcribe.py` | Transcribe existing WAV file |
| `test_full_workflow.py` | Complete record + transcribe workflow |
| `test_teams_audio.py` | Interactive Teams audio testing |

---

## Tips

1. **Start with `test_full_workflow.py`** - it tests everything at once
2. **Use `test_audio_devices.py`** to find your device index
3. **For Teams testing**, join a meeting first, then run `test_teams_audio.py`
4. All tests save audio files to the project root directory
