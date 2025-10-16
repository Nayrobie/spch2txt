"""
poetry run python tests/test_diarization.py
"""

from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
INPUT_AUDIO = "src/saved_audio/recording_20251015_183733_dev2_Headphones__WH-1000XM6___Loopback_.wav"

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGINGFACE_TOKEN
)

# perform speaker diarization locally
output = pipeline(INPUT_AUDIO)

for segment, _, speaker in output.itertracks(yield_label=True):
    print(f"{speaker} speaks between t={segment.start:.2f}s and t={segment.end:.2f}s")