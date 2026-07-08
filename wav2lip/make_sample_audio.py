"""Generate a short demo speech clip with Piper (offline TTS) -> ../assets/sample_audio.wav

Requires a Piper voice at ../models/piper/en_US-lessac-medium.onnx (+ .json).
Download once with:
  python -c "import urllib.request as u; base='https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/'; [u.urlretrieve(base+f, '../models/piper/'+f) for f in ['en_US-lessac-medium.onnx','en_US-lessac-medium.onnx.json']]"
"""
import os
import wave

from piper import PiperVoice

HERE = os.path.dirname(__file__)
VOICE = os.path.abspath(os.path.join(HERE, "..", "models", "piper", "en_US-lessac-medium.onnx"))
OUT = os.path.abspath(os.path.join(HERE, "..", "assets", "sample_audio.wav"))

TEXT = (
    "Hello! This is a local lip sync proof of concept, "
    "running entirely on the C P U with an open source model."
)

if __name__ == "__main__":
    voice = PiperVoice.load(VOICE)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with wave.open(OUT, "wb") as wf:
        voice.synthesize_wav(TEXT, wf)
    print("wrote", OUT)
