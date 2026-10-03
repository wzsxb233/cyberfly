"""Resolve only a verified chunk belonging to the displayed numerical turn."""
import hashlib
import io
from pathlib import Path
import re
import wave

ROOT = Path(__file__).resolve().parents[1]


def read_stream_audio(progress, event_id, index, sha256):
    if not re.fullmatch(r'[a-f0-9]{32}', event_id) or not re.fullmatch(r'[a-f0-9]{64}', sha256):
        raise ValueError('Invalid stream identity')
    if (not progress or progress.get('event_id') != event_id or
            not progress.get('numerical_feedback_verified') or progress.get('error')):
        raise ValueError('This numerical turn has no verified playable stream')
    rows = progress.get('audio_chunks', [])
    if type(index) is not int or not 0 <= index < len(rows):
        raise ValueError('Native audio chunk is not ready')
    row = rows[index]
    if row['sequence'] != index or row['sha256'] != sha256:
        raise ValueError('Native chunk sequence or identity mismatch')
    path = Path(row.get('path', '')).resolve()
    if (not path.is_relative_to(ROOT / 'artifacts/vllm_omni_migration/flows') or
            path.name != f'chunk-{index:04d}.wav' or path.parent.name != 'chunks' or
            path.parent.parent.name != 'response' or path.parent.parent.parent.name != event_id or
            not path.is_file() or path.stat().st_size > 6_000_000):
        raise ValueError('Native chunk is outside the original turn')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError('Native chunk bytes changed')
    with wave.open(io.BytesIO(raw), 'rb') as wav:
        if ((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (24000, 1, 2) or
                wav.getnframes() != row['samples'] or len(wav.readframes(wav.getnframes())) != row['samples'] * 2):
            raise ValueError('Native chunk PCM identity changed')
    return raw
