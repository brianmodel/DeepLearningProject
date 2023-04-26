import os

_dirname = os.path.dirname(__file__)

SAMPLE_LENGTH_MS = 5000
AUDIO_ROOT = os.path.join(_dirname, '../data/audio')
SPLIT_AUDIO_ROOT = os.path.join(_dirname, '../data/samples')
SAMPLE_RATE = 16000
SAVE_DIR = os.path.join(_dirname, '../saves')
