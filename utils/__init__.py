import os

_dirname = os.path.dirname(__file__)

AUDIO_ROOT = os.path.join(_dirname, '../data/audio')
SPLIT_AUDIO_ROOT = os.path.join(_dirname, '../data/samples')
SAVE_DIR = os.path.join(_dirname, '../saves')

SAMPLE_LENGTH_MS = 5000
SAMPLE_RATE = 16000

def set_data_root_dir(rootdir):
    global AUDIO_ROOT, SPLIT_AUDIO_ROOT

    AUDIO_ROOT = os.path.join(rootdir, 'data/audio')
    SPLIT_AUDIO_ROOT = os.path.join(rootdir, 'data/samples')

