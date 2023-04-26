import os

from torch.utils.data import Dataset, DataLoader, random_split
import librosa
from transformers import AutoFeatureExtractor
import pandas as pd
from pydub import AudioSegment

from utils.audio_utils import AudioHelper, AudioTransformations
from utils import SAMPLE_LENGTH_MS, SPLIT_AUDIO_ROOT, AUDIO_ROOT, SAMPLE_RATE

class ProcessedAudioDS(Dataset):
    def __init__(self, df, pretrained=False, inference=False):
        self.df = df
        self.duration = SAMPLE_LENGTH_MS # Truncating all audio clips to SAMPLE_LENGTH_MS
        # self.sr = 44100
        self.sr = SAMPLE_RATE
        self.channel = 2
        self.shift_pct = 0.4
        self.pretrained = pretrained
        self.inference = inference
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("bookbot/distil-ast-audioset")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        audio_file = self.df.loc[idx, 'file']

        if not self.inference:
            class_id = self.df.loc[idx, 'labelId']
        
        # Use pretrained data preprocessor if we are using transfer learning
        if self.pretrained:
            audio, sr = librosa.load(audio_file, sr=self.sr)
            
            # Funny business to get the shape to work correctly
            pretrained_feat = self.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
            feat_shape = pretrained_feat['input_values'].shape
            pretrained_feat['input_values'] = pretrained_feat['input_values'].view(feat_shape[1], feat_shape[2])
            
            if not self.inference:
                return pretrained_feat, class_id
            return pretrained_feat

        aud = AudioHelper.open(audio_file)
        reaud = AudioHelper.resample(aud, self.sr)
        rechan = AudioHelper.rechannel(reaud, self.channel)
        dur_aud = AudioHelper.pad_trunc(rechan, self.duration)
        
        shift_aud = AudioTransformations.time_shift(dur_aud, self.shift_pct)
        sgram = AudioTransformations.spectrogram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioTransformations.spectrogram_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        if not self.inference:
            return aug_sgram, class_id
        return aug_sgram
    
class RawDS(Dataset):
    def __init__(self, df):
        self.df = df
        self.duration = SAMPLE_LENGTH_MS
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("bookbot/distil-ast-audioset")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        audio_file = self.df.loc[idx, 'file']
        audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
        return audio, self.df.loc[idx, 'labelId']
    
def get_train_val(dataset, train_split=0.8, batch_size=16):
    num_items = len(dataset)
    num_train = round(num_items * train_split)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(dataset, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl

def read_into_df():
    files = []
    labels = []
    for dirent in os.listdir(SPLIT_AUDIO_ROOT):
        dirpath = os.path.join(SPLIT_AUDIO_ROOT, dirent)
        if os.path.isdir(dirpath):
            for file in os.listdir(dirpath):
                if file == '.DS_Store':
                    continue
                path = os.path.join(dirpath, file)
                files.append(path)
                labels.append(dirent)

    dataframe = pd.DataFrame({'file': files, 'label': labels})
    dataframe['labelId'] = dataframe.groupby('label', sort=True).ngroup()
    return dataframe

def setup_dataset():
    # Split audio into SAMPLE_LENGTH_MS segments
    try:
        os.mkdir(SPLIT_AUDIO_ROOT)
    except:
        print("Dataset already setup!")
        return
    for dirent in os.listdir(AUDIO_ROOT):
        dirpath = os.path.join(AUDIO_ROOT, dirent)
        if os.path.isdir(dirpath):
            try:
                os.mkdir(os.path.join(SPLIT_AUDIO_ROOT, dirent))
            except:
                pass

            for file in os.listdir(dirpath):
                if file == '.DS_Store':
                    continue
                    
                path = os.path.join(dirpath, file)
                audio = AudioSegment.from_wav(path)
                
                total_ms = int(audio.duration_seconds) * 1000
                for start_ms in range(0, total_ms, SAMPLE_LENGTH_MS):
                    end_ms = start_ms + SAMPLE_LENGTH_MS
                    split = audio[start_ms:end_ms]
                    
                    split_filename = os.path.join(SPLIT_AUDIO_ROOT, dirent, f"{start_ms}_{file}")
                    split.export(split_filename, format="wav")