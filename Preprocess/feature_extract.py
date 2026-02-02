import os
import librosa
import numpy as np
import pandas as pd

def extract_fbank_to_csv(wav_path, output_csv_path, n_mels=80, frame_length=0.025, frame_shift=0.010):
    """Extract Fbank features from a single audio file and save it as a CSV file."""
    try:
        audio, sr = librosa.load(wav_path, sr=None)
        n_fft = int(frame_length * sr)
        hop_length = int(frame_shift * sr)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        output_dir = os.path.dirname(output_csv_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savetxt(output_csv_path, log_mel_spectrogram, delimiter=',')
        
        return True, log_mel_spectrogram.shape
        
    except Exception as e:
        print(f"An error occurred while processing file {wav_path}: {e}")
        return False, None


def extract_mfcc_to_csv(wav_path, output_csv_path, n_mfcc=13, frame_length=0.025, frame_shift=0.010):
    """Extract MFCC, Delta, and Delta-Delta features from a single audio file and save it as a CSV file."""
    try:
        audio, sr = librosa.load(wav_path, sr=None)
        n_fft = int(frame_length * sr)
        hop_length = int(frame_shift * sr)
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=n_mfcc, 
            hop_length=hop_length, 
            n_fft=n_fft
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc_delta)
        features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savetxt(output_csv_path, features, delimiter=',')
        
        return True, features.shape
        
    except Exception as e:
        print(f"An error occurred while processing file {wav_path}: {e}")
        return False, None


def extract_egemaps_to_csv(wav_path, output_csv_path, exe_path, config_path):
    """Use openSMILE to extract eGeMAPS features from a single audio file and save it as a CSV file."""
    try:
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cmd = f'"{exe_path}" -C "{config_path}" -I "{wav_path}" -csvoutput "{output_csv_path}"'
        result = os.system(cmd)
        if result == 0:
            print(f"Extraction successful, saved to: {output_csv_path}")
            return True
        else:
            print(f"Extraction failed, error code: {result}")
            return False

    except Exception as e:
        print(f"An error occurred while processing file {wav_path}: {e}")
        return False

