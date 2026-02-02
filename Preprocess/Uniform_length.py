import os
import librosa
import numpy as np
import soundfile as sf

def resample_single_audio(input_path, output_path, target_sr=16000):
    """Resample and save a single audio file"""
    try:
        y, sr = librosa.load(input_path, sr=None)
        resampled_data = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        sf.write(output_path, resampled_data, target_sr)
        return True
    
    except Exception as e:
        print(f"An error occurred while processing file {input_path}: {e}")
        return False

def process_audio_fixed_length(input_path, output_path, segment_duration=23):
    """Perform fixed-length processing (padding or truncation) on a single audio file."""
    try:
        y, sr = librosa.load(input_path, sr=None)
        target_length = int(segment_duration * sr)
        current_length = len(y)
        
        if current_length < target_length:
            padding_length = target_length - current_length
            processed_audio = np.pad(y, (0, padding_length), 'constant')
        else:
            processed_audio = y[:target_length]
            
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        sf.write(output_path, processed_audio, sr)
        return True
        
    except Exception as e:
        print(f"An error occurred while processing file {input_path}: {e}")
        return False
