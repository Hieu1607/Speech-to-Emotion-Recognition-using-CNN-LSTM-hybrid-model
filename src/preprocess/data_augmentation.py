"""Function to augment data"""

import random

import librosa
import numpy as np

random.seed(42)


### Data Augmentation based on Time Domain Techniques ###
# NOISE
def noise(data, noise_factor=0.035):
    """
    Add random white noise to an audio signal
    Args:
        data (numpy array): Original audio signal
        noise_factor (float): Factor to scale the noise
    Returns:
        numpy array: Augmented audio signal with added noise
    """
    noise_amp = noise_factor * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


# STRETCH
def stretch(data, rate=0.8):
    """
    Stretch an audio signal by a fixed rate / Change the speed of an audio signal
    Args:
        data (numpy array): Original audio signal
        rate (float): Rate to stretch the audio signal
    Returns:
        numpy array: Augmented audio signal after stretching
    """
    return librosa.effects.time_stretch(data, rate=rate)


# SHIFT
def shift(data):
    """
    Shift an audio signal by a random number of samples
    Args:
        data (numpy array): Original audio signal
    Returns:
        numpy array: Augmented audio signal after shifting
    """
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(
        data, shift_range
    )  # When moved, the samples pushed out of one end will roll back out the other end (like rolling a tape).


# PITCH
def pitch(data, sampling_rate, pitch_factor=0.7):
    """
    Change the pitch of an audio signal / This simulated a speaker with a higher or lower voice(e.g., male vs female)
    Args:
        data (numpy array): Original audio signal
        sampling_rate (int): Sampling rate of the audio signal
        pitch_factor (float): Factor to change the pitch
    Returns:
        numpy array: Augmented audio signal after changing pitch
    """
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


# COMBINE
def augment_data(data, sampling_rate, rate=0.8, noise_factor=0.035, pitch_factor=0.7):
    """
    Apply all augmentation techniques to an audio signal
    Args:
        data (numpy array): Original audio signal
        sampling_rate (int): Sampling rate of the audio signal
    Returns:
        numpy array: Augmented audio signal after applying all techniques
    """
    augmented_data = noise(data, noise_factor=noise_factor)
    augmented_data = stretch(augmented_data, rate=rate)
    augmented_data = shift(augmented_data)
    augmented_data = pitch(augmented_data, sampling_rate, pitch_factor=pitch_factor)
    return augmented_data


### Data Augmentation based on Frequency Domain Techniques ###


def frequency_masking(file_path: str, n_mels: int = 128, max_mask_width: int = 20):
    """
    Thực hiện Frequency Masking trên Mel Spectrogram của file audio.

    Args:
        file_path (str): Đường dẫn đến file âm thanh (.wav, .mp3, v.v.).
        n_mels (int): Số lượng Mel bands cho Spectrogram.
        max_mask_width (int): Chiều rộng tối đa của dải tần số bị che (tính bằng Mel bins).

    Returns:
        tuple: (Mel Spectrogram đã mask, Tần số lấy mẫu, n_mels, Original audio, Reconstructed audio từ masked spectrogram)
    """
    try:
        # 1. Load file audio
        data, sr = librosa.load(file_path)

        # 2. Create Mel Spectrogram (and convert to Log-scale for easier processing)
        mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Keep original spectrogram for reconstruction
        original_mel_spec = mel_spec.copy()
        masked_mel_spec = mel_spec.copy()

        # 3. Perform Frequency Masking

        # Ensure max_mask_width doesn't exceed available mel bands
        max_mask_width = min(max_mask_width, n_mels)

        # Randomly choose mask width (from 0 to max_mask_width)
        F = np.random.randint(low=0, high=max_mask_width + 1)

        # Randomly choose starting point of frequency band (f)
        if F > 0 and F < n_mels:
            f = np.random.randint(low=0, high=n_mels - F + 1)

            # Mask the frequency band [f, f + F) with minimum value
            log_mel_spec[f : f + F, :] = log_mel_spec.min()
            # Also mask the linear mel spectrogram for audio reconstruction
            masked_mel_spec[f : f + F, :] = 0

        # 4. Reconstruct audio from masked spectrogram
        try:
            # Convert back to audio using Griffin-Lim algorithm
            reconstructed_audio = librosa.feature.inverse.mel_to_audio(
                masked_mel_spec, sr=sr, n_fft=2048, hop_length=512
            )
        except:
            # Fallback method if mel_to_audio fails
            reconstructed_audio = data  # Return original if reconstruction fails

        return log_mel_spec, sr, n_mels, data, reconstructed_audio

    except Exception as e:
        print(f"Error in frequency_masking: {e}")
        return None, None, None, None, None


def time_masking(file_path: str, n_mels: int = 128, max_mask_time_frames: int = 50):
    """
    Thực hiện Time Masking trên Mel Spectrogram của file audio.

    Args:
        file_path (str): Đường dẫn đến file âm thanh (.wav, .mp3, v.v.).
        n_mels (int): Số lượng Mel bands cho Spectrogram.
        max_mask_time_frames (int): Số khung thời gian tối đa bị che.

    Returns:
        tuple: (Mel Spectrogram đã mask, Tần số lấy mẫu, n_mels, Original audio, Reconstructed audio từ masked spectrogram)
    """
    try:
        # 1. Load audio signal
        data, sr = librosa.load(file_path)

        # 2. Create Mel Spectrogram (and convert to Log-scale)
        mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Keep original spectrogram for reconstruction
        original_mel_spec = mel_spec.copy()
        masked_mel_spec = mel_spec.copy()

        # Spectrogram size: (n_mels, n_frames)
        n_frames = log_mel_spec.shape[1]

        # 3. Perform Time Masking

        # Ensure max_mask_time_frames doesn't exceed available time frames
        max_mask_time_frames = min(max_mask_time_frames, n_frames)

        # Randomly choose mask width (from 0 to max_mask_time_frames)
        T = np.random.randint(low=0, high=max_mask_time_frames + 1)

        # Randomly choose starting point of time band (t)
        if T > 0 and T < n_frames:
            t = np.random.randint(low=0, high=n_frames - T + 1)

            # Mask the time band [t, t + T) with minimum value
            log_mel_spec[:, t : t + T] = log_mel_spec.min()
            # Also mask the linear mel spectrogram for audio reconstruction
            masked_mel_spec[:, t : t + T] = 0

        # 4. Reconstruct audio from masked spectrogram
        try:
            # Convert back to audio using Griffin-Lim algorithm
            reconstructed_audio = librosa.feature.inverse.mel_to_audio(
                masked_mel_spec, sr=sr, n_fft=2048, hop_length=512
            )
        except:
            # Fallback method if mel_to_audio fails
            reconstructed_audio = data  # Return original if reconstruction fails

        return log_mel_spec, sr, n_mels, data, reconstructed_audio

    except Exception as e:
        print(f"Error in time_masking: {e}")
        return None, None, None, None, None


# Example usage:
# Replace 'path_to_file.wav' with the path to your audio file
# masked_time_spec, sr_time, n_mels_time = time_masking('path_to_file.wav', max_mask_time_frames=80)
# plot_spectrogram(masked_time_spec, sr_time, n_mels_time, title="Time Masking")
