"""Module for extracting audio features with data augmentation"""

import random

import librosa
import numpy as np

from .data_augmentation import noise, pitch, shift, stretch


def zcr(data, frame_length, hop_length):
    """Extract Zero Crossing Rate feature from audio data"""
    zcr = librosa.feature.zero_crossing_rate(
        data, frame_length=frame_length, hop_length=hop_length
    )
    return np.squeeze(zcr)


def rmse(data, frame_length=2048, hop_length=512):
    """Extract Root Mean Square Energy feature from audio data"""
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)


def mfcc(
    data,
    sr,
    frame_length=2048,
    hop_length=512,
    flatten=True,
    frequency_mask: bool = True,
    time_mask: bool = True,
):
    """Extract MFCC feature from audio data with optional masking"""
    mfcc_result = librosa.feature.mfcc(
        y=data, sr=sr, n_fft=frame_length, hop_length=hop_length
    )
    if frequency_mask:
        # Apply frequency masking
        num_mel_channels = mfcc_result.shape[0]
        F = np.random.randint(0, 15)  # Randomly choose mask width
        f = np.random.randint(0, num_mel_channels - F)  # Randomly choose start point
        mfcc_result[f : f + F, :] = 0  # Apply mask
    if time_mask:
        # Apply time masking
        num_time_frames = mfcc_result.shape[1]
        T = np.random.randint(0, 20)  # Randomly choose mask width
        t = np.random.randint(0, num_time_frames - T)  # Randomly choose start point
        mfcc_result[:, t : t + T] = 0  # Apply mask
    # Return flattened MFCCs if specified
    return np.squeeze(mfcc_result.T) if not flatten else np.ravel(mfcc_result.T)


def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    """Extract features from audio data"""
    result = np.array([])
    i = random.randint(0, 1)
    result = np.hstack(
        (
            result,
            zcr(data, frame_length, hop_length),
            rmse(data, frame_length, hop_length),
            mfcc(
                data,
                sr,
                frame_length,
                hop_length,
                flatten=True,
                frequency_mask=bool(i),
                time_mask=bool(1 - i),
            ),
        )
    )

    return result


def get_features(path, duration=2.5, offset=0.6):
    """Get features from audio file with data augmentation"""
    data, sr = librosa.load(path, duration=duration, offset=offset)
    aud = extract_features(data)
    audio = np.array(aud)

    noised_audio = noise(data)
    aud2 = extract_features(noised_audio)
    audio = np.vstack((audio, aud2))

    pitched_audio = pitch(data, sr)
    aud3 = extract_features(pitched_audio)
    audio = np.vstack((audio, aud3))

    shifted_audio = shift(data)
    aud4 = extract_features(shifted_audio)
    audio = np.vstack((audio, aud4))

    # Can't stretch data cause it will change the length of features

    return audio
