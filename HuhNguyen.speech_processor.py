#!/usr/bin/env python3

"""
Benjamin Huh and Minh Nguyen
Professor McPherson, Professor Levin, Professor Diabate
Ling 11.13 Final Project
Speech Processing Script
This script provides various transformations for speech signals.
"""

# Configuration Macros
# Input/Output Settings
FILE_NAME = "path_to_your_audio_file"  # Base name of the input file (without extension). Prefer .m4a
OUTPUT_DIR = "output"  # Directory for output files

# Audio Processing Parameters
SAMPLE_RATE = 44100  # Default sample rate in Hz
FRAME_SIZE = 2048    # Frame size for spectral analysis
HOP_LENGTH = 512     # Number of samples between successive frames

# Synthesis Parameters
INSTRUMENT = "piano"  # Instrument type for synthesis
MIN_PITCH = 100.0   # Approximately G2 - lower bound for human speech, avoiding octave errors
MAX_PITCH = 400.0   # Upper bound for typical human speech pitch

# Instrument Parameters
INSTRUMENTS = {
    "piano": {
        "frequency_range": (27.5, 4186),  # Hz (A0 to C8)
        "attack_time": 0.02,
        "decay_time": 0.15,
        "sustain_level": 0.6,
        "release_time": 0.5
    },
    "guitar": {
        "frequency_range": (82, 1397),    # Hz (E2 to F6)
        "attack_time": 0.05,
        "decay_time": 0.1,
        "sustain_level": 0.7,
        "release_time": 0.3
    },
    "flute": {
        "frequency_range": (262, 2093),   # Hz (C4 to C7)
        "attack_time": 0.1,
        "decay_time": 0.05,
        "sustain_level": 0.8,
        "release_time": 0.1
    }
}

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter
import librosa.display

# moviepy will be used to read/write mp4 (video) files and handle audio tracks
# Try the convenient editor import; if that submodule isn't available in the
# installed moviepy build, import the pieces we need and provide a small
# compatibility shim named `mp` so the rest of the script can use mp.VideoFileClip,
# mp.AudioFileClip, and mp.AudioArrayClip.
try:
    import moviepy.editor as mp
except Exception:
    # Import the specific classes directly
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.audio.AudioClip import AudioArrayClip

    class _MPShim:
        pass

    mp = _MPShim()
    mp.VideoFileClip = VideoFileClip
    mp.AudioFileClip = AudioFileClip
    mp.AudioArrayClip = AudioArrayClip
import sys

def load_audio(file_path):
    """Load the audio from an input file (supports .mp4 and common audio files).

    For .mp4 files this extracts the audio track (using moviepy) and returns a
    mono numpy array sampled at SAMPLE_RATE. For other audio files, librosa
    is used.

    Returns:
        tuple: (signal, sample_rate)
    """
    lower = file_path.lower()
    # Handle video files (.mp4) by extracting the audio track from the video
    if lower.endswith('.mp4'):
        clip = mp.VideoFileClip(file_path)
        if clip.audio is None:
            raise ValueError(f"No audio track found in {file_path}")
        arr = clip.audio.to_soundarray(fps=SAMPLE_RATE)
        # If stereo/multi-channel, convert to mono by averaging channels
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr_mono = arr.mean(axis=1)
        else:
            arr_mono = arr.reshape(-1)
        return arr_mono.astype(np.float32), SAMPLE_RATE

    # Handle audio-only files (e.g., .m4a, .mp3, .wav) using moviepy's AudioFileClip
    if lower.endswith('.m4a') or lower.endswith('.mp3') or lower.endswith('.wav'):
        clip = mp.AudioFileClip(file_path)
        arr = clip.to_soundarray(fps=SAMPLE_RATE)
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr_mono = arr.mean(axis=1)
        else:
            arr_mono = arr.reshape(-1)
        return arr_mono.astype(np.float32), SAMPLE_RATE
    else:
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
        return signal, sample_rate

def create_adsr_envelope(duration, sample_rate, instrument_params):
    """Create an ADSR (Attack, Decay, Sustain, Release) envelope.
    
    Args:
        duration (float): Total duration in seconds
        sample_rate (int): Sampling rate in Hz
        instrument_params (dict): Dictionary containing ADSR parameters
        
    Returns:
        numpy.ndarray: The ADSR envelope
    """
    total_samples = int(max(1, duration * sample_rate))
    attack_samples = int(instrument_params['attack_time'] * sample_rate)
    decay_samples = int(instrument_params['decay_time'] * sample_rate)
    release_samples = int(instrument_params['release_time'] * sample_rate)

    # Ensure segment counts are non-negative
    attack_samples = max(0, attack_samples)
    decay_samples = max(0, decay_samples)
    release_samples = max(0, release_samples)

    # If the sum of ADSR segments exceeds total, scale them down proportionally
    seg_sum = attack_samples + decay_samples + release_samples
    if seg_sum >= total_samples:
        # Avoid zero division; ensure at least one sample per segment if possible
        if seg_sum == 0:
            attack_samples = decay_samples = release_samples = 0
        else:
            scale = total_samples / float(seg_sum)
            attack_samples = max(0, int(attack_samples * scale))
            decay_samples = max(0, int(decay_samples * scale))
            release_samples = max(0, total_samples - (attack_samples + decay_samples))

    sustain_samples = total_samples - attack_samples - decay_samples - release_samples
    sustain_samples = max(0, sustain_samples)

    # Create envelope segments (use at least length-1 arrays when needed)
    attack = np.linspace(0, 1, attack_samples, endpoint=False) if attack_samples > 0 else np.array([])
    decay = np.linspace(1, instrument_params['sustain_level'], decay_samples, endpoint=False) if decay_samples > 0 else np.array([])
    sustain = np.ones(sustain_samples) * instrument_params['sustain_level'] if sustain_samples > 0 else np.array([])
    release = np.linspace(instrument_params['sustain_level'], 0, release_samples) if release_samples > 0 else np.array([])

    # Combine segments
    envelope = np.concatenate([attack, decay, sustain, release]) if total_samples > 0 else np.array([1.0])

    # If rounding produced a mismatch in length, trim or pad with zeros
    if envelope.size > total_samples:
        envelope = envelope[:total_samples]
    elif envelope.size < total_samples:
        pad = np.zeros(total_samples - envelope.size)
        envelope = np.concatenate([envelope, pad])

    return envelope

def synthesize_note(frequency, duration, sample_rate, instrument_params):
    """Synthesize a note with the given frequency using the specified instrument parameters.
    
    Args:
        frequency (float): Frequency of the note in Hz
        duration (float): Duration of the note in seconds
        sample_rate (int): Sampling rate in Hz
        instrument_params (dict): Dictionary containing instrument parameters
        
    Returns:
        numpy.ndarray: The synthesized note
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate basic waveform (sine wave)
    signal = np.sin(2 * np.pi * frequency * t)
    
    # Apply ADSR envelope
    envelope = create_adsr_envelope(duration, sample_rate, instrument_params)
    signal = signal * envelope
    
    return signal

def transform_form1(signal, sample_rate):
    """Transform the input speech signal into a pure pitch sequence using the specified instrument.
    This transformation extracts the pitch from the input signal and synthesizes it using
    the instrument defined in the INSTRUMENT macro.
    
    Args:
        signal (numpy.ndarray): The input audio signal
        sample_rate (int): The sampling rate of the signal in Hz
        
    Returns:
        numpy.ndarray: The transformed signal containing the synthesized pitch sequence
    """
    # Extract fundamental frequency (f0) using librosa's pyin (preferred)
    # pyin returns f0 values in Hz (with np.nan for unvoiced frames).
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            signal,
            fmin=MIN_PITCH,
            fmax=MAX_PITCH,
            sr=sample_rate,
            frame_length=FRAME_SIZE,
            hop_length=HOP_LENGTH,
        )
    except Exception:
        # Fallback to yin if pyin isn't available
        f0 = librosa.yin(signal, fmin=MIN_PITCH, fmax=MAX_PITCH, sr=sample_rate, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)
        voiced_flag = ~np.isnan(f0)

    # Compute RMS energy for each frame to detect silence
    rms = librosa.feature.rms(y=signal, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    # Set silence threshold as a fraction of the maximum RMS
    silence_threshold = 0.02 * np.max(rms)

    # Build pitch sequence from f0 frames
    pitch_sequence = []
    frame_duration = HOP_LENGTH / float(sample_rate)
    for t in range(len(f0)):
        pitch = f0[t]
        # Check if this frame is silent based on RMS energy
        is_silent = t < len(rms) and rms[t] < silence_threshold

        if is_silent or np.isnan(pitch) or pitch <= 0.0:
            pitch_sequence.append((0.0, frame_duration))
        else:
            # Apply octave shift: shift all pitches up one octave
            pitch = pitch * 2.0

            # Clamp pitch to instrument range as a safety
            if pitch < INSTRUMENTS[INSTRUMENT]['frequency_range'][0] or pitch > INSTRUMENTS[INSTRUMENT]['frequency_range'][1]:
                pitch_sequence.append((0.0, frame_duration))
            else:
                pitch_sequence.append((float(pitch), frame_duration))
    
    # Synthesize the pitch sequence using the specified instrument
    transformed_signal = np.array([])
    instrument_params = INSTRUMENTS[INSTRUMENT]
    
    # Synthesize each detected pitch
    for pitch, duration in pitch_sequence:
        if pitch > 0:  # If there is a pitch (not silence)
            note = synthesize_note(pitch, duration, sample_rate, instrument_params)
        else:  # For silence
            note = np.zeros(int(duration * sample_rate))
        transformed_signal = np.concatenate([transformed_signal, note])
    
    # Normalize the output
    transformed_signal = librosa.util.normalize(transformed_signal)
    
    return transformed_signal

def transform_form2(signal, sample_rate):
    """Apply a high-pass filter to the input audio signal.
    This filter is dynamically tuned to make speech unintelligible while retaining sound.
    
    Args:
        signal (numpy.ndarray): The input audio signal
        sample_rate (int): The sampling rate of the signal in Hz
        
    Returns:
        numpy.ndarray: The high-pass filtered signal
    """
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        # Ensure cutoff is valid (must be < Nyquist frequency)
        if normal_cutoff >= 1.0:
            normal_cutoff = 0.99
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def highpass_filter(data, cutoff, fs, order=5):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        # Ensure output length matches input length
        if len(y) > len(data):
            y = y[:len(data)]
        elif len(y) < len(data):
            y = np.pad(y, (0, len(data) - len(y)))
        return y

    # Compute the signal's spectral centroid to find where most energy is
    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, hop_length=HOP_LENGTH)[0]
    median_centroid = np.median(spectral_centroids[spectral_centroids > 0])
    
    # Set cutoff to be well above speech formants (typically 500-3500 Hz)
    # This removes intelligibility while keeping high-frequency content
    # Use 1.5x the median centroid, clamped between 3000-5000 Hz
    cutoff = min(max(median_centroid * 1.5, 3000.0), 5000.0)
    
    filtered_signal = highpass_filter(signal.copy(), cutoff, sample_rate, order=5)
    
    # Check if signal is too quiet and apply adaptive amplification
    max_amplitude = np.max(np.abs(filtered_signal))
    if max_amplitude < 0.01:
        # Signal is very quiet, apply stronger amplification
        amplification = min(0.3 / max(max_amplitude, 1e-6), 20.0)
    else:
        amplification = 3.0
    
    filtered_signal = librosa.util.normalize(filtered_signal) * amplification
    return filtered_signal

def transform_form3(signal, sample_rate):
    """Apply a low-pass filter to the input audio signal.
    This filter is dynamically tuned to make speech unintelligible while retaining sound.
    
    Args:
        signal (numpy.ndarray): The input audio signal
        sample_rate (int): The sampling rate of the signal in Hz
        
    Returns:
        numpy.ndarray: The low-pass filtered signal
    """
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        # Ensure cutoff is valid
        if normal_cutoff >= 1.0:
            normal_cutoff = 0.99
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        # Ensure output length matches input length
        if len(y) > len(data):
            y = y[:len(data)]
        elif len(y) < len(data):
            y = np.pad(y, (0, len(data) - len(y)))
        return y

    # Analyze signal's pitch content to set an appropriate cutoff
    # Extract fundamental frequency to understand speech range
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            signal,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate,
            frame_length=FRAME_SIZE,
            hop_length=HOP_LENGTH,
        )
        # Use median pitch of voiced frames
        voiced_pitches = f0[~np.isnan(f0)]
        if len(voiced_pitches) > 0:
            median_pitch = np.median(voiced_pitches)
        else:
            median_pitch = 200.0  # Default fallback
    except Exception:
        median_pitch = 200.0  # Fallback if pitch detection fails
    
    # Set cutoff to remove most formants but keep fundamental and some harmonics
    # This keeps a "muffled" sound but removes intelligibility
    # Use 40-60% of median pitch, clamped between 250-500 Hz
    cutoff = min(max(median_pitch * 1.8, 250.0), 500.0)
    
    filtered_signal = lowpass_filter(signal.copy(), cutoff, sample_rate, order=5)
    
    # Check if signal is too quiet and apply adaptive amplification
    max_amplitude = np.max(np.abs(filtered_signal))
    if max_amplitude < 0.01:
        # Signal is very quiet, apply stronger amplification
        amplification = min(0.3 / max(max_amplitude, 1e-6), 20.0)
    else:
        amplification = 3.0
    
    filtered_signal = librosa.util.normalize(filtered_signal) * amplification
    return filtered_signal

def transform_form4(signal, sample_rate):
    """Synthesize a single repeated note in the rhythm of the input audio file.
    The rhythm is extracted using onset detection, and a single note is synthesized and repeated at each onset.
    
    Args:
        signal (numpy.ndarray): The input audio signal
        sample_rate (int): The sampling rate of the signal in Hz
        
    Returns:
        numpy.ndarray: The transformed signal
    """
    # Calculate total duration from input signal
    total_duration = len(signal) / float(sample_rate)
    
    # Use librosa to detect onsets (rhythm)
    onset_frames = librosa.onset.onset_detect(y=signal, sr=sample_rate, hop_length=HOP_LENGTH, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=HOP_LENGTH)

    # Use a fixed note (e.g., middle C)
    note_freq = 261.63  # Middle C (C4)
    instrument_params = INSTRUMENTS[INSTRUMENT]

    # Calculate durations between onsets, ensuring total duration matches input
    if len(onset_times) > 0:
        durations = np.diff(onset_times)
        last_duration = total_duration - onset_times[-1]
        durations = np.append(durations, last_duration)
    else:
        # If no onsets detected, play a single note for the whole duration
        durations = [total_duration]

    # Synthesize the repeated notes
    transformed_signal = np.array([])
    for duration in durations:
        note = synthesize_note(note_freq, duration, sample_rate, instrument_params)
        transformed_signal = np.concatenate([transformed_signal, note])

    # Normalize output
    transformed_signal = librosa.util.normalize(transformed_signal)
    return transformed_signal

def save_transformations(transformed_signals, output_paths, sample_rate, original_input_path):
    """Save the transformed signals to audio/video files.

    If the original input is a video (.mp4) this will try to preserve the
    video stream and replace its audio track with the transformed audio.
    For audio-only outputs (e.g., .m4a) this writes audio-only files using
    AAC (via moviepy/ffmpeg). Output format is inferred from the output
    filename extension.
    """
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Try loading the original video clip (if available) so we can preserve video
    video_clip = None
    try:
        if original_input_path.lower().endswith('.mp4'):
            video_clip = mp.VideoFileClip(original_input_path)
    except Exception:
        video_clip = None

    for signal, path in zip(transformed_signals, output_paths):
        if signal is None:
            print(f"Skipping save for {path}: transformation is None")
            continue

        arr = np.asarray(signal)
        out_path = os.path.join(OUTPUT_DIR, path)

        try:
            # moviepy expects shape (n_frames, n_channels). Create mono channel.
            # Ensure we have a 2D array shaped (n_frames, n_channels).
            if arr.ndim == 1:
                audio_arr = arr.reshape(-1, 1)
            elif arr.ndim == 2:
                # If there are multiple channels, convert to mono by averaging
                if arr.shape[1] > 1:
                    mono = arr.mean(axis=1)
                    audio_arr = mono.reshape(-1, 1)
                else:
                    audio_arr = arr
            else:
                # Fallback: flatten into mono
                audio_arr = arr.reshape(-1, 1)

            # Create mono audio clip, maintaining original length
            audio_clip = mp.AudioArrayClip(audio_arr, fps=sample_rate)

            if video_clip is not None:
                # Attach transformed audio to original video and write a new mp4
                new_video = video_clip.set_audio(audio_clip)
                # Write video with new audio (preserve reasonable defaults)
                new_video.write_videofile(out_path, codec='libx264', audio_codec='aac')
                print(f"Wrote video with new audio: {out_path}")
            else:
                # Audio-only output. Use soundfile for direct writing to avoid resampling issues
                try:
                    # Convert audio_arr back to 1D for soundfile (it expects mono as 1D or Nx1)
                    if audio_arr.shape[1] == 1:
                        audio_1d = audio_arr.flatten()
                    else:
                        audio_1d = audio_arr.mean(axis=1)

                    # Write as WAV first (lossless), then convert to m4a with ffmpeg
                    import os
                    wav_temp = os.path.splitext(out_path)[0] + '_temp.wav'
                    sf.write(wav_temp, audio_1d, sample_rate)

                    # Convert to m4a using ffmpeg
                    import subprocess
                    subprocess.run(['ffmpeg', '-y', '-i', wav_temp, '-c:a', 'aac', '-b:a', '192k', out_path],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    os.remove(wav_temp)
                    print(f"Wrote audio-only file: {out_path}")
                except Exception as e:
                    print(f"Failed to write {out_path} with soundfile/ffmpeg: {e}, trying moviepy fallback")
                    # Fallback to moviepy
                    try:
                        audio_clip.write_audiofile(out_path, codec='aac')
                        print(f"Wrote audio-only file: {out_path}")
                    except Exception:
                        # As a fallback, try writing WAV (if codec/format not supported)
                        wav_fallback = os.path.splitext(out_path)[0] + '.wav'
                        audio_clip.write_audiofile(wav_fallback, codec=None)
                        print(f"Wrote fallback WAV audio: {wav_fallback}")

        except Exception as e:
            print(f"Failed to write {out_path}: {e}")

def main(file_name):
    """Main function to process the audio file.
    
    Args:
        file_name (str): Base name of the input file (without .wav extension)
    """
    # Prefer audio-only input (.m4a). If not found, fall back to .mp4
    import os
    m4a_path = f"{file_name}.m4a"
    mp4_path = f"{file_name}.mp4"
    if os.path.exists(m4a_path):
        input_file = m4a_path
    else:
        input_file = mp4_path

    # Load the audio track from the input (moviepy/librosa handled in load_audio)
    signal, sample_rate = load_audio(input_file)

    # Apply transformations
    form1 = transform_form1(signal, sample_rate)
    form2 = transform_form2(signal, sample_rate)
    form3 = transform_form3(signal, sample_rate)
    form4 = transform_form4(signal, sample_rate)

    # Extract just the base name from the file path (remove directory)
    base_name = os.path.basename(file_name)
    
    # Define output paths for each transformation (use .m4a for audio outputs)
    output_paths = [
        f"{base_name}_output1.m4a",
        f"{base_name}_output2.m4a",
        f"{base_name}_output3.m4a",
        f"{base_name}_output4.m4a",
    ]

    # Save all transformations and attach them to the original video when possible
    save_transformations([form1, form2, form3, form4], output_paths, sample_rate, input_file)

if __name__ == "__main__":
    # Usage: python HuhNguyen.speech_processor.py [basename]
    # If a basename is provided (without extension) it will be used; otherwise
    # the FILE_NAME constant is used.
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # Set this to your file's basename (without .mp4) or pass as argument
        main(FILE_NAME)