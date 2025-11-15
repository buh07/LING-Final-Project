# Speech Processing Transformations

A Python-based speech processing tool that applies various transformations to audio files, making speech unintelligible while preserving auditory characteristics.

## Authors
Benjamin Huh and Minh Nguyen  
Dartmouth College - LING 11.13 Final Project

## Features

This script provides four distinct audio transformations:

1. **Pitch Synthesis (Form 1)**: Extracts pitch from speech and synthesizes it using configurable instruments (piano, guitar, or flute) with ADSR envelope
2. **High-Pass Filter (Form 2)**: Dynamically filters out low frequencies, removing speech intelligibility while preserving high-frequency content
3. **Low-Pass Filter (Form 3)**: Dynamically filters out high frequencies, creating a "muffled" effect while maintaining some auditory presence
4. **Rhythm Synthesis (Form 4)**: Detects speech rhythm via onset detection and synthesizes repeated notes matching the original timing

## Key Features

- **Dynamic Filtering**: Both high-pass and low-pass filters automatically adapt to each audio file's frequency characteristics
- **Multiple Format Support**: Handles `.mp4`, `.m4a`, `.mp3`, and `.wav` files
- **Video Preservation**: For video inputs, preserves the video stream while replacing the audio track
- **Adaptive Amplification**: Prevents silent outputs by automatically adjusting volume levels

## Requirements

```bash
pip install numpy librosa soundfile scipy moviepy
```

You'll also need `ffmpeg` installed on your system:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

## Usage

```bash
python3 HuhNguyen.speech_processor.py <path_to_audio_file_without_extension>
```

Example:
```bash
python3 HuhNguyen.speech_processor.py backup_output/English1
```

The script will look for `<filename>.m4a` or `<filename>.mp4` and generate four output files in the `output/` directory:
- `<filename>_output1.m4a` - Pitch synthesis
- `<filename>_output2.m4a` - High-pass filtered
- `<filename>_output3.m4a` - Low-pass filtered  
- `<filename>_output4.m4a` - Rhythm synthesis

## Configuration

Edit the macros at the top of the script to customize:

- `SAMPLE_RATE`: Audio sample rate (default: 44100 Hz)
- `FRAME_SIZE`: Frame size for spectral analysis (default: 2048)
- `HOP_LENGTH`: Hop length for analysis (default: 512)
- `INSTRUMENT`: Instrument for synthesis - "piano", "guitar", or "flute" (default: "piano")
- `MIN_PITCH` / `MAX_PITCH`: Pitch detection range for human speech

## Technical Details

### Dynamic Filtering
- **High-pass filter**: Cutoff set to 1.5× median spectral centroid, clamped between 3000-5000 Hz
- **Low-pass filter**: Cutoff set to 1.8× median fundamental frequency, clamped between 250-500 Hz

Both filters include adaptive amplification to ensure audible output even when filtered signal energy is low.

## License

Academic project for Dartmouth College LING 11.13.
