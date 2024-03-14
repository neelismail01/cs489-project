import numpy as np
from pydub import AudioSegment

from constants import (
    AI,
    AUDIO_DIR_NAME,
    PASSAGES,
    SPEAKER_TYPES,
    SPEAKERS,
    WAV
)
from yin import get_yin

def convert_audio(input_file, output_file):
    """Function to convert audio file"""

    # Determine the file format from the input file
    format = input_file.split('.')[-1]

    # Load the audio file based on its format
    if format == 'mp3':
        audio = AudioSegment.from_mp3(input_file)
    elif format == 'm4a':
        audio = AudioSegment.from_file(input_file, format='m4a')
    else:
        raise ValueError("Unsupported format. Please use MP3 or M4A files.")

    # Export the audio file in WAV format
    audio.export(output_file, format='wav')
    print(f"Conversion successful. File saved as {output_file}")


"""
## Convert files to .wav
print("Starting converting files to .wav format")
for speaker_type in SPEAKER_TYPES:
    for speaker in SPEAKERS:
        for passage in PASSAGES:
            ext = ".mp3" if speaker_type == AI else ".m4a"
            in_path = "/".join([AUDIO_DIR_NAME, speaker_type, speaker, passage + ext])
            out_path = "/".join([AUDIO_DIR_NAME, speaker_type, speaker, passage + WAV])
            convert_audio(in_path, out_path)
print("Finished converting files to .wav format")
"""

## Get fundamental frequency for each speaker's passage
print("Starting yin pitch detection")
for passage in PASSAGES:
    for speaker in SPEAKERS:
        for speaker_type in SPEAKER_TYPES:
            audio_file = "/".join([AUDIO_DIR_NAME, speaker_type, speaker, passage + WAV])
            pitches, harmonic_rates, times_ = get_yin(audio_file)
print("Finished yin pitch detection")
