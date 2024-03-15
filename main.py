import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydub import AudioSegment
from tabulate import tabulate

from constants import (
    AI,
    AUDIO_DIR_NAME,
    PASSAGES,
    PLOTS_DIR_NAME,
    REAL,
    SPEAKER_TYPES,
    SPEAKERS,
    STATS_DIR_NAME,
    WAV
)

def convert_audio(input_file, output_file):
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

# Convert files to .wav
print("Starting converting files to .wav format")
for speaker_type in SPEAKER_TYPES:
    for speaker in SPEAKERS:
        for passage in PASSAGES:
            ext = ".mp3" if speaker_type == AI else ".m4a"
            in_path = "/".join([AUDIO_DIR_NAME, speaker_type, speaker, passage + ext])
            out_path = "/".join([AUDIO_DIR_NAME, speaker_type, speaker, passage + WAV])
            convert_audio(in_path, out_path)
print("Finished converting files to .wav format")


print("Starting yin pitch detection")
data_df = pd.DataFrame(columns=['passage', 'speaker', 'type', 'f0', 'prob'])
for passage in PASSAGES:
    for speaker in SPEAKERS:
        for speaker_type in SPEAKER_TYPES:
            audio_file = "/".join([AUDIO_DIR_NAME, speaker_type, speaker, passage + WAV])
            y, sr = librosa.load(audio_file)
            f0, _, prob = librosa.pyin(
                y,
                fmin= 125 if "female" in speaker else 80,
                fmax=350 if "female" in speaker else 200
            )
            rows = pd.DataFrame({
                'passage': [passage] * len(f0),
                'speaker': [speaker] * len(f0),
                'type': [speaker_type] * len(f0),
                'f0': f0,
                'prob': prob
            })
            data_df = pd.concat([data_df, rows], ignore_index=True)
            data_df = data_df.dropna(subset=['f0'])
            data_df = data_df[data_df['prob'] >= 0.8]
print("Finished yin pitch detection")

print("Starting pitch distribution graphs and statistics")
stats_df = pd.DataFrame(columns=[
    'passage', 'speaker', 'type', 'mean_f0', 'std_dev_f0', 'min_f0', 'max_f0', 'range_f0'
])
for passage in PASSAGES:
    for speaker in SPEAKERS:
        # Get ai and real data for speaker's recording of passage
        ai_df = data_df.query(
            f'(passage == "{passage}") & (speaker == "{speaker}") & (type == "{AI}")'
        )
        real_df = data_df.query(
            f'(passage == "{passage}") & (speaker == "{speaker}") & (type == "{REAL}")'
        )

        # Get statistics
        stats = {
            AI: {
                "mean": round(ai_df['f0'].mean(), 2),
                "std_dev": round(ai_df['f0'].std(), 2),
                "min": round(ai_df['f0'].min(), 2),
                "max": round(ai_df['f0'].max(), 2),
                "range": round(ai_df['f0'].max() - ai_df['f0'].min(), 2)
            },
            REAL: {
                "mean": round(real_df['f0'].mean(), 2),
                "std_dev": round(real_df['f0'].std(), 2),
                "min": round(real_df['f0'].min(), 2),
                "max": round(real_df['f0'].max(), 2),
                "range": round(real_df['f0'].max() - real_df['f0'].min(), 2)
            }
        }

        # Add data to stats_df
        for speaker_type in SPEAKER_TYPES:
            rows = pd.DataFrame({
                'passage': [passage],
                'speaker': [speaker],
                'type': [speaker_type],
                'mean_f0': [stats[speaker_type]["mean"]],
                'std_dev_f0': [stats[speaker_type]["std_dev"]],
                'min_f0': [stats[speaker_type]["min"]],
                'max_f0': [stats[speaker_type]["max"]],
                'range_f0': [stats[speaker_type]["range"]]
            })
            stats_df = pd.concat([stats_df, rows], ignore_index=True)

        # Create pdf graphs
        x_min = min(ai_df['f0'].min(), real_df['f0'].min())
        x_max = min(ai_df['f0'].max(), real_df['f0'].max())
        x = np.linspace(-500, 500, 1000)

        # Create the plot
        plt.figure(figsize=(6, 4))  # Set the figure size

        # Get percent 
        ai_weights = np.ones_like(ai_df['f0']) / len(ai_df['f0'])
        real_weights = np.ones_like(real_df['f0']) / len(real_df['f0'])

        # Plot histograms for both datasets
        plt.hist(ai_df['f0'], bins=30, alpha=0.5, label='AI', weights=ai_weights * 100)
        plt.hist(real_df['f0'], bins=30, alpha=0.5, label='Real', weights=real_weights * 100)

        # Adding plot title and labels
        speaker_formatted = " ".join(map(lambda x: x[0].upper() + x[1:], speaker.split("_")))
        passage_formatted = " ".join(map(lambda x: x[0].upper() + x[1:], passage.split("_")))
        plt.title(f'Distribution Of Fundamental Frequencies For \n{speaker_formatted} Reading {passage_formatted}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Percentage (%)')

        # Add a legend to distinguish the datasets
        plt.legend()

        # Save the plot
        file_name = "_".join([speaker, passage])
        file_path = "/".join([PLOTS_DIR_NAME, file_name])
        plt.savefig(file_path)

## Write stats_df to file
file_name = "speaker_stats"
file_path = "/".join([STATS_DIR_NAME, file_name])
with open(file_path, 'w') as file:
    table_string = tabulate(stats_df, headers='keys', tablefmt='plain', showindex=False)
    file.write(table_string)

print("Finished pitch distribution graphs and statistics")


