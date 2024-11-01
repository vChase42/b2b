from typing import List, Dict
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple
import random
import numpy as np



def visualize_diarization_spectrogram(
    audio_path: str,
    segments: List[Dict[str, float]],
    sampling_rate: int = 22050,
    speaker_colors: Dict[str, str] = None,
    transparency: float = 0.3,
    show_labels: bool = True,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Visualizes a spectrogram of the audio file with highlighted speaker segments.
    
    Args:
        audio_path (str): Path to the audio file.
        segments (List[Dict[str, float]]): Packaged segments containing start_time, end_time, and speaker_label.
        sampling_rate (int): Sampling rate for audio loading. Default is 22050 Hz.
        speaker_colors (Dict[str, str]): Optional. Dictionary mapping speaker labels to specific colors.
        transparency (float): Transparency level for segment overlays. Default is 0.3.
        show_labels (bool): Whether to show speaker labels on the segments. Default is True.
        figsize (Tuple[int, int]): Size of the output plot. Default is (12, 8).
    """
    
    
    # Load audio and extract waveform data
    audio, sr = librosa.load(audio_path, sr=sampling_rate)
    duration = librosa.get_duration(y=audio, sr=sr)
    time = np.linspace(0, duration, len(audio))  # Generate time values for x-axis

    # Set up plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, audio, color="black", lw=0.8)  # Plot waveform in black
    ax.set_title("Waveform with Speaker Diarization")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.0, 1.0)  # Set y-axis limits for consistent scaling

    # Define fixed color palette
    color_palette = ["red", "blue", "green", "yellow"]
    
    # Generate color mapping for each speaker using the fixed color palette
    if speaker_colors is None:
        unique_speakers = sorted({segment["speaker_label"] for segment in segments})
        speaker_colors = {speaker: color_palette[i % len(color_palette)] for i, speaker in enumerate(unique_speakers)}
    
    # Overlay each speaker segment
    for segment in segments:
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        speaker_label = segment["speaker_label"]
        
        # Add rectangle for each segment spanning the height of the waveform
        rect = Rectangle(
            (start_time, ax.get_ylim()[0]),  # x (time) and y set to bottom limit of y-axis
            end_time - start_time,  # width of segment
            ax.get_ylim()[1] - ax.get_ylim()[0],  # height to span full y-axis
            linewidth=0,
            edgecolor=None,
            facecolor=speaker_colors[speaker_label],
            alpha=transparency
        )
        ax.add_patch(rect)
        
        # Optionally add labels
        if show_labels:
            ax.text(
                (start_time + end_time) / 2,  # Center label in the segment
                ax.get_ylim()[1] * 0.85,  # Positioning label near the top of the y-axis
                speaker_label,
                ha='center',
                va='center',
                color='white',
                fontsize=8,
                weight='bold'
            )

    plt.tight_layout()
    plt.show()


def package_diarization_for_visualization(diarization) -> List[Dict[str, float]]:
    """
    Process the pyannote Diarization object into a list of dictionaries for visualization.
    
    Args:
        diarization: pyannote `Diarization` object.
        
    Returns:
        List of dictionaries, each containing:
        - 'start_time' (float): Start time of the segment.
        - 'end_time' (float): End time of the segment.
        - 'speaker_label' (str): Speaker label for the segment.
    """
    packaged_segments = []

    # Iterate through each segment in the diarization object
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        packaged_segments.append({
            "start_time": segment.start,
            "end_time": segment.end,
            "speaker_label": speaker
        })
    
    return packaged_segments

from pyannote.audio import Pipeline


if __name__ == "__main__":
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    audio_file_path = "../audio/2024-10-29-13-32-32.wav"   #1:30 minutes

    diarization = pipeline(audio_file_path)

    segments = package_diarization_for_visualization(diarization)

    print(len(segments))
    visualize_diarization_spectrogram(audio_file_path,segments,show_labels=False)

