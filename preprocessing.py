import pretty_midi
import numpy as np
from scipy.sparse import lil_matrix

def midi_to_piano_roll(midi_file, fs=100, pitch_range=(21, 108)):
    """
    Convert MIDI file to a Piano Roll array.
    :param midi_file: Path to the MIDI file.
    :param fs: Sampling frequency of the columns, example is like => each column is 1/fs seconds.
    :param pitch_range: Tuple (min_pitch, max_pitch) representing the range of pitches to include.
    :return: Piano roll of shape (num_pitches, time_steps).
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    # Calculate the piano roll matrix size
    num_pitches = pitch_range[1] - pitch_range[0] + 1
    max_time = int(np.ceil(midi_data.get_end_time() * fs))
    piano_roll = lil_matrix((num_pitches, max_time), dtype=np.int8)

    # Populate the piano roll matrix
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                start = int(note.start * fs)
                end = int(note.end * fs)
                pitch = note.pitch - pitch_range[0]
                # Fill the corresponding cells with velocity
                piano_roll[pitch, start:end] = note.velocity

    return piano_roll.toarray()

def preprocess_piano_roll(piano_roll, max_length=10000):
    """
    Preprocess the piano roll for input into the neural network.
    :param piano_roll: The piano roll array.
    :param max_length: The maximum length for padding/truncating the sequences.
    :return: Preprocessed piano roll.
    """
    # Padding to the max_length
    padded_piano_roll = np.zeros((piano_roll.shape[0], max_length))
    actual_length = min(max_length, piano_roll.shape[1])
    padded_piano_roll[:, :actual_length] = piano_roll[:, :actual_length]

    # Normalizing the velocity values to 0-1 range
    normalized_piano_roll = np.clip(padded_piano_roll, 0, 127) / 127

    return normalized_piano_roll

# Example of how we gonna use it
midi_file_path = 'path/to/your/midi/file.mid'  # Replace with the MIDI file path
piano_roll = midi_to_piano_roll(midi_file_path)
preprocessed_piano_roll = preprocess_piano_roll(piano_roll)
