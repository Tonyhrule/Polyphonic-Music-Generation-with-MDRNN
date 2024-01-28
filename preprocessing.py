import pretty_midi
import numpy as np
import os


def processMidi(midi_file):
    """
    Convert MIDI file to a Piano Roll array.
    :param midi_file: Path to the MIDI file.
    :param fs: Sampling frequency of the columns, example is like => each column is 1/fs seconds.
    :param pitch_range: Tuple (min_pitch, max_pitch) representing the range of pitches to include.
    :return: Piano roll of shape (num_pitches, time_steps).
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    def noteToVector(note):
        """
        Convert a note to a one-hot vector representation.
        :param note: Note to convert.
        :return: One-hot vector representation of the note.
        """
        return (
            note.pitch,
            midi_data.time_to_tick(note.start),
            midi_data.time_to_tick(note.end) - midi_data.time_to_tick(note.start),
        )

    return np.concatenate(
        (
            np.array(list(map(noteToVector, midi_data.instruments[0].notes))),
            np.array(
                [
                    (
                        0,
                        midi_data.time_to_tick(midi_data.instruments[0].notes[-1].end),
                        0,
                    )
                ]
            ),
        ),
        axis=0,
    )


midi_directory = "./midis"
fileList = []
for root, dirs, files in os.walk(midi_directory):
    for file in files:
        if file.endswith(".mid"):
            fileList.append(os.path.join(root, file))

if not os.path.exists("output"):
    os.mkdir("output")

for i in range(len(fileList)):
    try:
        notes = processMidi(fileList[i])
        np.save("output/output" + str(i) + ".npy", notes)
    except:
        print("Error with" + fileList[i])
