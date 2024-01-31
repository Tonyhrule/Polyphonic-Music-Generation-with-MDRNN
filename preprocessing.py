import pretty_midi
import numpy as np
import os
import functools


def concatNotes(a, b):
    return a + b.notes


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

    allNotes = list(
        sorted(
            map(
                noteToVector,
                functools.reduce(concatNotes, midi_data.instruments, []),
            ),
            key=lambda x: x[1],
        )
    )

    return np.concatenate(
        (
            np.array(allNotes),
            np.array(
                [
                    (
                        0,
                        midi_data.time_to_tick(allNotes[-1][1] + allNotes[-1][2]),
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

for file in fileList:
    try:
        notes = processMidi(file)
        np.save(
            "output/"
            + file.replace("./midis/", "").replace("/", "-").replace(".mid", "")
            + ".npy",
            notes,
        )
    except:
        print("Error with" + file)
