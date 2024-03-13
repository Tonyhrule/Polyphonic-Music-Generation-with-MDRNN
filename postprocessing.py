import pretty_midi
import numpy as np


def postProcess(notes):
    midi = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(0)

    for note in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=int(note[0]),
                start=midi.tick_to_time(int(note[1])),
                end=midi.tick_to_time(int(note[1] + note[2])),
            )
        )

        midi.instruments.append(instrument)

    midi.write("output.mid")

"""
output_array = np.load("output/bach-aof-can1.npy")
postProcess(output_array)
"""