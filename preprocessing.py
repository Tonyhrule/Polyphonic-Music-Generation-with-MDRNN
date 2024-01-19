import pretty_midi

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
    return (note.pitch, midi_data.time_to_tick(note.start), midi_data.time_to_tick(note.end) - midi_data.time_to_tick(note.start))

  return list(map(noteToVector, midi_data.instruments[0].notes))

midi_file_path = './Mozart/mozart_-_Turkish_March_in_Bb.mid'
notes = processMidi(midi_file_path)
