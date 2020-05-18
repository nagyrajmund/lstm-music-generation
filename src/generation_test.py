from utils.utils import write_mid_mp3_wav

# An example for generating midi from chordwise notation.

input_file = "../dataset/test_dataset/Bach-Prelude-No-1-Ave-Maria.txt"
with open(input_file) as f:
    data = f.read().split(" ")

stream = " ".join(data[:2000]) # First 2000 notes
sample_freq = 12
note_offset = 38
out = "../out/test_dataset"
fname = "bach_ave_maria_jazz.mid"
chordwise = False
write_mid_mp3_wav(stream, fname, sample_freq, note_offset, out, chordwise)