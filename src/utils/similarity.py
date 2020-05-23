import difflib

file_name_a = "d:/KTH/lstm-music-generation/data/datasets/bach/1030b1st.txt"
file_name_b = "d:/KTH/lstm-music-generation/data/datasets/bach/1030b3rd.txt"
f = open(file_name_a, "r")
a = f.read()
f.close()

f = open(file_name_b, "r")
b = f.read()
f.close()

seq = difflib.SequenceMatcher(None, a, b)
d = seq.ratio() * 100
print(d)