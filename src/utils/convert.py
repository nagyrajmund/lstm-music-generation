import os
import random
import numpy as np
import pandas as pd
from math import floor
import music21
import random

# Copied from https://github.com/mcleavey/musical-neural-net/. Paths have been changed.
    
def write_midi(s, filename, output_folder):
    fp = s.write('midi', fp=output_folder + "/" + filename)
    
def string_inds_to_stream(string, sample_freq, note_offset, chordwise):
    score_i = string.split(" ")
    if chordwise:
        return arrToStreamChordwise(score_i, sample_freq, note_offset)
    else:
        return arrToStreamNotewise(score_i, sample_freq, note_offset)

def arrToStreamChordwise(score, sample_freq, note_offset):

    speed=1./sample_freq
    piano_notes=[]
    violin_notes=[]
    time_offset=0
    for i in range(len(score)):
        if len(score[i])==0:
            continue

        for j in range(1,len(score[i])):
            if score[i][j]=="1":
                duration=2
                new_note=music21.note.Note(j+note_offset)    
                new_note.duration = music21.duration.Duration(duration*speed)
                new_note.offset=(i+time_offset)*speed
                if score[i][0]=='p':
                    piano_notes.append(new_note)
                elif score[i][0]=='v':
                    violin_notes.append(new_note)
    violin=music21.instrument.fromString("Violin")
    piano=music21.instrument.fromString("Piano")
    violin_notes.insert(0, violin)
    piano_notes.insert(0, piano)
    violin_stream=music21.stream.Stream(violin_notes)
    piano_stream=music21.stream.Stream(piano_notes)
    main_stream = music21.stream.Stream([violin_stream, piano_stream])
    return main_stream
                    
def arrToStreamNotewise(score, sample_freq, note_offset):
    speed=1./sample_freq
    piano_notes=[]
    violin_notes=[]
    time_offset=0
    
    i=0
    while i<len(score):
        if score[i][:9]=="p_octave_":
            add_wait=""
            if score[i][-3:]=="eoc":
                add_wait="eoc"
                score[i]=score[i][:-3]
            this_note=score[i][9:]
            score[i]="p"+this_note
            score.insert(i+1, "p"+str(int(this_note)+12)+add_wait)
            i+=1
        i+=1
        
    for i in range(len(score)):
        if score[i] in ["", " ", "<eos>", "<unk>"]:
            continue
        elif score[i][:3]=="end":
            if score[i][-3:]=="eoc":
                time_offset+=1
            continue
        elif score[i][:4]=="wait":
            time_offset+=int(score[i][4:])
            continue
        else:
            # Look ahead to see if an end<noteid> was generated
            # soon after.  
            duration=1
            has_end=False
            note_string_len = len(score[i])
            for j in range(1,200):
                if i+j==len(score):
                    break
                if score[i+j][:4]=="wait":
                    duration+=int(score[i+j][4:])
                if score[i+j][:3+note_string_len]=="end"+score[i] or score[i+j][:note_string_len]==score[i]:
                    has_end=True
                    break
                if score[i+j][-3:]=="eoc":
                    duration+=1

            if not has_end:
                duration=12

            add_wait = 0
            if score[i][-3:]=="eoc":
                score[i]=score[i][:-3]
                add_wait = 1

            try: 
                new_note=music21.note.Note(int(score[i][1:])+note_offset)    
                new_note.duration = music21.duration.Duration(duration*speed)
                new_note.offset=time_offset*speed
                if score[i][0]=="v":
                    violin_notes.append(new_note)
                else:
                    piano_notes.append(new_note)                
            except:
                print("Unknown note: " + score[i])
            
            time_offset+=add_wait
                
    violin=music21.instrument.fromString("Violin")
    piano=music21.instrument.fromString("Piano")
    violin_notes.insert(0, violin)
    piano_notes.insert(0, piano)
    violin_stream=music21.stream.Stream(violin_notes)
    piano_stream=music21.stream.Stream(piano_notes)
    main_stream = music21.stream.Stream([violin_stream, piano_stream])
    return main_stream

def write_mid_mp3_wav(stream, fname, sample_freq, note_offset, out, chordwise):
    stream_out=string_inds_to_stream(stream, sample_freq, note_offset, chordwise)
    write_midi(stream_out, fname, out)
    base=out + "/" + fname[:-4]
    # Unix commands
    # os.system(f'./scripts/mid2mp3.sh {base}.mid')
    # os.system(f'mpg123 -w {base}.wav {base}.mp3')