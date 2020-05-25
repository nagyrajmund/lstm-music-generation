
from utils.convert import write_mid_mp3_wav

def generate_sound(model, args, use_tqdm=True):
    print("Generating...")
    
    # Set default output file name
    if args.output_name is None:
        args.output_name = args.model_file + "_" + str(args.input_len) + "_" + str(args.predic_len)

    # Generate
    generated_nums = model.generate(args.random_seed, args.input_len, args.predic_len, use_tqdm)

    # Convert tokens to notes
    num_to_note = model.dataset.num_to_note
    notes = [num_to_note[num] for num in generated_nums]

    notes = " ".join(notes)

    # Save notes as txt
    f = open(args.output_path + "/" + args.output_name + ".txt", "w")
    f.write(notes)
    f.close()
    
    # Save as midi
    write_mid_mp3_wav(notes, args.output_name + ".mid", args.sample_freq, args.note_offset, args.output_path, args.use_chordwise)