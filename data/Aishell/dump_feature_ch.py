import sys, os
import json

import torch
import whisper

setname = "/WhisperBiasing/data/Aishell/aishell/data_aishell/wav/train"
outname = "/WhisperBiasing/data/Aishell/train"
tokenizer = whisper.tokenizer.get_tokenizer(True, language="en")

features = {}

with open("/WhisperBiasing/data/Aishell/trans/dev/text") as fin:    
    dev_dir = "/WhisperBiasing/data/Aishell/aishell/data_aishell/wav/dev"
    for line in fin:
        uttname = line.split()[0] 
        print(uttname)
        uttname_dir = uttname[6:11]
        utt = " " + ' '.join(line.split()[1:])
        utttokens = tokenizer.encode(utt.lower())
        audiopath = os.path.join( f"{dev_dir}/{uttname_dir}", "{}.wav".format(uttname))
        audio = whisper.load_audio(audiopath)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        dumppath = os.path.join(outname, "{}_fbank.pt".format(uttname))
        torch.save(mel, dumppath)
        datapiece = {"fbank": dumppath, "words": utt}
        features[uttname] = datapiece
    
output_file_path = "/WhisperBiasing/data/Aishell/dev.json"
with open(output_file_path, "w" ,encoding='utf-8') as fout:
    json.dump(features, fout, ensure_ascii=False, indent=4)
        



