import argparse
from multitts.models import ttsmodel

def parse_configs():
    parser = argparse.ArgumentParser(description="A demo of multi-language text to speech")
    parser.add_argument("--text", type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--speaker_wav", type=str)
    parser.add_argument("--out_dir", type=str)
    
    args = parser.parse_args()
    
    return args



if __name__ == "__main__":
    args = parse_configs()
    text = args.text
    lang = args.lang
    speaker_wav = args.speaker_wav
    out_dir = args.out_dir
    
    tts_obj = ttsmodel.TTSMulti(
            speaker_wav= speaker_wav,
            text= text, lang = lang, out_path = out_dir
        )
    tts_obj.synth()