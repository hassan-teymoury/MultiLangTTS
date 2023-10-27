#  MultiLangTTS
A python package that convert texts to speechs for multiple languages based on the following 
awesome baseline references (check these repos for more details):

1. [VITS](https://github.com/jaywalnut310/vits.git)

2. [TTS](https://github.com/coqui-ai/TTS)

3. [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/mms#asr)


# Get Started

## Install

To install __MultiLangTTS__ as a python package, please go through the following steps

1. insall  __`python > 3.9`__

2. Clone the repository : __`git clone https://github.com/hassan-teymoury/MultiLangTTS.git`__

3. Install __`multitts`__ from repo: 

    ```terminal
    cd  MultiLangTTS
    pip install . or (pip install -e .)
    ```


## Running demo

To test a text to speech task on your data run the follow these steps:


1. Please find the ISO code for your target language [here](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html). Also you can find more details about the languages that [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/mms#asr) currently supports for TTS in this [table](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html).

2. Run the following command on your terminal with arguments:

__` python demo.py --text=your_text --lang=your_target_language --speaker_wav=your_speaker_wav_path --out_dir=path_of_the_results_for_speech_generated `__

__Example:__

```terminal
python demo.py --text=Ey qəhrəman övladın şanlı Vətəni! Səndən ötrü can verməyə cümlə hazırız! --lang=azj-script_latin --speaker_wav=audio.wav --out_dir=results.wav
```

__Python Script:__

```python

from multitts.models import ttsmodel

# Azerbaijani
text = " Ey qəhrəman övladın şanlı Vətəni! Səndən ötrü can verməyə cümlə hazırız!"
language = "azj-script_latin"


"""
text = "मुदा आइ धरि ई तकनीक सौ सं किछु बेसी भाषा तक सीमित छल जे सात हजार सं बेसी ज्ञात भाषाक एकटा अंश अछी"
language = "mai" 
"""

tts_obj = ttsmodel.TTSMulti(
    speaker_wav= "audio.wav",
    text= text, lang = language, out_path = "/content/outtest_wav.wav"
)
tts_obj.synth()

```
