from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from pathlib import Path
from transformers import VitsModel, AutoTokenizer
import torch
import scipy
from IPython.display import Audio
import numpy as np
import os
import re
import glob
import json
import tempfile
import math
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from multitts.vits import *
import multitts.vits.commons as commons
import multitts.vits.utils as utils
import argparse
import subprocess
from multitts.vits.data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from multitts.vits.models import SynthesizerTrn
from scipy.io.wavfile import write
import os
import subprocess
import locale

locale.getpreferredencoding = lambda: "UTF-8"

pwd = os.getcwd()
mms_model_path = pwd + "/mms_models"
tts_json_path = pwd + "/models.json"


if not os.path.isfile(tts_json_path):
    subprocess.run(f"wget https://raw.githubusercontent.com/coqui-ai/TTS/dev/TTS/.models.json -O {tts_json_path}", shell=True)


# mms_model_path = "/content/mms_models"

if not os.path.isdir(mms_model_path):
    os.mkdir(mms_model_path)


def download(lang, tgt_dir="./"):
    lang_fn, lang_dir = os.path.join(
        tgt_dir, lang+'.tar.gz'), os.path.join(tgt_dir, lang)
    cmd = ";".join([
          f"wget https://dl.fbaipublicfiles.com/mms/tts/{lang}.tar.gz -O {lang_fn}",
          f"tar zxvf {lang_fn} --directory {tgt_dir}"
    ])
    print(f"Download model for language: {lang}")
    subprocess.check_output(cmd, shell=True)
    print(f"Model checkpoints in {lang_dir}: {os.listdir(lang_dir)}")
    return lang_dir


def preprocess_char(text, lang=None):
    """
    Special treatement of characters in certain languages
    """
    print(lang)
    if lang == 'ron':
        text = text.replace("ț", "ţ")
    return text


class TextMapper(object):
    def __init__(self, vocab_file):
        self.symbols = [x.replace("\n", "") for x in open(
            vocab_file, encoding="utf-8").readlines()]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text, cleaner_names):
        '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
        Returns:
        List of integers corresponding to the symbols in the text
        '''
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence

    def uromanize(self, text, uroman_pl):
        iso = "xxx"
        with tempfile.NamedTemporaryFile() as tf, \
                tempfile.NamedTemporaryFile() as tf2:
            with open(tf.name, "w") as f:
                f.write("\n".join([text]))
            cmd = f"perl " + uroman_pl
            cmd += f" -l {iso} "
            cmd += f" < {tf.name} > {tf2.name}"
            os.system(cmd)
            outtexts = []
            with open(tf2.name) as f:
                for line in f:
                    line = re.sub(r"\s+", " ", line).strip()
                    outtexts.append(line)
            outtext = outtexts[0]
        return outtext

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def filter_oov(self, text):
        val_chars = self._symbol_to_id
        txt_filt = "".join(list(filter(lambda x: x in val_chars, text)))
        print(f"text after filtering OOV: {txt_filt}")
        return txt_filt


def preprocess_text(txt, text_mapper, hps, uroman_dir=None, lang=None):
    txt = preprocess_char(txt, lang=lang)
    is_uroman = hps.data.training_files.split('.')[-1] == 'uroman'
    if is_uroman:
        with tempfile.TemporaryDirectory() as tmp_dir:
            if uroman_dir is None:
                cmd = f"git clone git@github.com:isi-nlp/uroman.git {tmp_dir}"
                print(cmd)
                subprocess.check_output(cmd, shell=True)
                uroman_dir = tmp_dir
            uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
            print(f"uromanize")
            txt = text_mapper.uromanize(txt, uroman_pl)
            print(f"uroman text: {txt}")
    txt = txt.lower()
    txt = text_mapper.filter_oov(txt)
    return txt


class TTSMulti(object):
    def __init__(self, speaker_wav, text, lang, out_path):
        self.speaker_wav = speaker_wav
        self.text = text
        self.lang = lang
        self.out_path = out_path
        self.model_hug = VitsModel.from_pretrained(
            "SeyedAli/Persian-Speech-synthesis")
        self.tokenizer_hug = AutoTokenizer.from_pretrained(
            "SeyedAli/Persian-Speech-synthesis")
        self.model_configs = tts_json_path
        path = Path(self.model_configs)
        manager = ModelManager(path)
        all_models = manager.list_models()
        tts_models = manager.list_tts_models()
        vocoder_models = all_models[len(tts_models):len(all_models)]
        
        for ttsmodel in tts_models:
            if "your_tts" in ttsmodel:
                sample_tts_model = ttsmodel
        
        sample_vocoder_model = vocoder_models[1]

        tts_path, tts_config_path, model_item = manager.download_model(
            sample_tts_model)

        # load vocoder
        vocoder_path, vocoder_config_path, _ = manager.download_model(
            sample_vocoder_model)
        self.synthesizer = Synthesizer(
            tts_path, tts_config_path, vocoder_path, vocoder_config_path, False)

        if self.lang == "fr":
            self.lang = "fr-fr"
        elif self.lang == "pt":
            self.lang = "pt-br"

    def synth(self):
        if self.lang in self.synthesizer.tts_model.language_manager.language_names:
            wav = self.synthesizer.tts(
                self.text,  language_name=self.lang, speaker_wav=self.speaker_wav
            )
            self.synthesizer.save_wav(wav, self.out_path)
        elif self.lang == "fa":
            inputs = self.tokenizer_hug(self.text, return_tensors="pt")
            with torch.no_grad():
                output = self.model_hug(**inputs).waveform
            scipy.io.wavfile.write(
                self.out_path, rate=self.model_hug.config.sampling_rate, data=(np.array(output[0])))

        else:
            try:
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")
                ckpt_dir = download(lang=self.lang, tgt_dir=mms_model_path)
                print(f"Run inference with {device}")
                vocab_file = f"{ckpt_dir}/vocab.txt"
                config_file = f"{ckpt_dir}/config.json"
                assert os.path.isfile(
                    config_file), f"{config_file} doesn't exist"
                hps = utils.get_hparams_from_file(config_file)
                text_mapper = TextMapper(vocab_file)
                net_g = SynthesizerTrn(
                    len(text_mapper.symbols),
                    hps.data.filter_length // 2 + 1,
                    hps.train.segment_size // hps.data.hop_length,
                    **hps.model)
                net_g.to(device)
                _ = net_g.eval()

                g_pth = f"{ckpt_dir}/G_100000.pth"
                print(f"load {g_pth}")

                _ = utils.load_checkpoint(g_pth, net_g, None)

                print(f"text: {self.text}")
                txt = preprocess_text(
                    self.text, text_mapper, hps, lang=self.lang)
                stn_tst = text_mapper.get_text(txt, hps)
                with torch.no_grad():
                    x_tst = stn_tst.unsqueeze(0).to(device)
                    x_tst_lengths = torch.LongTensor(
                        [stn_tst.size(0)]).to(device)
                    hyp = net_g.infer(
                        x_tst, x_tst_lengths, noise_scale=.667,
                        noise_scale_w=0.8, length_scale=1.0
                    )[0][0, 0].cpu().float().numpy()

                print(f"Generated audio")
                scipy.io.wavfile.write(
                    self.out_path, rate=self.model_hug.config.sampling_rate, data=hyp)
            except Exception as e:
                print("Your language is not supported. please choose your language from: https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html")
