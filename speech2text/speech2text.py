# ################################ #
# CONVERT AUDIO FILES TO TEXT #
# ################################ #


import warnings
warnings.simplefilter("ignore", UserWarning) ##to ignore numpy warning, not advised during testing

# For inputs & model
import numpy as np
from tflite_runtime.interpreter import Interpreter 
# For importing audio files with correct sample rate
import librosa
# Utilities
import json
from itertools import groupby
import sys
import argparse
import time




REQUIRED_SAMPLE_RATE = 16000 ##required samplerate for audio file to work with this model
MAX_LENGTH = 246000          ##model performs better when audio sequence is padded to this max length 


#----------------------- HELP FUNCTIONS ----------------------------#
## AUDIO PREP AND CLASSIFICATION
def normalize_pad(x, pad=True):
  """
  Normalize and pad input signal to match preprocessing of the model.
  Methodology from: https://github.com/thevasudevgupta/gsoc-wav2vec2/blob/main/src/wav2vec2/processor.py
  """
  MAX_LENGTH = 246000 ##https://tfhub.dev/vasudevgupta7/wav2vec2-960h/1
  #normalize()
  mean = np.mean(x, axis=-1, keepdims=True)
  var = np.var(x, axis=-1, keepdims=True)
  x = np.squeeze((x - mean) / np.sqrt(var + 1e-5))
  #pad
  if pad:
    padding = np.zeros(MAX_LENGTH - x.shape[0])
    x = np.concatenate((x, padding))
  return x

def resize_input_seq(interpreter, speech):
  "Resize the input signal to the size that the model will accept"
  _, seq_length = interpreter.get_input_details()[0]['shape']
  speech = np.resize(speech, (1, seq_length))
  return speech

def set_input_tensor(interpreter, speech):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:] = speech

def classify_speech(interpreter, speech):
  speech = normalize_pad(speech, pad=True)
  speech = resize_input_seq(interpreter, speech)
  
  set_input_tensor(interpreter, speech)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))
  
  return np.squeeze(np.argmax(output, axis=-1))    


## MODEL OUTPUT TO TEXT MAPPINGS
class model_to_text:
    def __init__(self, vocab_path):
        self.create_mappings(vocab_path)

    def create_mappings(self, vocab_path):
        try:
            self.token_to_id_mapping = self._get_vocab(vocab_path)
        except:
            print("failed to load 'vocab.json'")
            sys.exit()

        self.id_to_token_mapping = {v: k for k, v in self.token_to_id_mapping.items()}

        self.unk_token = "<unk>"
        self.unk_id = self.token_to_id_mapping[self.unk_token]

        self.dimiliter_token = "|"
        self.dimiliter_id = self.token_to_id_mapping[self.dimiliter_token]

        self.special_tokens = ["<pad>"]
        self.special_ids = [self.token_to_id_mapping[k] for k in self.special_tokens]

    def _get_vocab(self, vocab_path):
        if vocab_path is not None: #isinstance(VOCAB, str):
            with open(vocab_path, "r") as f:
                vocab = json.load(f)
        else:
            vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6, "A": 7, "O": 8, "N": 9, "I": 10, "H": 11, "S": 12, "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18, "C": 19, "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26, "'": 27, "X": 28, "J": 29, "Q": 30, "Z": 31}
        return vocab

    def decode(self, input_ids: list, skip_special_tokens=True, group_tokens=True):
        """
        Use this method to decode your ids back to string.
        Args:
            input_ids (:obj: `list`):
                input_ids you want to decode to string.
            skip_special_tokens (:obj: `bool`, `optional`):
                Whether to remove special tokens (like `<pad>`) from string.
            group_tokens (:obj: `bool`, `optional`):
                Whether to group repeated characters.
        """
        if group_tokens:
            input_ids = [t[0] for t in groupby(input_ids)]
        if skip_special_tokens:
            input_ids = [k for k in input_ids if k not in self.special_ids]
        tokens = [self.id_to_token_mapping.get(k, self.unk_token) for k in input_ids]
        tokens = [k if k != self.dimiliter_token else " " for k in tokens]
        return "".join(tokens).strip()


#----------------------------- MAIN ----------------------------------#
def run(model, vocab, audio_path):
    
    print("initializing model...")
    # initialize label encodings in advance of while loop
    label_encodings = model_to_text(vocab_path=vocab)
    
    # Initialize tflite interpreter
    interpreter = Interpreter(model_path=model) ##remove 'tf.lite.'
    interpreter.allocate_tensors()

    print("transcribing audio...")
    while True:
        # Load file first to determine if specs are correct
        signal, samplerate = librosa.load(audio_path, sr=REQUIRED_SAMPLE_RATE, mono=True)
        assert samplerate==REQUIRED_SAMPLE_RATE, f"sample rate {sample_rate} does not match required sr of {REQUIRED_SAMPLE_RATE}"

        # forward pass the speech signal through model and get encoded predictions
        start = time.time()
        model_output = classify_speech(interpreter, signal)
        end = time.time()
        # decode the predictions into text
        text = label_encodings.decode(model_output.tolist(), skip_special_tokens=True, group_tokens=True)
        print(text)
        print(f"Inference time: {round(end-start, 3)}s")
        sys.exit() ##temp


def parse_args_and_run():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--audio', '-a',
        help="path to an audio file if you want one short audio file transcribed",
        required=False,
        default=None)
    parser.add_argument(
        '--model', '-m',
        help="path to speech to text model",
        required=False,
        default='./wave2vec2-960h.tflite')
    parser.add_argument(
        '--vocab', '-v',
        help="path to vocab file containg index to character mappings",
        required=False,
        default=None)

    args=parser.parse_args()
    
    run(model=args.model, 
        vocab=args.vocab,
        audio_path=args.audio)
    


if __name__=='__main__':
    parse_args_and_run()
    




