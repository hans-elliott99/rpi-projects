# ############################################# #
# CONVERT AUDIO FILES OR LIVE RECORDING TO TEXT #
# ############################################# #


import warnings
warnings.simplefilter("ignore", UserWarning) ##to ignore numpy warning, not advised during testing

# For inputs & model
import numpy as np
from tflite_runtime.interpreter import Interpreter 
# For importing audio files with correct format
import librosa
# For live recording
import pyaudio
import wave
# Utilities
import json
from itertools import groupby
import sys
import os
import argparse
import time
# For LEDs
from pixels import pixels as pixels
pixels.off()

#---------------------HELP FUNCTIONS ----------------------------#
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
  #cropping
  if x.shape[0] > MAX_LENGTH:
    n_remove = x.shape[0] - MAX_LENGTH
    x = x[:-n_remove]
  #padding
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

## PIXELS - pixels.py which depends on apa102.py
#---------------------------------------------------------------------#
#----------------------------- MAIN ----------------------------------#
def run(model, vocab, audio_path, duration):
    REQUIRED_SAMPLE_RATE = RESPEAKER_RATE = 16000 ##required samplerate for audio file to work with this model, also happebs to be respeaker's sample rate
    MAX_LENGTH = 246000          ##model performs better when audio sequence is padded to this max length 
    RESPEAKER_CHANNELS = 2
    RESPEAKER_WIDTH = 2
    RESPEAKER_INDEX = 1 #run getDeviceInfo.py to get respeaker index (input device id)
    CHUNK = 1024
    RECORD_SECONDS = int(duration)

    print("initializing model...")
    # initialize label encodings in advance of while loop
    label_encodings = model_to_text(vocab_path=vocab)

    # Initialize tflite interpreter
    interpreter = Interpreter(model_path=model) ##remove 'tf.lite.'
    interpreter.allocate_tensors()
    
    if audio_path==None: ##if no audio file is provided, we are recording live audio
        # Init pyaudio and start live stream
        p = pyaudio.PyAudio()
        stream = p.open(
                    rate=RESPEAKER_RATE,
                    format=p.get_format_from_width(RESPEAKER_WIDTH),
                    channels=RESPEAKER_CHANNELS,
                    input=True,
                    input_device_index=RESPEAKER_INDEX
            )
        while True:
            try:
                pixels.off()
                print("listening...")
                pixels.wakeup() ##start leds
                frames = []
                start = time.time()
                pixels.think()
                for _ in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)): ##take 16000samples per sec * 5secs = 80000 samples, each iter takes 1024 samples so 80000/1024 = ~78 iterations 
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    # to try to improve, save one channel (either 0 with 0::2 or 1 with 1::2)
                    # also have to change the wf.setnchannels to 1, or back to 2 if deleting the next 2 lines
                    a = np.frombuffer(data, dtype=np.int16)[0::2]
                    data = a.tobytes()
                    frames.append(data)
                stop = time.time()
                pixels.off() ##stop leds
                print(f"record time = {round(stop-start, 3)}s")

                # Now save the audio to a temp wav file and load in with librosa
                wav_file = "./tmp.wav"
                with wave.open(wav_file, "w") as wf:
                    wf.setnchannels(1) #RESPEAKER_CHANNELS=2, mono audio=1
                    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
                    wf.setframerate(RESPEAKER_RATE)
                    wf.writeframes(b''.join(frames))
                full_signal, _ = librosa.load(wav_file, sr=REQUIRED_SAMPLE_RATE, mono=True)
                print("signal length =", full_signal.shape[0])
                ### full_signal = np.concatenate([f for f in frames])

                print("transcribing audio...")
                # forward pass the speech signal through model and get encoded predictions
                start = time.time()
                model_output = classify_speech(interpreter, full_signal)
                print(model_output)
                stop = time.time()
                # decode the predictions into text
                text = label_encodings.decode(model_output.tolist(), skip_special_tokens=True, group_tokens=True)
                print(text)
                print(f"Inference time: {round(stop-start, 3)}s")

                stop_words = [g+" "+b for g in ["good", "gud", "god"] for b in ["bye", "by"]]
                if text.lower() in stop_words:
                    raise KeyboardInterrupt

            except KeyboardInterrupt:
                pixels.off()
                try: ##remove temp wav file if it exists
                    os.remove(wav_file)
                except OSError:
                    pass
                print("KeyboardInterrupt: Program ended by user"); sys.exit()


    else: #if audio file is provide
        print("transcribing audio file...")
        full_signal, samplerate = librosa.load(audio_path, sr=REQUIRED_SAMPLE_RATE, mono=True)
        assert samplerate==REQUIRED_SAMPLE_RATE, f"sample rate {samplerate} does not match required sr of {REQUIRED_SAMPLE_RATE}"
        print("signal length =", full_signal.shape[0])

        # run inference
        start = time.time()
        model_output = classify_speech(interpreter, full_signal)
        print(model_output)
        stop = time.time()
        # decode the predictions into text
        text = label_encodings.decode(model_output.tolist(), skip_special_tokens=True, group_tokens=True)
        print(text)
        print(f"Inference time: {round(stop-start, 3)}s")

                                                                                       
                                                                                       
def parse_args_and_run():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--audio', '-a',
        help="path to an audio file if you want one audio file transcribed rather than live audio",
        required=False,
        default=None)
    parser.add_argument(
        '--model', '-m',
        help="path to speech to text model if not in current directory",
        required=False,
        default='./models/quant-wave2vec2-960h.tflite')
    parser.add_argument(
        '--vocab', '-v',
        help="path to vocab file containg index to character mappings if not using default",
        required=False,
        default=None)
    parser.add_argument(
        '--duration', '-d',
        help="duration of live recording",
        required=False,
        default = 8)

    args=parser.parse_args()
    run(model=args.model,
        vocab=args.vocab,
        audio_path=args.audio,
        duration=args.duration
    )


if __name__=='__main__':
    parse_args_and_run()


# So, it works (some how) but this model runs way too slow on raspberry pi's tiny compute - quantization led to 3x speed up (from ~ 45s to ~15s)
# This process will work with other types of models (though some tweaks will need to be made in the audio preprocessing and label decoding phase)




