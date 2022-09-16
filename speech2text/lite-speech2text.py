# ############################################################# #
# CONVERT AUDIO FILES OR LIVE RECORDING TO TEXT - LIGHTER MODEL #
# ############################################################# #


import warnings
warnings.simplefilter("ignore", UserWarning) ##to ignore numpy warning, not advised during testing

# For inputs & model
import numpy as np
from tflite_runtime.interpreter import Interpreter 
import tflite_support
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
# def resize_input_seq(interpreter, speech):
#   "Resize the input signal to the size that the model will accept"
#   _, seq_length = interpreter.get_input_details()[0]['shape']
#   speech = np.resize(speech, (1, seq_length))
#   return speech

def set_tensors(interpreter, signal):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(input_details[0]["index"], signal.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], signal)
    interpreter.set_tensor(
        input_details[1]["index"],
        np.array(0).astype('int32')
    )
    interpreter.set_tensor(
        input_details[2]["index"],
        np.zeros([1,2,1,320]).astype('float32')
    )
    return input_details, output_details


def classify_speech(interpreter, speech):
    #speech = resize_input_seq(interpreter, speech)

    input_details, output_details = set_tensors(interpreter, speech)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output    


## MODEL OUTPUT TO TEXT MAPPINGS
class model_to_text:
    def __init__(self):
        self._create_mappings()

    def _create_mappings(self):
        self.token_to_id_mapping = self._get_vocab() #character to ascii int

        self.id_to_token_mapping = {v: k for k, v in self.token_to_id_mapping.items()} #ascii int to character

        self.special_tokens = ["<pad>"]
        self.special_ids = [self.token_to_id_mapping[k] for k in self.special_tokens]

    def _get_vocab(self):
        alph = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        ascii_int = [ord(c) for c in alph] 
        vocab = {c : i for i,c in zip(ascii_int, alph)} ##token to id mapping
        vocab["<pad>"] = 0
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
        tokens = [self.id_to_token_mapping.get(k, "<unk>") for k in input_ids]
        tokens = [k if k not in self.special_tokens else "" for k in tokens]
        return "".join(tokens).strip()

## PIXELS - pixels.py which depends on apa102.py
#---------------------------------------------------------------------#
#----------------------------- MAIN ----------------------------------#
def run(model, vocab, audio_path, duration):
    REQUIRED_SAMPLE_RATE = RESPEAKER_RATE = 16000 ##required samplerate for audio file to work with this model, also happebs to be respeaker's sample rate
    RESPEAKER_CHANNELS = 2
    RESPEAKER_WIDTH = 2
    RESPEAKER_INDEX = 1 #run getDeviceInfo.py to get respeaker index (input device id)
    CHUNK = 1024
    RECORD_SECONDS = int(duration)

    print("initializing model...")
    # initialize label encodings in advance of while loop
    label_encodings = model_to_text()

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
        default='./models/lite-model_ASR_TFLite_pre_trained_models_English_1.tflite') 
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





