#!/usr/bin/env python3
# coding: utf-8

import os
import json
import time
import tqdm
import pprint
import logging
import tarfile
import urllib
import importlib
import argparse
import sys

import torch
import numpy as np

from models.asr.ctc_decoder import CTCDecoder
from triton_interface import tritonInterface

# TODO: integrating these to remove dependency on jetson_voice library
from jetson_voice import list_audio_devices, AudioInput
from jetson_voice.utils import audio_to_float, softmax

from colorama import init
from termcolor import colored

# pp = pprint.PrettyPrinter(indent=4)
# print("[jetson_voice/tritonASRclient] Printing GPU stats below:")
# print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
# print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
# print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
# print(f"torch.cuda.device(0): {torch.cuda.device(0)}")
# print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
# pp.pprint(f"torch.cuda.memory_stats(0): {torch.cuda.memory_stats(0)}")

class tritonASRclient():
    def __init__(self, args):

        # print(f"[jetson_voice/tritonASRclient] Initialising tritonASRclient")

        # pp = pprint.PrettyPrinter(indent=4)
        # print("[jetson_voice/tritonASRclient] Printing GPU stats below:")
        # print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        # print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        # print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
        # print(f"torch.cuda.device(0): {torch.cuda.device(0)}")
        # print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
        # pp.pprint(f"torch.cuda.memory_stats(0): {torch.cuda.memory_stats(0)}")

        # load arguments
        with open(args.model) as json_file:
            self.config = json.load(json_file)

        ### set some default config options that are non-standard in nemo

        # duration of signal frame, seconds (TODO shorter defaults for VAD/command classifiers)
        # duration of overlap before/after current frame, seconds
        if 'streaming' not in self.config:
            self.config['streaming'] = {
                "frame_length": 1.0,
                "frame_overlap": 0.5
                # "frame_length": 0.1,
                # "frame_overlap": 0.1
            }
        if 'preprocessor' not in self.config:
            self.config['preprocessor'] = {
                "dither": 0.0,
                "pad_to": 0
            }
        # greedy or beamsearch
        # add period to the end of sentences
        if 'ctc_decoder' not in self.config:
            self.config['ctc_decoder'] = {
                "type": "greedy",
                # "add_punctuation": True
                "add_punctuation": False
            }
        if self.config['preprocessor']['features'] == 64:   # TODO normalization coefficients for citrinet (N=80)
            self.config['preprocessor']['normalize'] = {
                "fixed_mean": [
                 -14.95827016, -12.71798736, -11.76067913, -10.83311182,
                 -10.6746914,  -10.15163465, -10.05378331, -9.53918999,
                 -9.41858904,  -9.23382904,  -9.46470918,  -9.56037,
                 -9.57434245,  -9.47498732,  -9.7635205,   -10.08113074,
                 -10.05454561, -9.81112681,  -9.68673603,  -9.83652977,
                 -9.90046248,  -9.85404766,  -9.92560366,  -9.95440354,
                 -10.17162966, -9.90102482,  -9.47471025,  -9.54416855,
                 -10.07109475, -9.98249912,  -9.74359465,  -9.55632283,
                 -9.23399915,  -9.36487649,  -9.81791084,  -9.56799225,
                 -9.70630899,  -9.85148006,  -9.8594418,   -10.01378735,
                 -9.98505315,  -9.62016094,  -10.342285,   -10.41070709,
                 -10.10687659, -10.14536695, -10.30828702, -10.23542833,
                 -10.88546868, -11.31723646, -11.46087382, -11.54877829,
                 -11.62400934, -11.92190509, -12.14063815, -11.65130117,
                 -11.58308531, -12.22214663, -12.42927197, -12.58039805,
                 -13.10098969, -13.14345864, -13.31835645, -14.47345634],
                 "fixed_std": [
                 3.81402054, 4.12647781, 4.05007065, 3.87790987,
                 3.74721178, 3.68377423, 3.69344,    3.54001005,
                 3.59530412, 3.63752368, 3.62826417, 3.56488469,
                 3.53740577, 3.68313898, 3.67138151, 3.55707266,
                 3.54919572, 3.55721289, 3.56723346, 3.46029304,
                 3.44119672, 3.49030548, 3.39328435, 3.28244406,
                 3.28001423, 3.26744937, 3.46692348, 3.35378948,
                 2.96330901, 2.97663111, 3.04575148, 2.89717604,
                 2.95659301, 2.90181116, 2.7111687,  2.93041291,
                 2.86647897, 2.73473181, 2.71495654, 2.75543763,
                 2.79174615, 2.96076456, 2.57376336, 2.68789782,
                 2.90930817, 2.90412004, 2.76187531, 2.89905006,
                 2.65896173, 2.81032176, 2.87769857, 2.84665271,
                 2.80863137, 2.80707634, 2.83752184, 3.01914511,
                 2.92046439, 2.78461139, 2.90034605, 2.94599508,
                 2.99099718, 3.0167554,  3.04649716, 2.94116777]
            }
        # print(f"[jetson_voice/tritonASRclient] self.config: {self.config}")

        ###

        ### create preprocessor instance

        print("[jetson_voice/tritonASRclient] Creating preprocessor instance")
        preprocessor_name = self.config['preprocessor']['_target_'].rsplit(".", 1)
        print(f"[jetson_voice/tritonASRclient] preprocessor_name: {preprocessor_name}, of type: {type(preprocessor_name)}")

        preprocessor_class = getattr(importlib.import_module(preprocessor_name[0]), preprocessor_name[1])
        print("[jetson_voice/tritonASRclient] Arguments to: getattr( importlib.import_module(preprocessor_name[0]) , preprocessor_name[1] )")
        print(f"[jetson_voice/tritonASRclient] where preprocessor_name[0]: {preprocessor_name[0]}, of type: {type(preprocessor_name[0])}")
        print(f"[jetson_voice/tritonASRclient] where preprocessor_name[1]: {preprocessor_name[1]}, of type: {type(preprocessor_name[1])}")
        print(f"[jetson_voice/tritonASRclient] preprocessor_class: {preprocessor_class}, of type: {type(preprocessor_class)}")

        preprocessor_config = self.config['preprocessor'].copy()
        preprocessor_config.pop('_target_')
        print(f"[jetson_voice/tritonASRclient] preprocessor_config after pop: {preprocessor_config}, of type: {type(preprocessor_config)}")
        self.preprocessor = preprocessor_class(**preprocessor_config)
        print(f"[jetson_voice/tritonASRclient] self.preprocessor: {self.preprocessor}, of type: {type(self.preprocessor)}")
        print("[jetson_voice/tritonASRclient] self.preprocessor = preprocessor_class(**preprocessor_config)")

        ###

        ### some core settings

        self.wav = args.wav
        self.mic = args.mic
        self.sample_rate = self.config["sample_rate"]
        self.frame_length = self.config["streaming"]["frame_length"]
        self.frame_overlap = self.config["streaming"]["frame_overlap"]
        self.chunk_size = int(self.frame_length * self.sample_rate)

        ###

        # converter (adapted from load the model)
        self.features = self.config["preprocessor"]["features"]
        self.time_to_fft = self.sample_rate * (1.0 / 160.0)     # rough conversion from samples to MEL spectrogram dims
        self.dynamic_shapes = {
            'min' : (1, self.features, int(0.1 * self.time_to_fft)), # minimum plausible frame length
            'opt' : (1, self.features, int(1.5 * self.time_to_fft)), # default of .5s overlap factor (1,64,121)
            'max' : (1, self.features, int(3.0 * self.time_to_fft))  # enough for 2s overlap factor
        }
        #

        # create CTC decoder
        print("[jetson_voice/tritonASRclient] Creating CTC decoder")
        self.ctc_decoder = CTCDecoder.from_config( 
            self.config['ctc_decoder'], 
            self.config['decoder']['vocabulary'], 
            os.path.dirname(self.config["model_path"]) 
            )                                      
        print(f"[jetson_voice/tritonASRclient] Creating CTC decoder in resource.py: {self.ctc_decoder}, of type: {self.ctc_decoder.type}")
        #

        # create streaming buffer
        print("[jetson_voice/tritonASRclient] Loading configs for streaming buffer")
        self.n_frame_len = int(self.frame_length * self.sample_rate)
        self.n_frame_overlap = int(self.frame_overlap * self.sample_rate)
        self.buffer_length = self.n_frame_len + self.n_frame_overlap
        self.buffer_duration = self.buffer_length / self.sample_rate
        self.buffer = np.zeros(shape=self.buffer_length, dtype=np.float32)  # 2*self.n_frame_overlap
        print(f"[jetson_voice/tritonASRclient] self.n_frame_len: {self.n_frame_len}, type: {self.n_frame_len}")
        print(f"[jetson_voice/tritonASRclient] self.n_frame_overlap: {self.n_frame_overlap}, type: {self.n_frame_overlap}")
        print(f"[jetson_voice/tritonASRclient] self.buffer_length: {self.buffer_length}, type: {self.buffer_length}")
        print(f"[jetson_voice/tritonASRclient] self.buffer_duration: {self.buffer_duration}, type: {self.buffer_duration}")
        print(f"[jetson_voice/tritonASRclient] self.buffer: {self.buffer}, type: {self.buffer}")
        #

        # print(f"[jetson_voice/tritonASRclient] Finished initialising tritonASRclient")

        # pp = pprint.PrettyPrinter(indent=4)
        # print("[jetson_voice/tritonASRclient] Printing GPU stats below:")
        # print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        # print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        # print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
        # print(f"torch.cuda.device(0): {torch.cuda.device(0)}")
        # print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
        # pp.pprint(f"torch.cuda.memory_stats(0): {torch.cuda.memory_stats(0)}")

    def preprocess(self, samples):
        """
        Transcribe streaming audio samples to text, returning the running phrase.
        Phrases are broken up when a break in the audio is detected (i.e. end of sentence)
        
        Parameters:
          samples (array) -- Numpy array of audio samples.

        Returns a dict of the running phrase.
          transcript (string) -- the current transcript
          latest (string) -- the latest additions to the transcript
          end (bool) -- if true, end-of-sequence due to silence
        """

        samples = audio_to_float(samples)

        if len(samples) < self.n_frame_len:
            samples = np.pad(samples, [0, self.n_frame_len - len(samples)], 'constant')
            
        self.buffer[:self.n_frame_overlap] = self.buffer[-self.n_frame_overlap:]
        self.buffer[self.n_frame_overlap:] = samples

        # apply pre-processing
        preprocessed_signal, _ = self.preprocessor(
            input_signal=torch.as_tensor(self.buffer, dtype=torch.float32).unsqueeze(dim=0), 
            length=torch.as_tensor(self.buffer.size, dtype=torch.int64).unsqueeze(dim=0)
        )
        return preprocessed_signal

    def decodeOutput(self, logits):
        # # run the asr model
        # logits = self.model.execute(torch_to_numpy(preprocessed_signal))
        logits = np.squeeze(logits)
        # print(f"[jetson_voice/tritonASRclient] logits.shape after squeeze: {logits.shape}, type(logits) after squeeze: {type(logits)}")
        logits = softmax(logits, axis=-1)
        # print(f"[jetson_voice/tritonASRclient] logits.shape after softmax: {logits.shape}, type(logits) after softmax: {type(logits)}")
        
        self.timestep_duration = self.buffer_duration / logits.shape[0]
        self.n_timesteps_frame = int(self.frame_length / self.timestep_duration)
        self.n_timesteps_overlap = int(self.frame_overlap / self.timestep_duration)

        # print(f"[jetson_voice/tritonASRclient] setting ctc_decoder set_timestep_duration as self.timestep_duration:{self.timestep_duration}")
        # print(f"[jetson_voice/tritonASRclient] setting ctc_decoder set_timestep_delta as self.n_timesteps_frame:{self.n_timesteps_frame}")
        self.ctc_decoder.set_timestep_duration(self.timestep_duration)
        self.ctc_decoder.set_timestep_delta(self.n_timesteps_frame)

        transcripts = self.ctc_decoder.decode(logits)
        # print(f"[jetson_voice/tritonASRclient] Transcripts: {transcripts}")
        # print(f"[jetson_voice/tritonASRclient] Transcripts length: {len(transcripts)}")
        # print(f"[jetson_voice/tritonASRclient] Transcripts of type: {type(transcripts)}")

        return transcripts

        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='quartznet', type=str, help='path to model, service name, or json config file')
    parser.add_argument('--wav', default=None, type=str, help='path to input wav file')
    parser.add_argument('--mic', default=None, type=str, help='device name or number of input microphone')
    parser.add_argument('--list-devices', action='store_true', help='list audio input devices')
    args = parser.parse_args()
    print(args)
    
    # list audio devices
    if args.list_devices:
        list_audio_devices()
        sys.exit()

    # for terminal colors
    init()

    # initialise asr client
    asrClient = tritonASRclient(args)
    
    # create the audio input stream
    stream = AudioInput(wav=asrClient.wav, mic=asrClient.mic, 
                         sample_rate=asrClient.sample_rate, 
                         chunk_size=asrClient.chunk_size)

    # initialise grpc client
    interface = tritonInterface()

    # run transcription
    for samples in stream:

        # print(f"[jetson_voice/tritonASRclient] len(samples): {len(samples)}, type(samples): {type(samples)}")
        input_samples = asrClient.preprocess(samples=samples)

        # print(f"[jetson_voice/tritonASRclient] len(input_samples): {len(input_samples)}, type(input_samples): {type(input_samples)}")
        output_samples = interface.streaming_asr(input_samples=input_samples)

        # print(f"[jetson_voice/tritonASRclient] len(output_samples): {len(output_samples)}, type(output_samples): {type(output_samples)}")
        transcripts = asrClient.decodeOutput(logits=output_samples)

        if len(transcripts[0]['text']) > 0:
            if transcripts[0]['end']:
                print(colored("{}".format(transcripts[0]['text']), "yellow"))
            else:
                print(transcripts[0]['text'])

    # # run transcription
    # for samples in stream:
    #     #samples = audio_to_float(samples)
    #     #print(f'samples {samples.shape} ({audio_db(samples):.1f} dB)')
    #     input_samples = asr(samples)
        
    #     if asr.classification:
    #         print(f"class '{input_samples[0]}' ({input_samples[1]:.3f})")
    #     else:
    #         for transcript in input_samples:
    #             print(transcript['text'])
                
    #             if transcript['end']:
    #                 print('')
                    
    print('\naudio stream closed.')
    
    # pp = pprint.PrettyPrinter(indent=4)
    # print("[jetson_voice/tritonASRclient] Printing GPU stats below:")
    # print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    # print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    # print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
    # print(f"torch.cuda.device(0): {torch.cuda.device(0)}")
    # print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    # pp.pprint(f"torch.cuda.memory_stats(0): {torch.cuda.memory_stats(0)}")