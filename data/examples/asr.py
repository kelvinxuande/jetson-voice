#!/usr/bin/env python3
# coding: utf-8

# standard library import
import sys
# import jetson_voice library
from jetson_voice import ASR, AudioInput, ConfigArgParser, list_audio_devices

# parse some configs
parser = ConfigArgParser()
parser.add_argument('--model', default='quartznet', type=str, help='path to model, service name, or json config file')
parser.add_argument('--wav', default=None, type=str, help='path to input wav/ogg/flac file')
parser.add_argument('--mic', default=None, type=str, help='device name or number of input microphone')
parser.add_argument('--list-devices', action='store_true', help='list audio input devices')
args = parser.parse_args()
print("[asr.py] Running asr.py within examples directory, with the following arguments:")
print(args)
    
# list audio devices
if args.list_devices:
    print("[asr.py>jetson_voice/utils/audio.py] Calling list_audio_devices()")
    list_audio_devices()
    sys.exit()
    
# load the model by passing in a str
print("[asr.py>jetson_voice/asr.py] Loading ASR model, passing args.model to ASR object")
asr = ASR(args.model)

print("[asr.py>jetson_voice/utils/audio.py] Creating audio input stream, ")
print(f"[asr.py>jetson_voice/utils/audio.py] wav={args.wav}, mic={args.mic}, sample_rate={asr.sample_rate}, chunk_size={asr.chunk_size}")
# create the audio input stream
stream = AudioInput(wav=args.wav, mic=args.mic, 
                     sample_rate=asr.sample_rate, 
                     chunk_size=asr.chunk_size)

# run transcription
for samples in stream:
    results = asr(samples)
    
    if asr.classification:
        print(f"class '{results[0]}' ({results[1]:.3f})")
    else:
        for transcript in results:
            print(transcript['text'])
            
            if transcript['end']:
                print('')
                
print('\naudio stream closed.')
    