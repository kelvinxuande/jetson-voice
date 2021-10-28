#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import sys
import struct
from functools import partial
import random

import numpy as np
import grpc
import torch
from jetson_voice.utils import softmax
from termcolor import colored

import tritonclient
from tritonclient.grpc import *
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc

from tritonclient.utils import InferenceServerException

class tritonInterface:

    def __init__(self, ctc_decoder, buffer_duration, frame_length, frame_overlap):
        self.flags = {
            "verbose": False,
            "async": False,
            "streaming": True,
            "model-name": "test_stt_en_quartznet15x5",
            "model-version": "1",
            "batch-size": 1,
            "url": "localhost:8001",
        }

        self.ctc_decoder = ctc_decoder
        self.buffer_duration = buffer_duration
        self.frame_length = frame_length
        self.frame_overlap = frame_overlap

        self.response_modelconfigrequest=None
        self.buffer = []

        ### create grpc stub for communicating with the server
        if self.flags["streaming"]:
            self.correlation_id = random.randint(1,2**31-1)

            # Create gRPC stub for communicating with the server
            print(f'[jetson_voice/tritonInterface] opening GRPC channel: {str(self.flags["url"])}')
            try:
                channel = grpc.insecure_channel(self.flags["url"],
                                options=[('grpc.keepalive_time_ms', 3000),
                                        ('grpc.keepalive_timeout_ms', 3000),
                                        ('grpc.keepalive_permit_without_calls', True)])
                self.grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
                print(f"[jetson_voice/tritonInterface] self.grpc_stub: {self.grpc_stub}")
            except Exception as e:
                print(f"[jetson_voice/tritonInterface] self.grpc_stub ERROR: {e}")
            #

            ### Health
            # check server live
            try:
                request_serverliverequest = service_pb2.ServerLiveRequest()
                response_serverliverequest = self.grpc_stub.ServerLive(request_serverliverequest)
                print(f"[jetson_voice/tritonInterface] response_serverliverequest: {response_serverliverequest}")
            except Exception as e:
                print(f"[jetson_voice/tritonInterface] response_serverliverequest ERROR: {e}")
            # check server ready
            try:
                request_serverreadyrequest = service_pb2.ServerReadyRequest()
                response_serverreadyrequest = self.grpc_stub.ServerReady(request_serverreadyrequest)
                print(f"[jetson_voice/tritonInterface] response_serverreadyrequest: {response_serverreadyrequest}")
            except Exception as e:
                print(f"[jetson_voice/tritonInterface] response_serverreadyrequest ERROR: {e}")
            # check model ready
            try:
                request_modelreadyrequest = service_pb2.ModelReadyRequest(name=self.flags["model-name"], version=self.flags["model-version"])
                response_modelreadyrequest = self.grpc_stub.ModelReady(request_modelreadyrequest)
                print(f"[jetson_voice/tritonInterface] response_modelreadyrequest: {response_modelreadyrequest}")
            except Exception as e:
                print(f"[jetson_voice/tritonInterface] response_modelreadyrequest ERROR: {e}")
            ###

            # Metadata
            try:
                request_servermetadatarequest = service_pb2.ServerMetadataRequest()
                response_servermetadatarequest = self.grpc_stub.ServerMetadata(request_servermetadatarequest)
                print(f"[jetson_voice/tritonInterface] server metadata:\n{response_servermetadatarequest}")
                request_modelmetadatarequest = service_pb2.ModelMetadataRequest(name=self.flags["model-name"], version=self.flags["model-version"])
                response_modelmetadatarequest = self.grpc_stub.ModelMetadata(request_modelmetadatarequest)
                print(f"[jetson_voice/tritonInterface] model metadata:\n{response_modelmetadatarequest}")
            except Exception as e:
                print(f"[jetson_voice/tritonInterface] server metadata/ model metadata ERROR:{e}")

            # Configuration
            try:
                request_modelconfigrequest = service_pb2.ModelConfigRequest(name=self.flags["model-name"], version=self.flags["model-version"])
                response_modelconfigrequest = self.grpc_stub.ModelConfig(request_modelconfigrequest)
                print(f"[jetson_voice/tritonInterface] model config:\n{response_modelconfigrequest}")
                self.response_modelconfigrequest = response_modelconfigrequest
            except Exception as e:
                print(f"[jetson_voice/tritonInterface] response_modelconfigrequest ERROR: {e}")

        else:
            # self.ctx = InferContext(self.flags["url"], protocol, self.flags["model-name"], self.flags["model-version"],
            #                         verbose, self.correlation_id, False)
            # server_ctx = ServerStatusContext(self.flags["url"], protocol, self.flags["model-name"],
            #                                  verbose)
            # # server_status = server_ctx.get_server_status()
            pass
        ###

    def torch_to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def callback(self, result, error):
        if error:
            if type(error) == InferenceServerException:
                print(f"[jetson_voice/tritonInterface] Caught InferenceServerException: {error}")
            else:
                print(f"[jetson_voice/tritonInterface] Unknown error in callback: {error}")
        else:
            # error=None, results of an inference as grpcclient.InferResult in result
            logits = result.as_numpy(name="logprobs") # aka InferResultNumpy

            # # run the asr model
            # logits = self.model.execute(torch_to_numpy(preprocessed_signal))
            logits = np.squeeze(logits)
            # print(f"[jetson_voice/tritonASRclient] logits.shape after squeeze: {logits.shape}, type(logits) after squeeze: {type(logits)}")
            logits = softmax(logits, axis=-1)
            # print(f"[jetson_voice/tritonASRclient] logits.shape after softmax: {logits.shape}, type(logits) after softmax: {type(logits)}")


            # set some parameters for the CTC decoder
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

            if len(transcripts[0]['text']) > 0:
                if transcripts[0]['end']:
                    print(colored("{}".format(transcripts[0]['text']), 'yellow'))
                else:
                    print(transcripts[0]['text'])


    def streaming_asr(self, input_samples):
        # TODO, good reference: https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/grpc/__init__.py

        # converting torch tensor to numpy array
        input_samples = self.torch_to_numpy(input_samples)
        # print(f"[jetson_voice/tritonInterface] input_samples.shape: {input_samples.shape}, type(input_samples): {type(input_samples)}")

        # initialise a new triton Inference Server Client
        # An InferenceServerClient object is used to perform any kind of
        # communication with the InferenceServer using gRPC protocol. Most
        # of the methods are thread-safe except start_stream, stop_stream
        # and async_stream_infer. Accessing a client stream with different
        # threads will cause undefined behavior.
        client = tritonclient.grpc.InferenceServerClient(url=self.flags["url"])
        # after this, we will call the various inference methods

        # initialise an object of InferInput class is used to describe input tensor for an inference request
        input_tensor = tritonclient.grpc.InferInput(name='audio_signal', shape=list(input_samples.shape), datatype='FP32')
        input_tensor.set_data_from_numpy(input_tensor=input_samples)

        # NOT NEEDED
        # An object of InferRequestedOutput class is used to describe a
        # requested output tensor for an inference request.
        # output_tensor = tritonclient.grpc.InferRequestedOutput(name='logprobs', class_count=26)

        # Inference call
        client.async_infer( 
            self.flags["model-name"], 
            inputs=[input_tensor], 
            callback=partial(self.callback), 
            client_timeout=3000)
        # InferResult = client.infer(
        #     model_name=self.flags["model-name"],
        #     inputs=[input_tensor],
        #     model_version=self.flags["model-version"],
        # )



        # NOT NEEDED, for debugging response
        # InferResultResponse = InferResult.get_response(as_json=True)
        # print(f"InferResultResponse: {InferResultResponse}, type(InferResultResponse): {type(InferResultResponse)}")

        # InferResultOutput = InferResult.get_output(name="logprobs", as_json=True)
        # print(f"InferResultOutput: {InferResultOutput}, type(InferResultOutput): {type(InferResultOutput)}")
        



        # InferResultNumpy = InferResult.as_numpy(name="logprobs")
        # # print(f"[jetson_voice/tritonInterface] InferResultNumpy.shape: {InferResultNumpy.shape}, type(InferResultNumpy): {type(InferResultNumpy)}")

        # return InferResultNumpy # as output_samples