#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time

class ONNXClassifierWrapper():
    def __init__(self, file, num_classes, target_dtype=np.float32):

        self.target_dtype = target_dtype
        self.num_classes = num_classes
        self.load(file)

        self.stream = None

    def load(self, file):
        f = open(file, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()

    def allocate_memory(self, batch):
        self.output = np.empty(self.num_classes, dtype=self.target_dtype) # Need to set both input and output precisions to FP16 to fully enable FP16

        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()

    def predict(self, batch, eval_exec_time = False): # result gets copied into output
        if self.stream is None:
            self.allocate_memory(batch)

        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)

        # Execute model
        if eval_exec_time:
            t_start = time.time()
        # self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        if eval_exec_time:
            t_inference = time.time() - t_start
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()

        return (t_inference, self.output) if eval_exec_time else self.output

def convert_trt(onnx_filename, trt_filename, half):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_filename, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ Failed to parse ONNX model.")
            for error_idx in range(parser.num_errors):
                print(parser.get_error(error_idx))
            raise RuntimeError("❌ ONNX parsing failed.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)  # 3GB
    if half:
        config.set_flag(trt.BuilderFlag.FP16)

    print("🛠️ Building TensorRT engine...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("❌ TensorRT build failed.")

    with open(trt_filename, "wb") as f:
        f.write(engine)
    print(f"✅ Engine saved at {trt_filename}")
