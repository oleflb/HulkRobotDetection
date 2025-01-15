import argparse
import numpy as np
import onnxruntime as ort
import time
from tqdm import tqdm
import openvino as ov

def make_measurement(type: str, inference_function, shape, repeat: int):
    inference_times = []
    for _ in tqdm(range(repeat)):
        input_values = np.random.randn(*shape).astype(np.float32)
        start = time.time()
        _ = inference_function(input_values)
        end = time.time()
        inference_times.append(end - start)

    print(f"{type}:")
    print(f"μ: {1000 * np.mean(inference_times):.2f}ms")
    print(f"𝛔: ±{1000 * np.std(inference_times):.2f}ms")
    print(f"[min, max]: [{1000 * np.min(inference_times):.2f}ms, {1000 * np.max(inference_times):.2f}ms]")

def main(args):
    model = ort.InferenceSession(args.onnx)
    input_layer = model.get_inputs()[0]
    output_layer = model.get_outputs()[0]

    def onnx_run(input_values):
        return model.run([output_layer.name], {input_layer.name: input_values})

    make_measurement("onnxruntime", onnx_run, input_layer.shape, repeat=int(args.repeat))


    ov_model = args.onnx # ov.convert_model(args.onnx)
    core = ov.Core()
    compiled_model = core.compile_model(model=ov_model, device_name="CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    def openvino_run(input_values):
        return compiled_model([input_values])[output_layer]
    
    make_measurement("openvino", openvino_run, input_layer.shape, repeat=int(args.repeat))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', help="the onnx model path")
    parser.add_argument('--repeat', default=10, help="the onnx model path")
    main(parser.parse_args())
