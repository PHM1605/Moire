import onnxruntime as ort, glob
from haarDwt import create_input
import numpy as np

session = ort.InferenceSession('fake_detect.onnx')

for file in glob.glob('testImages/*.jpg'):
    [wavelet1, wavelet2] = create_input(file, (256, 256))
    wavelet1 = np.ascontiguousarray(np.expand_dims(wavelet1, 0)).astype(np.float32)/255.0
    wavelet2 = np.ascontiguousarray(np.expand_dims(wavelet2, 0)).astype(np.float32)/255.0
    onnx_inputs = {
    session.get_inputs()[0].name: wavelet1,
    session.get_inputs()[1].name: wavelet2
    }
    onnx_outputs = session.run(None, onnx_inputs)[0]
    print(np.argmax(onnx_outputs[0]))