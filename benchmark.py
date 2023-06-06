import pyarmnn as ann
import numpy as np
import cv2
import time
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import Interpreter

print("Working with Arm NN version", ann.ARMNN_VERSION)
base_dir = "/home/hakan/Desktop/"


model_id = 1 #Change this number to switch between models.
model_name = ["MobileNetV2.tflite",'Xception.tflite', 'model_nasnetlarge.tflite'][model_id]



tflite_path = base_dir + model_name
# CIFAR 100 Test Data
x_test = np.load(base_dir + "x_test.npy")
y_test = np.load(base_dir + "y_test.npy")
test_n = 1000

# Define PYARMNN Interpreter
parser = ann.ITfLiteParser()
network = parser.CreateNetworkFromBinaryFile(tflite_path)

if True:
    graph_id = 0
    input_names = parser.GetSubgraphInputTensorNames(graph_id)
    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
    input_tensor_id = input_binding_info[0]
    input_tensor_info = input_binding_info[1]
    print('tensor id: ' + str(input_tensor_id))
    print('tensor info: ' + str(input_tensor_info))
    # Create a runtime object that will perform inference.
    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    # Backend choices earlier in the list have higher preference.
    preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
    opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())
    # Load the optimized network into the runtime.
    net_id, _ = runtime.LoadNetwork(opt_network)
    print(f"Loaded network, id={net_id}")
    # Create an inputTensor for inference.
    output_names = parser.GetSubgraphOutputTensorNames(graph_id)
    output_binding_info = parser.GetNetworkOutputBindingInfo(0, output_names[0])

# Define TFLite Interpreter
interpreter = Interpreter(tflite_path)
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']

# Define Coral Interpreter
coral_interpreter = tflite.Interpreter(model_path=tflite_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
coral_interpreter.allocate_tensors()



# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_inference(im, interpreter = interpreter, print_input_type = 0):

  # Allocate tensors
  interpreter.allocate_tensors()
  # If the expected input type is int8 (quantized model), rescale data
  input_type = input_details[0]['dtype']
  if print_input_type: print(input_type)
  input_shape = input_details[0]["shape"]
  im1 = im.astype(input_type)
  #print(input_type)
  start = time.perf_counter()
  interpreter.set_tensor(input_details[0]['index'], im1.reshape(input_shape))

  # Run inference
  interpreter.invoke()

  # output_details[0]['index'] = the index which provides the input
  output = interpreter.get_tensor(output_details[0]['index'])
  latency = time.perf_counter() - start
  return output.argmax(), latency
print(tflite_inference(x_test[1,:,:,:], print_input_type=1))
print(tflite_inference(x_test[1,:,:,:],interpreter=coral_interpreter, print_input_type=1))

# PYARMNN Inference
def pyarmnn_inference(image):
    input_tensors = ann.make_input_tensors([input_binding_info], [image])

    # Get output binding information for an output layer by using the layer name.
    #output_names = parser.GetSubgraphOutputTensorNames(graph_id)
    #output_binding_info = parser.GetNetworkOutputBindingInfo(0, output_names[0])
    output_tensors = ann.make_output_tensors([output_binding_info])
    runtime.EnqueueWorkload(0, input_tensors, output_tensors)
    results = ann.workload_tensors_to_ndarray(output_tensors)
    return results[0].argmax()

print(pyarmnn_inference(x_test[1,:,:,:]), y_test[1])

print(pyarmnn_inference(x_test[2,:,:,:]), y_test[2])

preds_pyarmnn = np.zeros(test_n)
lat2_pyarmnn = time.perf_counter()
lat1_pyarmnn = 0
for i in range(test_n):
    if i % int(test_n/5)  == 0: print(f"[PYARMNN] Test: {i}/{test_n}")
    start1 = time.perf_counter()
    pred = pyarmnn_inference(x_test[i,:,:,:])
    end1 = time.perf_counter()
    lat1_pyarmnn += (end1-start1)
    preds_pyarmnn[i] = pred
lat2_pyarmnn = time.perf_counter() - lat2_pyarmnn

preds_tflite = np.zeros(test_n)
lat2_tflite = time.perf_counter()
lat1_tflite = 0
for i in range(test_n):
    if i%int(test_n/5) == 0: print(f"[TFLITE] Test: {i}/{test_n}")
    pred , latency = tflite_inference(x_test[i,:,:,:])
    lat1_tflite+=latency
    preds_tflite[i] = pred
lat2_tflite = time.perf_counter() - lat2_tflite

preds_coral = np.zeros(test_n)
lat2_coral = time.perf_counter()
lat1_coral = 0
for i in range(test_n):
    if i%int(test_n/5) == 0: print(f"[CORAL] Test: {i}/{test_n}")
    pred , latency = tflite_inference(x_test[i,:,:,:], interpreter=coral_interpreter)
    lat1_coral+=latency
    preds_coral[i] = pred
lat2_coral = time.perf_counter() - lat2_coral
print("Model Name:", model_name)
print("\n|---------------------------------------------------------|")
print("[PYARMNN] Predictions([:10]) :", preds_pyarmnn[:10])
print("[PYARMNN] Accuracy:", (y_test[:test_n].T == preds_pyarmnn).sum()/len(preds_pyarmnn))
print(f"[PYARMNN] Time it took for {len(preds_tflite)} images", lat1_pyarmnn)#, lat2_pyarmnn)
print(f"[PYARMNN] Time for a single images (ms)", lat1_pyarmnn*1000/len(preds_pyarmnn))#, lat2_pyarmnn*1000/len(preds_pyarmnn))
print("[PYARMNN] FPS", 1/(lat1_pyarmnn/len(preds_pyarmnn)))#, 1/(lat2_pyarmnn/len(preds_pyarmnn)))
print("|---------------------------------------------------------|")
print("[TFLITE] Predictions([:10]) :", preds_tflite[:10])
print("[TFLITE] Accuracy:", (y_test[:test_n].T == preds_tflite).sum() / len(preds_tflite))
print(f"[TFLITE] Time it took for {len(preds_tflite)} images", lat1_tflite)#, lat2_tflite)
print(f"[TFLITE] Time for a single image (ms)", lat1_tflite*1000/len(preds_tflite))#, lat2_tflite*1000/len(preds_tflite))
print("[TFLITE] FPS:", 1/(lat1_tflite/len(preds_tflite)))#, 1/(lat2_tflite/len(preds_tflite)))
print("|---------------------------------------------------------|")
print("[CORAL] Predictions([:10]) :", preds_coral[:10])
print("[CORAL] Accuracy:", (y_test[:test_n].T == preds_coral).sum() / len(preds_coral))
print(f"[CORAL] Time it took for {len(preds_coral)} images", lat1_coral)#, lat2_coral)
print(f"[CORAL] Time for a single image (ms)", lat1_coral*1000/len(preds_coral))#, lat2_coral*1000/len(preds_coral))
print("[CORAL] FPS:", 1/(lat1_coral/len(preds_coral)))#, 1/(lat2_coral/len(preds_coral)))
print("|---------------------------------------------------------|")


