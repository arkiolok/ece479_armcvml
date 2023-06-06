import pyarmnn as ann
import numpy as np
import cv2
import time
from tflite_runtime.interpreter import Interpreter
import pickle

base_dir = "/home/hakan/Desktop/"

def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

def import_meta_test(meta_name="meta",test_name="test"):
    meta = unpickle(meta_name)
    test = unpickle(test_name)
    return meta, test

if __name__=="__main__":
    meta, test = import_meta_test()
    
def get_fine_label_names(meta):
    fine_label_names = list(map(str,meta[b"fine_label_names"]))
    return fine_label_names

def import_cifar100(base_dir=base_dir):
    x_test = np.load(base_dir + "x_test.npy")
    y_test = np.load(base_dir + "y_test.npy")
    return x_test, y_test

print("Working with Arm NN version", ann.ARMNN_VERSION)

#tflite_path = base_dir + "model_rasnetlarge.tflite"
tflite_path = base_dir+"model_xception1.tflite"
#tflite_path = base_dir + "model2.tflite"
# CIFAR 100 Test Data
x_test, y_test = import_cifar100()

test_n = 1000

# Define PYARMNN Interpreter
def define_armnn_parser(tflite_p):
    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(tflite_p)
    return parser, network

parser, network = define_armnn_parser(tflite_path)

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

def define_tflite_interpreter(tflite_path):
    interpreter = Interpreter(tflite_path)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    return interpreter, height, width


# Get input and output tensors.
interpreter, height, width = define_tflite_interpreter(tflite_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_inference(im, details, interpreter = interpreter, print_input_type = 0, with_perc=False):
  input_details, output_details = details
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
  if with_perc==False: return output.argmax(), latency
  else: return output, latency
print(tflite_inference(x_test[1,:,:,:], (input_details, output_details), print_input_type=1))

# PYARMNN Inference
def pyarmnn_inference(image, with_perc=False):
    input_tensors = ann.make_input_tensors([input_binding_info], [image])
    # Get output binding information for an output layer by using the layer name.
    #output_names = parser.GetSubgraphOutputTensorNames(graph_id)
    #output_binding_info = parser.GetNetworkOutputBindingInfo(0, output_names[0])
    output_tensors = ann.make_output_tensors([output_binding_info])
    runtime.EnqueueWorkload(0, input_tensors, output_tensors)
    results = ann.workload_tensors_to_ndarray(output_tensors)
    if with_perc==False: return results[0].argmax()
    else: return results[0]


if __name__ == "__main__":

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
        pred , latency = tflite_inference(x_test[i,:,:,:], (input_details, output_details))
        lat1_tflite+=latency
        preds_tflite[i] = pred
    lat2_tflite = time.perf_counter() - lat2_tflite
    print("\n|---------------------------------------------------------|")
    print("[PYARMNN] Predictions([:10]) :", preds_pyarmnn[:10])
    print("[PYARMNN] Accuracy:", (y_test[:test_n].T == preds_pyarmnn).sum()/len(preds_pyarmnn))
    print(f"[PYARMNN] Time it took for {len(preds_tflite)} images", lat1_pyarmnn, lat2_pyarmnn)
    print(f"[PYARMNN] Time for a single images (ms)", lat1_pyarmnn*1000/len(preds_pyarmnn), lat2_pyarmnn*1000/len(preds_pyarmnn))
    print("[PYARMNN] FPS", 1/(lat1_pyarmnn/len(preds_pyarmnn)), 1/(lat2_pyarmnn/len(preds_pyarmnn)))
    print("|---------------------------------------------------------|")
    print("[TFLITE] Predictions([:10]) :", preds_tflite[:10])
    print("[TFLITE] Accuracy:", (y_test[:test_n].T == preds_tflite).sum() / len(preds_tflite))
    print(f"[TFLITE] Time it took for {len(preds_tflite)} images", lat1_tflite, lat2_tflite)
    print(f"[TFLITE] Time for a single image (ms)", lat1_tflite*1000/len(preds_tflite), lat2_tflite*1000/len(preds_tflite))
    print("[TFLITE] FPS:", 1/(lat1_tflite/len(preds_tflite)), 1/(lat2_tflite/len(preds_tflite)))
    print("|---------------------------------------------------------|")


