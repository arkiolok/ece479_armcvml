import labels
import matplotlib
from matplotlib import pyplot as plt
import queue
import time
import pyarmnn as ann
import picamera2
print("Working with Arm NN version", ann.ARMNN_VERSION)

interpreter_id = 1 #Change this number to switch between interpreters
interpreter_type = ["TFLITE", "ARMNN"][interpreter_id]

model_id = 2 #Change this number to switch between models.
model_name = ["MobileNetV2.tflite",'Xception.tflite', 'model_nasnetlarge.tflite'][model_id]



base_dir = labels.base_dir

meta, test = labels.import_meta_test()
fine_label_names = labels.get_fine_label_names(meta)
x_test, y_test = labels.import_cifar100()


tflite_filename = base_dir + model_name

interpreter1, height, width = labels.define_tflite_interpreter(tflite_filename)

input_details = interpreter1.get_input_details()
output_details = interpreter1.get_output_details()
details = (input_details, output_details)
####

parser = ann.ITfLiteParser()
network = parser.CreateNetworkFromBinaryFile(tflite_filename)


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
    
###
def get_top_k(x,k):
    #sorted_list = sorted([(idx, x[idx]/sum(x)) for idx in range(len(x))], key=(lambda x: x[1]))[::-1]
    #return sorted_list[:k]
    #print(x.shape)
    return sorted([(idx, x[idx]/sum(x)) for idx in range(len(x))], key=(lambda x: x[1]))[::-1][:k]
def plot_image(image):
    plt.imshow(image)
    plt.axis('off')  # Remove axis ticks and labels
    plt.show(block=True)
    #plt.pause(5)
    return plt


def run_model(image, interpreter_):
    pred, lat = labels.tflite_inference(image, details, interpreter = interpreter_, with_perc=True)
    #print(pred)
    return pred



def to_name(pred):
    return str(fine_label_names[pred])[2:-1]

def test(n):
    num = 1
    while num<n:
        num+=1
        example_image = x_test[num,:,:,:]
        #plot_image(example_image)

        pred, lat = labels.tflite_inference(example_image, details, interpreter = interpreter1)
        if y_test[num][0] == pred:
            
            print(f"Image {num} name from prediction:", fine_label_names[pred])
            print(f"Image {num} prediction:", pred)
            print(f"Image {num} truth:", y_test[num][0])
            print(f"Image {num} name from truth:", fine_label_names[y_test[num][0]])
            plot_image(example_image)

current_class = "apple"
#Preview = picamera2.Preview
#from libcamera import Transform
#picam2 = picamera2.Picamera2()
import cv2
if __name__ == "__main__":
    #camera_buffer = queue.Queue(maxsize=5)
    #picam2 = picamera2.Picamera2()
    #picam2.start_preview(Preview.QTGL, x=100, y=200, width=244, height=244, transform=Transform(hflip=1))
    #picam2.start()
    #cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)

    #set dimensions
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

    
    
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
    cap.set(cv2.CAP_PROP_FPS, 90)
    k = 0
    current_class = None
    lat_list = []
    try:
        while True:
            ret, image = cap.read()
            
            
            image2 = image
            if current_class:
                text_position = (image2.shape[1] - 200, 30)  # (x, y) position
                font_scale = 0.6
                font_color = (255, 255, 255)  # White
                font_thickness = 2
                cv2.putText(image2, f"Class: {current_class}", text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
            cv2.imshow('a',image2)
            image = cv2.resize(image, (32,32))
            k = cv2.waitKey(10)
            # Press q to break
            if k == ord('q'):
                break
            image = image[:,:,[2,1,0]].reshape(32,32,3)
            x_test[1,:,:,:] = image
            if not ret: raise RuntimeError("failed to frame ya")

            start = time.perf_counter()
            if interpreter_type == "TFLITE":
                pred = run_model(image, interpreter1)[0]
                #print(pred)
            if interpreter_type == "ARMNN":
                pred = pyarmnn_inference(x_test[1,:,:,:], 1)[0]
            #print(pred)
            lat = time.perf_counter() - start
            lat_list = lat_list[:10] + [1/lat,]
            #print(pred)
            res = get_top_k(pred,3)
            print(res)
            top_k = [(to_name(res[idx][0]),res[idx][1]*100) for idx in range(len(res))]
            print(top_k)
            current_class = str(top_k[0][0])
            print("FPS Now:", sum(lat_list)/len(lat_list))
    except KeyboardInterrupt as e:
        cap.release()
        exit()
    """
    image = image[:,:, [2,1,0]]
    plot_image(image)
    time.sleep(10)
    #camera_thread()
    """
    print("hi")
