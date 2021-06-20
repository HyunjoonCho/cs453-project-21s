import random
from collections import defaultdict

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
    return input_img_data


def deprocess_image(x):
    x = x.reshape((224, 224, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def decode_label(pred):
    return decode_predictions(pred)[0][0][1]


def retrieve_path(param):
    cov_type = 0
    if param == "basic":
        cov_type = 1
    elif param == "multi":
        cov_type = 2
    elif param == "strong":
        cov_type = 3
    else:
        cov_type = 4
    gen_input_path = './generated_inputs/NC' + str(cov_type)
    result_path = './results/NC' + str(cov_type)

    return gen_input_path, result_path


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = 1e4 * np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(10, 10)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads

def init_minmax_tables(model1, model2, model3):
    model_layer_minmax_dict1 = defaultdict(bool)
    model_layer_minmax_dict2 = defaultdict(bool)
    model_layer_minmax_dict3 = defaultdict(bool)
    init_minmax_dict(model1, model_layer_minmax_dict1)
    init_minmax_dict(model2, model_layer_minmax_dict2)
    init_minmax_dict(model3, model_layer_minmax_dict3)
    return model_layer_minmax_dict1, model_layer_minmax_dict2, model_layer_minmax_dict3


def init_minmax_dict(model, model_layer_minmax_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_minmax_dict[(layer.name, index, 'max')] = np.NINF
            model_layer_minmax_dict[(layer.name, index, 'min')] = np.PINF

def store_minmax(input_data, model, minmax_dict):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]
    
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        curr = intermediate_layer_output
        for num_neuron in range(curr.shape[-1]):
            axis = tuple(range(1, len(curr.shape)-1))
            neuron_val = np.mean(curr[..., num_neuron], axis=axis)[0]
            curr_max = minmax_dict.get((layer_names[i], num_neuron, 'max'), np.NINF)
            curr_min = minmax_dict.get((layer_names[i], num_neuron, 'min'), np.PINF)
            minmax_dict[(layer_names[i], num_neuron, 'max')] = max(curr_max, neuron_val)
            minmax_dict[(layer_names[i], num_neuron, 'min')] = min(curr_min, neuron_val)