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


def init_coverage_tables(model1, model2, model3, k=5):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1, k)
    init_dict(model2, model_layer_dict2, k)
    init_dict(model3, model_layer_dict3, k)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_dict(model, model_layer_dict, k):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False
            model_layer_dict[(layer.name, index, 'max')] = np.NINF
            model_layer_dict[(layer.name, index, 'min')] = np.PINF
            model_layer_dict[(layer.name, index, 'boundary')] = False
            model_layer_dict[(layer.name, index, 'strong')] = False
            for i in range(k):
                model_layer_dict[(layer.name, index, i)] = False

def neuron_to_cover(model_layer_dict, param = None):
    if param is None:
        not_covered = [k for k, v in model_layer_dict.items() if len(k) == 2 and not v]
    elif param == 'multi':
        not_covered = [(k[0], k[1]) for k, v in model_layer_dict.items() if len(k) == 3 and isinstance(k[2], int) and not v]
    else:
        not_covered = [(k[0], k[1]) for k, v in model_layer_dict.items() if param in k and not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index

def neuron_covered(model_layer_dict, param=None):
    if param is None:
        covered_neurons = len([v for k, v in model_layer_dict.items() if len(k) == 2 and v])
        total_neurons = len([0 for k in model_layer_dict if len(k) == 2])
    elif param == 'multi':
        covered_neurons = len([v for k, v in model_layer_dict.items() if len(k) == 3 and isinstance(k[2], int) and v])
        total_neurons = len([0 for k in model_layer_dict if len(k) == 3 and isinstance(k[2], int)])
    else:
        covered_neurons = len([v for k, v in model_layer_dict.items() if param in k and v])
        total_neurons = len([0 for k in model_layer_dict if param in k])
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

def store_minmax(input_data, model, model_layer_dict):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]
    
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        curr = intermediate_layer_output
        for num_neuron in range(curr.shape[-1]):
            axis = tuple(range(1, len(curr.shape)-1))
            neuron_val = np.mean(curr[..., num_neuron], axis=axis)
            neuron_max = np.max(neuron_val)
            neuron_min = np.min(neuron_val)
            bef_max = model_layer_dict.get((layer_names[i], num_neuron, "max"), np.NINF)
            bef_min = model_layer_dict.get((layer_names[i], num_neuron, "min"), np.PINF)
            new_max = max(bef_max, neuron_max)
            new_min = min(bef_min, neuron_min)
            model_layer_dict[(layer_names[i], num_neuron, "max")] = new_max
            model_layer_dict[(layer_names[i], num_neuron, "min")] = new_min

def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def update_coverage(input_data, model, model_layer_dict, threshold=0, k=5):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        curr = intermediate_layer_output[0]
        scaled = scale(curr)
        for num_neuron in range(curr.shape[-1]):
            curr_neuron = np.mean(curr[..., num_neuron])
            scaled_curr = np.mean(scaled[..., num_neuron])
            max_thres = model_layer_dict[(layer_names[i], num_neuron, "max")]
            min_thres = model_layer_dict[(layer_names[i], num_neuron, "min")]
            if scaled_curr > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True
            if curr_neuron > max_thres and not model_layer_dict[(layer_names[i], num_neuron, "strong")]:
                model_layer_dict[(layer_names[i], num_neuron, "strong")] = True
            if (curr_neuron > max_thres or curr_neuron < min_thres) and not model_layer_dict[(layer_names[i], num_neuron, "boundary")]:
                model_layer_dict[(layer_names[i], num_neuron, "boundary")] = True
            section_length = (max_thres - min_thres) / k
            for i in range(k):
                if min_thres + section_length * i <= curr_neuron <= min_thres + section_length * (i + 1):
                    model_layer_dict[(layer_names[i], num_neuron, k)] = True
            break

def full_coverage(model_layer_dict, param=None):
    if param is None:
        for key in model_layer_dict:
            if len(key) == 2 and not model_layer_dict[key]:
                return False
        return True
    elif param == "multi":
        for key in model_layer_dict:
            if len(key) == 3 and isinstance(key[2], int) and not model_layer_dict[key]:
                return False
        return True
    else:
        for key in model_layer_dict:
            if param in key and not model_layer_dict[key]:
                return False
        return True


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False