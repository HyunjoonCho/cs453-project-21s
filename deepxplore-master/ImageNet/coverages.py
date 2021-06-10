from collections import defaultdict
import random

import numpy as np
from tensorflow.keras.models import Model

class NeuronCoverage:
    def __init__(self, models, minmax_dicts, threshold = 0.2, k = 5):
        self._threshold = threshold
        self._model_layer_dicts = []
        for model in models:
            self._init_dict(model)

    def _init_dict(self, model):
        model_layer_dict = {}
        for layer in model.layers:
            if 'flatten' in layer.name or 'input' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                model_layer_dict[(layer.name, index)] = False
        self._model_layer_dicts.append(model_layer_dict)

    def neuron_to_cover(self, model_index):
        not_covered = [k for k, v in self._model_layer_dicts[model_index].items() if len(k) == 2 and not v]
        if not_covered:
            layer_name, index = random.choice(not_covered)
        else:
            layer_name, index = random.choice(self._model_layer_dicts[model_index].keys())
        return layer_name, index

    def neuron_covered(self, model_index):
        model_layer_dict = self._model_layer_dicts[model_index]
        covered_neurons = len([v for k, v in model_layer_dict.items() if v])
        total_neurons = len(model_layer_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def update_coverage(self, input_data, model, model_index):
        layer_names = [layer.name for layer in model.layers if
                    'flatten' not in layer.name and 'input' not in layer.name]

        model_layer_dict = self._model_layer_dicts[model_index]
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
        intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            scaled = self._scale(intermediate_layer_output[0])
            for num_neuron in range(scaled.shape[-1]):
                output = np.mean(scaled[..., num_neuron])
                if output > self._threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                    model_layer_dict[(layer_names[i], num_neuron)] = True

    def _scale(self, intermediate_layer_output, rmax=1, rmin=0):
        X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
                intermediate_layer_output.max() - intermediate_layer_output.min())
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled

    def full_coverage(self, model_index):
        model_layer_dict = self._model_layer_dicts[model_index]
        for key in model_layer_dict:
            if len(key) == 2 and not model_layer_dict[key]:
                return False
        return True

class MutlisectionNeuronCoverage:
    def __init__(self, models, minmax_dicts, threshold = 0.2, k = 5):
        self._k = k
        self._minmax_dicts = minmax_dicts
        self._model_layer_dicts = []
        for model in models:
            self._init_dict(model)

    def _init_dict(self, model):
        model_layer_dict = {}
        for layer in model.layers:
            if 'flatten' in layer.name or 'input' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                for i in range(self._k):
                    model_layer_dict[(layer.name, index, i)] = False
        self._model_layer_dicts.append(model_layer_dict)

    def neuron_to_cover(self, model_index):
        not_covered = [(k[0], k[1]) for k, v in self._model_layer_dicts[model_index].items() if not v]
        if not_covered:
            layer_name, index = random.choice(not_covered)
        else:
            layer_name, index = random.choice(self._model_layer_dicts[model_index].keys())
        return layer_name, index

    def neuron_covered(self, model_index):
        model_layer_dict = self._model_layer_dicts[model_index]
        covered_neurons = len([v for k, v in model_layer_dict.items() if v])
        total_neurons = len(model_layer_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def update_coverage(self, input_data, model, model_index):
        layer_names = [layer.name for layer in model.layers if
                    'flatten' not in layer.name and 'input' not in layer.name]

        model_layer_dict = self._model_layer_dicts[model_index]
        minmax_dict = self._minmax_dicts[model_index]
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
        intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            curr = intermediate_layer_output[0]
            for num_neuron in range(curr.shape[-1]):
                curr_neuron = np.mean(curr[..., num_neuron])
                max_thres = minmax_dict[(layer_names[i], num_neuron, 'max')]
                min_thres = minmax_dict[(layer_names[i], num_neuron, 'min')]
                section_length = (max_thres - min_thres) / self._k
                for section_count in range(self._k):
                    if (min_thres + section_length * section_count <= curr_neuron <= min_thres + section_length * (section_count + 1) 
                        and not model_layer_dict[(layer_names[i], num_neuron, section_count)]):
                        model_layer_dict[(layer_names[i], num_neuron, section_count)] = True

    def full_coverage(self, model_index):
        model_layer_dict = self._model_layer_dicts[model_index]
        for key in model_layer_dict:
            if not model_layer_dict[key]:
                return False
        return True

class NeuronBoundaryCoverage:
    def __init__(self, models, minmax_dicts, threshold = 0.2, k = 5):
        self._minmax_dicts = minmax_dicts
        self._model_layer_dicts = []
        for model in models:
            self._init_dict(model)

    def _init_dict(self, model):
        model_layer_dict = {}
        for layer in model.layers:
            if 'flatten' in layer.name or 'input' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                model_layer_dict[(layer.name, index)] = 0 # 0: Not covered, 1: only min covered, 2: only max covered, 3: both covered
        self._model_layer_dicts.append(model_layer_dict)

    def neuron_to_cover(self, model_index):
        not_covered = [(k[0], k[1]) for k, v in self._model_layer_dicts[model_index].items() if v != 3]
        if not_covered:
            layer_name, index = random.choice(not_covered)
        else:
            layer_name, index = random.choice(self._model_layer_dicts[model_index].keys())
        return layer_name, index

    def neuron_covered(self, model_index):
        model_layer_dict = self._model_layer_dicts[model_index]
        covered_neurons = len([v for k, v in model_layer_dict.items() if v == 3])
        total_neurons = len(model_layer_dict.items())
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def update_coverage(self, input_data, model, model_index):
        layer_names = [layer.name for layer in model.layers if
                    'flatten' not in layer.name and 'input' not in layer.name]

        model_layer_dict = self._model_layer_dicts[model_index]
        minmax_dict = self._minmax_dicts[model_index]
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
        intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            curr = intermediate_layer_output[0]
            for num_neuron in range(curr.shape[-1]):
                output = np.mean(curr[..., num_neuron])
                max_thres = minmax_dict[(layer_names[i], num_neuron, 'max')]
                min_thres = minmax_dict[(layer_names[i], num_neuron, 'min')]
                cov_state = model_layer_dict[(layer_names[i], num_neuron)]
                if output < min_thres and (cov_state % 2) != 1:
                    model_layer_dict[(layer_names[i], num_neuron)] = cov_state + 1
                if output > max_thres and cov_state < 2:
                    model_layer_dict[(layer_names[i], num_neuron)] = cov_state + 2

    def full_coverage(self, model_index):
        model_layer_dict = self._model_layer_dicts[model_index]
        for key in model_layer_dict:
            if model_layer_dict[key] != 3:
                return False
        return True

class StrongNeuronBoundaryCoverage:
    def __init__(self, models, minmax_dicts, threshold = 0.2, k = 5):
        self._minmax_dicts = minmax_dicts
        self._model_layer_dicts = []
        for model in models:
            self._init_dict(model)

    def _init_dict(self, model):
        model_layer_dict = {}
        for layer in model.layers:
            if 'flatten' in layer.name or 'input' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                model_layer_dict[(layer.name, index)] = False
        self._model_layer_dicts.append(model_layer_dict)

    def neuron_to_cover(self, model_index):
        not_covered = [(k[0], k[1]) for k, v in self._model_layer_dicts[model_index].items() if not v]
        if not_covered:
            layer_name, index = random.choice(not_covered)
        else:
            layer_name, index = random.choice(self._model_layer_dicts[model_index].keys())
        return layer_name, index

    def neuron_covered(self, model_index):
        model_layer_dict = self._model_layer_dicts[model_index]
        covered_neurons = len([v for k, v in model_layer_dict.items() if v])
        total_neurons = len(model_layer_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def update_coverage(self, input_data, model, model_index):
        layer_names = [layer.name for layer in model.layers if
                    'flatten' not in layer.name and 'input' not in layer.name]

        model_layer_dict = self._model_layer_dicts[model_index]
        minmax_dict = self._minmax_dicts[model_index]
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
        intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            curr = intermediate_layer_output[0]
            for num_neuron in range(curr.shape[-1]):
                output = np.mean(curr[..., num_neuron])
                max_thres = minmax_dict[(layer_names[i], num_neuron, 'max')]
                if output > max_thres and not model_layer_dict[(layer_names[i], num_neuron)]:
                    model_layer_dict[(layer_names[i], num_neuron)] = True

    def full_coverage(self, model_index):
        model_layer_dict = self._model_layer_dicts[model_index]
        for key in model_layer_dict:
            if not model_layer_dict[key]:
                return False
        return True