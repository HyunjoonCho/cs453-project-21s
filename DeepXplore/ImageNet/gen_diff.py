'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input
from keras_preprocessing.image import list_pictures

from imageio import imwrite
from json import dump
import _pickle as pickle

import os
import time
import sys

from configs import bcolors
from utils import *

# read the parameter
# argument parsing
tf.compat.v1.disable_eager_execution() # L166 - K.gradients cannot be called in eager exec mode

parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in ImageNet dataset')
parser.add_argument('cov_type', type=str, help="type of neuron coverage", choices=["basic", "multi", "strong", "boundary"])
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-k', '--k_section', help="number of sections in multisection coverage", type=int)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(50, 50), type=tuple)

args = parser.parse_args()
if args.cov_type == 'multi' and not args.k_section:
    parser.error('k-multisection neruon coverage requires -k/--k_section')

# input image dimensions
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = VGG16(input_tensor=input_tensor)
model2 = VGG19(input_tensor=input_tensor)
model3 = ResNet50(input_tensor=input_tensor)

img_paths = sorted(list_pictures('./seeds/', ext='jpeg'))
minmax_seeds = 1000
gen_set = img_paths[minmax_seeds:]

files = os.listdir('./minmaxdict/')
if len(files) != 3:
    minmax_set = img_paths[:minmax_seeds]
    minmax_dict1, minmax_dict2, minmax_dict3 = init_minmax_tables(model1, model2, model3)
    print('Start min-max iteration')
    for i in range(len(minmax_set)):
        x = preprocess_image(minmax_set[i])
        store_minmax(x, model1, minmax_dict1)
        store_minmax(x, model2, minmax_dict2)
        store_minmax(x, model3, minmax_dict3)
        if (i + 1) % 10 == 0:
            print("{}-th iteration ended".format(i + 1))
    with open('./minmaxdict/dict1.txt', 'wb') as file:
        pickle.dump(minmax_dict1, file)
    with open('./minmaxdict/dict2.txt', 'wb') as file:
        pickle.dump(minmax_dict2, file)
    with open('./minmaxdict/dict3.txt', 'wb') as file:
        pickle.dump(minmax_dict3, file)
    print('Saved initial min-max dictionary')

else:
    with open('./minmaxdict/dict1.txt', 'rb') as file:
        minmax_dict1 = pickle.load(file)
    with open('./minmaxdict/dict2.txt', 'rb') as file:
        minmax_dict2 = pickle.load(file)
    with open('./minmaxdict/dict3.txt', 'rb') as file:
        minmax_dict3 = pickle.load(file)
    print('Loaded initial min-max dictionary')

if args.cov_type == 'basic':
    from coverages import NeuronCoverage as Coverage
elif args.cov_type == 'multi':
    from coverages import MutlisectionNeuronCoverage as Coverage
elif args.cov_type == 'strong':
    from coverages import StrongNeuronBoundaryCoverage as Coverage
elif args.cov_type == 'boundary':
    from coverages import NeuronBoundaryCoverage as Coverage

cov = Coverage([model1, model2, model3], [minmax_dict1, minmax_dict2, minmax_dict3], args.threshold, args.k_section)

# ==============================================================================================
# start gen inputs
result_list = []
gen_input_path, result_path = retrieve_path(args.cov_type)
start_time = time.perf_counter()
adv_count = 0

for n in range(args.seeds):
    gen_img = preprocess_image(random.choice(gen_set))
    orig_img = gen_img.copy()
    # first check if input already induces differences
    pred1, pred2, pred3 = model1.predict(gen_img), model2.predict(gen_img), model3.predict(gen_img)
    label1, label2, label3 = np.argmax(pred1[0]), np.argmax(pred2[0]), np.argmax(pred3[0])
    if not label1 == label2 == label3:
        print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(decode_label(pred1),
                                                                                            decode_label(pred2),
                                                                                            decode_label(
                                                                                                pred3)) + bcolors.ENDC)

        cov.update_coverage(gen_img, model1, 0)
        cov.update_coverage(gen_img, model2, 1)
        cov.update_coverage(gen_img, model3, 2)

        cover1 = cov.neuron_covered(0)
        cover2 = cov.neuron_covered(1)
        cover3 = cov.neuron_covered(2)

        print('covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (cover1[0], cover1[2], cover2[0], cover2[2], cover3[0], cover3[2]))
        averaged_nc = (cover1[0] + cover2[0] + cover3[0]) / float(cover1[1] + cover2[1] +cover3[1])
        print('averaged covered neurons %.3f' % averaged_nc)

        # gen_img_deprocessed = deprocess_image(gen_img)

        # # save the result to disk
        # imwrite(os.path.join(gen_input_path, 'already_differ_' + decode_label(pred1) + '_' + decode_label(
        #     pred2) + '_' + decode_label(pred3) + '.png'), gen_img_deprocessed)
        continue

    print('seed-{} start gradient ascent'.format(n + 1))
    # if all label agrees
    orig_pred = pred1
    orig_label = label1
    layer_name1, index1 = cov.neuron_to_cover(0)
    layer_name2, index2 = cov.neuron_to_cover(1)
    layer_name3, index3 = cov.neuron_to_cover(2)
    # TODO: modify coverages to return loss right away / give pred as param

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(model1.get_layer('predictions').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('predictions').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('predictions').output[..., orig_label])
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer('predictions').output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('predictions').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('predictions').output[..., orig_label])
    elif args.target_model == 2:
        loss1 = K.mean(model1.get_layer('predictions').output[..., label1])
        loss2 = K.mean(model2.get_layer('predictions').output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('predictions').output[..., orig_label])
    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
    loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])
    layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron + loss2_neuron + loss3_neuron)
    # coverage tends to be low, thus use bigger args.weight_diff

    # for adversarial image generation
    final_loss = K.mean(layer_output)

    # we compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, grads])

    # we run gradient ascent for 20 steps
    for iters in range(args.grad_iterations):
        loss_value1, loss_value2, loss_value3, loss_neuron1, loss_neuron2, loss_neuron3, grads_value = iterate(
            [gen_img])
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point, args.occlusion_size)
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)

        gen_img += grads_value * args.step
        predict1 = model1.predict(gen_img)
        predict2 = model2.predict(gen_img)
        predict3 = model3.predict(gen_img)
        predictions1 = np.argmax(predict1[0])
        predictions2 = np.argmax(predict2[0])
        predictions3 = np.argmax(predict3[0])

        if not predictions1 == predictions2 == predictions3:
            adv_count += 1
            cov.update_coverage(gen_img, model1, 0)
            cov.update_coverage(gen_img, model2, 1)
            cov.update_coverage(gen_img, model3, 2)

            cover1 = cov.neuron_covered(0)
            cover2 = cov.neuron_covered(1)
            cover3 = cov.neuron_covered(2)
            print('covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (cover1[0], cover1[2], cover2[0], cover2[2], cover3[0], cover3[2]))
            averaged_nc = (cover1[0] + cover2[0] + cover3[0]) / float(cover1[1] + cover2[1] +cover3[1])
            print('averaged covered neurons %.3f' % averaged_nc)

            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)

            l2_distance = np.linalg.norm(orig_img_deprocessed - gen_img_deprocessed)
            result_list.append((decode_label(orig_pred), decode_label(predict1), decode_label(predict2), decode_label(predict3), 
                                iters, averaged_nc, l2_distance))
            print('L2 distance to original image %d' % l2_distance)
            # save the result to disk
            imwrite(os.path.join(gen_input_path, args.transformation + '_' + decode_label(predict1) + '_' + decode_label(
                predict2) + '_' + decode_label(predict3) + '.png'),
                   gen_img_deprocessed)
            imwrite(os.path.join(gen_input_path, args.transformation + '_' + decode_label(predict1) + '_' + decode_label(
                predict2) + '_' + decode_label(predict3) + '_orig.png'),
                   orig_img_deprocessed)
            break
        if iters == args.grad_iterations - 1:
            result_list.append((decode_label(orig_pred), decode_label(predict1), decode_label(predict2), decode_label(predict3), 
                                iters, 0, 0))
            print('Failed generating image with different output')

elapsed_time = time.perf_counter() - start_time

hash = hex(abs(hash(frozenset(vars(args).items()))))[2:10]

with open(os.path.join(result_path, 'summary_' + args.cov_type + "_" + args.transformation + "_" + hash + ".csv"), 'w') as summary_file:
    summary_file.write("Original Prediction,Prediction 1,Prediction 2,Prediction 3,Iter Num,Averaged NC,L2 Distance\n")
    for item in result_list:
        summary_file.write(f"{item[0]},{item[1]},{item[2]},{item[3]},{item[4]},{item[5]},{item[6]}\n")

with open(os.path.join(result_path, "config_" + args.cov_type + "_" + args.transformation + "_" + hash + ".json"), 'w') as config_file:
    result = vars(args)
    result['average time'] = elapsed_time/adv_count
    dump(result, config_file)