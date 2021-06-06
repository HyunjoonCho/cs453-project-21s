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

from configs import bcolors
from utils import *

# read the parameter
# argument parsing
tf.compat.v1.disable_eager_execution() # L166 - K.gradients cannot be called in eager exec mode

parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in ImageNet dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('k', help="number of sections in multisection coverage", type=int)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(50, 50), type=tuple)
parser.add_argument('--param', type=str, default=None, choices=["multi", "strong", "boundary"], help="type of neuron coverage, classical one if none")

args = parser.parse_args()

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

# init coverage table
img_paths = sorted(list_pictures('./seeds/', ext='jpeg'))
minmax_seeds = 1000
minmax_set = img_paths[:minmax_seeds]
gen_set = img_paths[minmax_seeds:]

files = os.listdir('./minmaxdict/')
if len(files) != 3:
    model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3, args.k)
    print('Start min-max iteration')
    for i in range(len(minmax_set)):
        x = preprocess_image(minmax_set[i])
        store_minmax(x, model1, model_layer_dict1)
        store_minmax(x, model2, model_layer_dict2)
        store_minmax(x, model3, model_layer_dict3)
        if (i + 1) % 10 == 0:
            print("{}-th iteration ended".format(i + 1))
    with open('./minmaxdict/dict1.txt', 'wb') as file:
        pickle.dump(model_layer_dict1, file)
    with open('./minmaxdict/dict2.txt', 'wb') as file:
        pickle.dump(model_layer_dict2, file)
    with open('./minmaxdict/dict3.txt', 'wb') as file:
        pickle.dump(model_layer_dict3, file)
    print('Saved initial min-max dictionary')

else:
    with open('./minmaxdict/dict1.txt', 'rb') as file:
        model_layer_dict1 = pickle.load(file)
    with open('./minmaxdict/dict2.txt', 'rb') as file:
        model_layer_dict2 = pickle.load(file)
    with open('./minmaxdict/dict3.txt', 'rb') as file:
        model_layer_dict3 = pickle.load(file)
    print('Loaded initial min-max dictionary')

# ==============================================================================================
# start gen inputs
result_list = []
for _ in range(args.seeds):
    gen_img = preprocess_image(random.choice(gen_set))
    orig_img = gen_img.copy()
    # first check if input already induces differences
    pred1, pred2, pred3 = model1.predict(gen_img), model2.predict(gen_img), model3.predict(gen_img)
    label1, label2, label3 = np.argmax(pred1[0]), np.argmax(pred2[0]), np.argmax(pred3[0])
    sure1, sure2, sure3 = np.max(pred1[0]), np.max(pred2[0]), np.max(pred3[0])
    if not label1 == label2 == label3:
        print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(decode_label(pred1),
                                                                                            decode_label(pred2),
                                                                                            decode_label(
                                                                                                pred3)) + bcolors.ENDC)

        print ('with sureness', sure1, sure2, sure3)
        update_coverage(gen_img, model1, model_layer_dict1, args.threshold, args.k)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold, args.k)
        update_coverage(gen_img, model3, model_layer_dict3, args.threshold, args.k)

        cover1 = neuron_covered(model_layer_dict1, args.param)
        cover2 = neuron_covered(model_layer_dict2, args.param)
        cover3 = neuron_covered(model_layer_dict3, args.param)

        print('covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (cover1[0], cover1[2], cover2[0], cover2[2], cover3[0], cover3[2]))
        averaged_nc = (cover1[0] + cover2[0] + cover3[0]) / float(cover1[1] + cover2[1] +cover3[1])
        print('averaged covered neurons %.3f' % averaged_nc)

        gen_img_deprocessed = deprocess_image(gen_img)

        # save the result to disk
        imwrite('./generated_inputs/' + 'already_differ_' + decode_label(pred1) + '_' + decode_label(
            pred2) + '_' + decode_label(pred3) + '.png', gen_img_deprocessed)
        continue

    # if all label agrees
    orig_label = label1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1, args.param)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2, args.param)
    layer_name3, index3 = neuron_to_cover(model_layer_dict3, args.param)

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
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value

        gen_img += grads_value * args.step
        predict1 = model1.predict(gen_img)[0]
        predict2 = model2.predict(gen_img)[0]
        predict3 = model3.predict(gen_img)[0]
        predictions1 = np.argmax(predict1)
        predictions2 = np.argmax(predict2)
        predictions3 = np.argmax(predict3)
        sureness1 = np.max(predict1)
        sureness2 = np.max(predict2)
        sureness3 = np.max(predict3)

        if not predictions1 == predictions2 == predictions3:
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold, args.k)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold, args.k)
            update_coverage(gen_img, model3, model_layer_dict3, args.threshold, args.k)

            cover1 = neuron_covered(model_layer_dict1, args.param)
            cover2 = neuron_covered(model_layer_dict2, args.param)
            cover3 = neuron_covered(model_layer_dict3, args.param)
            print('covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (cover1[0], cover1[2], cover2[0], cover2[2], cover3[0], cover3[2]))
            averaged_nc = (cover1[0] + cover2[0] + cover3[0]) / float(cover1[1] + cover2[1] +cover3[1])
            print('averaged covered neurons %.3f' % averaged_nc)

            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)

            l2_distance = np.linalg.norm(orig_img_deprocessed - gen_img_deprocessed)
            result_list.append((decode_label(predictions1), decode_label(predictions2), decode_label(predictions3), 
                                sureness1, sureness2, sureness3, iters, l2_distance))
            print('L2 distance to original image %d' % l2_distance)
            # save the result to disk
            imwrite('./generated_inputs/' + args.transformation + '_' + decode_label(predictions1) + '_' + decode_label(
                predictions2) + '_' + decode_label(predictions3) + '.png',
                   gen_img_deprocessed)
            imwrite('./generated_inputs/' + args.transformation + '_' + decode_label(predictions1) + '_' + decode_label(
                predictions2) + '_' + decode_label(predictions3) + '_orig.png',
                   orig_img_deprocessed)
            break
        if iters == args.grad_iterations:
            print('Failed generation in {} steps'.format(args.grad_iterations))

hash = hex(abs(hash(frozenset(vars(args).items()))))[2:10]

with open("./results/summary_" + args.param + "_" + args.transformation + "_" + hash + ".csv", 'w') as summary_file:
    summary_file.write("Prediction 1,Prediction 2,Prediction 3,Sureness 1,Sureness 2,Sureness 3,Iter Num,L2 Distance\n")
    for item in result_list:
        summary_file.write(f"{item[0]},{item[1]},{item[2]},{item[3]},{item[4]},{item[5]},{item[6]},{item[7]}\n")

with open("./results/config_" + args.param + "_" + args.transformation + "_" + hash + ".json", 'w') as config_file:
    dump(vars(args), config_file)