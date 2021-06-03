'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from tensorflow import compat
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
from imageio import imwrite
from json import dump
# from scipy.misc import imsave

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import bcolors
from utils import *

compat.v1.disable_eager_execution()
# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
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
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(10, 10), type=tuple)
parser.add_argument('--param', type=str, default=None, choices=["multi", "strong", "boundary"], help="type of neuron coverage, classical one if none")

args = parser.parse_args()

print(f"""Transformation : {args.transformation}
weight_diff/nc : {args.weight_diff}/{args.weight_nc}
gradient : {args.grad_iterations} steps with {args.step} lr
seeds : {args.seeds}
param : {args.param}, {args.k} sections, threshold {args.threshold}""")
# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(x_train, _), (x_test, _) = mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_train /= 255

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)

# init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)

# start training input
# split into batch 
batch_size = 100
batch_num = (len(x_train) // batch_size) + 1
for i in range(batch_num):
    x = x_train[(i * batch_size):((i+1) * batch_size), ...]
    if len(x) != 0:
        store_minmax(x, model1, model_layer_dict1)
        store_minmax(x, model2, model_layer_dict2)
        store_minmax(x, model3, model_layer_dict3)
        if (i + 1) % 10 == 0:
            print("{}-th iteration ended".format((i + 1) * 100))

# ==============================================================================================
# start gen inputs
result_list = []
for _ in range(args.seeds):
    gen_img = np.expand_dims(random.choice(x_test), axis=0)
    orig_img = gen_img.copy()
    # first check if input already induces differences
    label1, label2, label3 = np.argmax(model1.predict(gen_img)[0]), np.argmax(model2.predict(gen_img)[0]), np.argmax(
        model3.predict(gen_img)[0])

    if not label1 == label2 == label3:
        print('input already causes different outputs: {}, {}, {}'.format(label1, label2, label3))

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
        imwrite('./generated_inputs/' + 'already_differ_' + str(label1) + '_' + str(
            label2) + '_' + str(label3) + '.png', gen_img_deprocessed)
        continue

    # if all label agrees
    orig_label = label1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1, args.param)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2, args.param)
    layer_name3, index3 = neuron_to_cover(model_layer_dict3, args.param)

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model == 2:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('before_softmax').output[..., orig_label])
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
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
            update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

            
            cover1 = neuron_covered(model_layer_dict1, args.param)
            cover2 = neuron_covered(model_layer_dict2, args.param)
            cover3 = neuron_covered(model_layer_dict3, args.param)
            print('covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (cover1[0], cover1[2], cover2[0], cover2[2], cover3[0], cover3[2]))
            averaged_nc = (cover1[0] + cover2[0] + cover3[0]) / float(cover1[1] + cover2[1] +cover3[1])
            print('averaged covered neurons %.3f' % averaged_nc)

            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)

            l1_distance = abs(orig_img_deprocessed - gen_img_deprocessed).sum()
            result_list.append((predictions1, predictions2, predictions3, sureness1, sureness2, sureness3,
                               iters, l1_distance))
            print('L1 distance to original image %d' % l1_distance)
            # save the result to disk
            imwrite('./generated_inputs/' + args.transformation + '_' + str(predictions1) + '_' + str(
                predictions2) + '_' + str(predictions3) + '.png',
                   gen_img_deprocessed)
            imwrite('./generated_inputs/' + args.transformation + '_' + str(predictions1) + '_' + str(
                predictions2) + '_' + str(predictions3) + '_orig.png',
                   orig_img_deprocessed)
            break

hash = hex(abs(hash(frozenset(vars(args).items()))))[2:10]

with open("./summary_" + args.param + "_" + args.transformation + "_" + hash + ".csv", 'w') as summary_file:
    summary_file.write("Prediction 1,Prediction 2,Prediction 3,Sureness 1,Sureness 2,Sureness 3,Iter Num,L1 Distance\n")
    for item in result_list:
        summary_file.write(f"{item[0]},{item[1]},{item[2]},{item[3]},{item[4]},{item[5]},{item[6]},{item[7]}\n")

with open("./config_" + args.param + "_" + args.transformation + "_" + hash + ".json", 'w') as config_file:
    dump(vars(args), config_file)