## DeepXplore - MNIST  

Several Modifications
- Replace `from keras.utils import to_categorical` with `from keras.utils.np_utils import to_categorical`  
Not sure about the reason first approach does not work - it seems not be deprecated
- Replace scipy.misc.imsave with imageio.imwrite

Train LeNet 1, 4, 5 before image generation (takes about 20 minutes in colab runtime w/o GPU)

You can run it on [this colab notebook](https://colab.research.google.com/drive/1jQMTT7XfPNl2WkrYySG-DeWUBxDgV9nq?usp=sharing)