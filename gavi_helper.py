def show_version():
    import tensorflow as tf
    print("Tensor Flow version: {}".format(tf.__version__))
    built_with_CUDA = tf.test.is_built_with_cuda()
    GPU_is_available = tf.test.is_gpu_available(
        cuda_only=False,
        min_cuda_compute_capability=None
    )
    try:
      import keras as krs
    except:
      import tensorflow.keras as krs
    print("Keras version: {}".format(krs.__version__))
    import sys
    print("Python version: {}.{}".format(sys.version_info[0],sys.version_info[1]))
    # print("Tensor Flow Built_for_GPU: {}".format(built_with_CUDA))
    print("GPU is available: {}".format(GPU_is_available))

def colab_setup():
    # Don't display numpy in scientific notation
    import numpy as np
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    # Load the matplotlib import pyplot to set the figure size
    from matplotlib import pyplot
    pyplot.rcParams["figure.figsize"] = (20,10)
    # mount Google-Drive inside Colab
    from google.colab import drive
    drive.mount('/content/gdrive')
    # with open('/content/gdrive/My Drive/file.txt', 'w') as f:
    #  f.write('content inside file.txt')
    # Shows any files saved in Google Cloab
    import os
    print( os.getcwd() )
    print( os.listdir() )
    from google.colab import files 
    # files.download( "test.png" ) 


if __name__ == '__main__':
    show_version()
