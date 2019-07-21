from __future__ import absolute_import

def show_version():
    import numpy as np
    #Anything from keras
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
    # before importing keras
    import numpy as np
    # Don't display numpy in scientific notation
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    
    # Load the matplotlib import pyplot to set the figure size
    from matplotlib import pyplot
    pyplot.rcParams["figure.figsize"] = (20,10)
    
    # Shows any files saved in Google Cloab
    import os
    print( os.getcwd() )
    print( os.listdir() )
    
    from google.colab import files 
    # files.download( "test.png" ) 
    
    # this prevents an error I was facing when downloading images from the internet
    import logging
    logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

def colab_mount_gdrive():
    # mount Google-Drive inside Colab
    from google.colab import drive
    drive.mount('/content/gdrive')
    # with open('/content/gdrive/My Drive/file.txt', 'w') as f:
    #  f.write('content inside file.txt')
    
    
# number of seconds nicely formatted as a time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"
    
    


def grid_display_images(list_of_images, list_of_titles=[], no_of_columns=5, figsize=(20,20)):
  import matplotlib.pyplot as plt  
  plt.rcParams["figure.figsize"] = (20,10)
  # tested with   
  # from PIL import Image
  fig = plt.figure(figsize = figsize)
  column = 0
  for i in range(len(list_of_images)):
      column += 1
      #  check for end of column and create a new figure
      if column == no_of_columns+1:
          fig = plt.figure(figsize = figsize)
          column = 1
      fig.add_subplot(1, no_of_columns, column)
      plt.imshow(list_of_images[i])
      plt.axis('off')
      if len(list_of_titles) >= len(list_of_images):
          plt.title(list_of_titles[i])
 
# usage:
# history = model.fit(......
# plot_keras_fit_history('loss', 'val_loss', history)
# plot_keras_fit_history('accuracy', 'val_accuracy', history)

def plot_keras_fit_history(train_label, validation_label, history):
  #  %matplotlib inline
  import matplotlib.pyplot as plt
  history_dict = history.history
  loss = history_dict[train_label]
  val_loss = history_dict[validation_label]  
  epochs = range(1, len(loss) + 1)
  # "bo" is for "blue dot"
  plt.plot(epochs, loss, 'bo', label=train_label)
  # b is for "solid blue line"
  plt.plot(epochs, val_loss, 'b', label=validation_label)
  plt.title('Keras fit history')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()




if __name__ == '__main__':
    show_version()
