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
    # import sys
    # print("Python version: {}.{}".format(sys.version_info[0],sys.version_info[1]))
    # print("Tensor Flow Built_for_GPU: {}".format(built_with_CUDA))
    print("GPU is available: {}".format(GPU_is_available))

def colab_setup():
    import numpy as np
    # Don't display numpy in scientific notation
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    
    # Load the matplotlib import pyplot to set the figure size
    from matplotlib import pyplot as plt
    plt.rcParams["figure.figsize"] = (20,10)
    
    # Shows any files saved in Google Cloab
    import os
    print( os.getcwd()  )
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


# usage: unzip_file('images.zip', '.')
def unzip_file(path_to_zip_file, directory_to_extract_to):
  import zipfile
  with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
      zip_ref.extractall(directory_to_extract_to)

#---------------------------------------------------------    
# recording and showing video on Colab:
# https://star-ai.github.io/Rendering-OpenAi-Gym-in-Colaboratory/
#
# !pip install pyvirtualdisplay > /dev/null 2>&1
# !apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1

def start_pyvirtualdisplay(rows=1400, columns=900):
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(rows, columns))
    display.start()
    
def show_video():
  import glob
  import io
  import base64
  from IPython.display import HTML
  from IPython import display as ipythondisplay 

  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
#=========================================================    


#---------------------------------------------------------    
# showing gym video on Colab:
# https://star-ai.github.io/Rendering-OpenAi-Gym-in-Colaboratory/
#
# !pip install gym > /dev/null 2>&1
"""
Utility functions to enable video recording of gym environment
To enable video, just do
env = colab_helper.wrap_gym_env(gym.make("MsPacman-v0"))
"""
def wrap_gym_env(env):
  from gym.wrappers import Monitor
  env = Monitor(env, './video', force=True)
  return env  
#=========================================================    
  

if __name__ == '__main__':
    show_version()
