from __future__ import absolute_import
import numpy as np
import cv2

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
  

#---------------------------------------------------------    
# create mp4 video from frame file names:     
def get_file_names(folder, suffix = ""):
  import os
  file_names = list()
  for file in os.listdir(folder):
      if len(suffix) > 0 and not file.lower().endswith(suffix.lower()):
        pass
      else:
        file_name = os.path.join(folder, file)
        file_names.append(file_name)
  file_names.sort()
  return file_names


def delete_file(file_name, verbose = False):
  import os
  if os.path.exists(file_name):
    os.remove(file_name)
  else:
    if verbose:
      print(f"The file {file_name} does not exist") 
    
    
def video_start_recording(video_file_name, height, width, fps=24):
  import numpy as np
  import cv2
  
  delete_file(video_file_name)

  # Create the OpenCV VideoWriter
  # fourcc = cv2.VideoWriter_fourcc(*'avc1')  
  fourcc = cv2.VideoWriter_fourcc(*'DIVX')

  video = cv2.VideoWriter(video_file_name,
                          fourcc, # -1 denotes manual codec selection. You can make this automatic by defining the "fourcc codec" with "cv2.VideoWriter_fourcc"
                          int(fps), 
                          (width,height) # The width and height come from the stats of image1
                          )
  return video


def video_end_recording(video):
  # Release the video for it to be committed to a file
  video.release()  
  

def video_add_frame_as_numpy_array(video, numpy_array_image):
  import numpy as np
  import cv2
  # out = video.write(cv2.cvtColor(np.array(numpy_array_image), cv2.COLOR_RGB2BGR))
  out = video.write(np.array(numpy_array_image))
  # print(out)
      
    
def read_image_as_np_array(file_name, image_type=cv2.IMREAD_COLOR):
  """
  returns None in case the file is not found
  cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
  cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
  cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
  """
  import cv2
  return cv2.imread(file_name, image_type) 


def create_video_from_frame_file_names(video_file_name, file_names, fps=24):
  if len(file_names)>0:
    img = read_image_as_np_array(file_names[0])  
    height, width, _ = img.shape
    video= video_start_recording(video_file_name, height, width, fps)
    video_add_frame_as_numpy_array(video, img)
    for file_name in file_names:
      img = read_image_as_np_array(file_name) 
      video_add_frame_as_numpy_array(video, img)
    video_end_recording(video)
#=========================================================      
    
    
if __name__ == '__main__':
    show_version()
