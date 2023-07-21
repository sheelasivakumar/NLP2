import pandas as pd
import numpy as np
import os
import pickle
from tqdm.notebook import tqdm 

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequence
from tensorflow.keras.models import Model 
from tensorflow.keras.utils import to_categorrical, plot_model
from tensorflow.keras.layers import input, Dense, LSTM 