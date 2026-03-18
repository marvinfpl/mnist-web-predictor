import pandas as pd
import numpy as np

def process_input(image):
    image = np.resize(image, (28, 28))