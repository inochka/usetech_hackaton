import io
import numpy as np    
from PIL import Image
import pandas as pd 



image_string = open('res/binary_plann.png', 'rb').read()
img = Image.open(io.BytesIO(image_string))
arr = np.asarray(img)


df = pd.DataFrame(1 - arr//255)
df.to_csv("res/binary_plan.csv")