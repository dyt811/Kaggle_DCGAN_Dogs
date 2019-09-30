import keras
import numpy as np
from imageio import imsave
from PythonUtils.file import unique_name
from tqdm import tqdm

a = keras.models.load_model(r"C:\Dogs\Models\2019-08-29T11_39_51.849180_generat.h5")


for index in tqdm(range(50)):
    b = np.random.rand(1, 100)
    c = a.predict(b)
    d = c[0, :, :, :]
    images_generated = 0.5 * d + 0.5
    imsave(f"C:\Dogs\Hotdog_{unique_name()}.png", images_generated)
