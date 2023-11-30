from fastai.vision.all import *
from tabulate import tabulate
import os

learn = load_learner('cattle-4.pk1')

path = Path('test_cattle')
print("Image \t\t breed \t\t \t\t probability")
result = []
images = os.listdir(path)
images.sort()
for image in images:
    src = (path/image)
    breed,index,probs = learn.predict(PILImage.create(src))
    result.append([image, breed, f'{probs[index]:.4f}'])

print(tabulate(result, headers=['Image', 'Breed', 'Probability']))