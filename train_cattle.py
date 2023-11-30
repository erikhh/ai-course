from duckduckgo_search import DDGS
from fastcore.all import *
from fastai.vision.all import *
from fastdownload import download_url
import matplotlib.pyplot as plt
import random
from time import sleep

def search_images(term, max_images=50):
    print(f"Searching for '{term}'")
    return L(DDGS().images(keywords=term, max_results=max_images)).itemgot('image')

urls = search_images('cow photos', max_images=100)
dest = 'cow.jpg'
download_url(random.choice(urls), dest, show_progress=False)

searches = [
    'holstein cattle', 
    'hereford cattle', 
    'simmertal cattle', 
    'aberdeen angus cattle', 
    'belgian blue cattle', 
    'limousin cattle', 
    'hungarian grey cattle', 
    'highland cattle', 
    'heck cattle',
    'british blue cattle',
    'belted galloway cattle',
    'ayrshire cattle',
]

variations = 'photo', 'sun photo', 'shade photo', 'stable photo', 'calf photo'
path = Path('cattle_breeds')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    for v in variations:
        download_images(dest, urls=search_images(f'{o} {v}'))
        sleep(10)  
    resize_images(path/o, max_size=400, dest=path/o)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f'Failed: {len(failed)}')

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=random.randint(0, 4096)),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch()
plt.show()

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)

breed,index,probs = learn.predict(PILImage.create('cow.jpg'))
print(f"This is a: {breed}. Idx: {index}")
print(f"Probability it's a {breed}: {probs[index]:.4f}")

learn.export('cattle-4.pk1')