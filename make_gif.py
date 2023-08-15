import imageio
import re
import os
import tqdm
images = []



filter = [i for i in os.listdir("./") if i.split(".")[0].isnumeric()]

for img in tqdm.tqdm(sorted(filter, key = lambda x: int(x.split(".")[0]))):
    images.append(imageio.imread(img))

f = 'mnist_video.mp4'
imageio.mimwrite(f, images[::2], fps=70)