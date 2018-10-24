from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage import io
import numpy as np
def test():
    # Invert the horse image
    image = invert(data.horse())

    # perform skeletonization
    skeleton = skeletonize(image)

    # display results
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                             sharex=True, sharey=True)

    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize=20)

    ax[1].imshow(skeleton, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('skeleton', fontsize=20)

    fig.tight_layout()
    plt.show()

def ske_ext(file_dir, draw = True):
    img = io.imread(file_dir)
    img_bi = img > 200

    width = len(img_bi)
    height = len(img_bi[0])

    # Invert the horse image
    image = invert(img_bi)

    # perform skeletonization
    skeleton = skeletonize(image)

    ske =[]
    for x in range(width):
        for y in range(height):
            if skeleton[x, y] == True:
                ske.append([y, x])

    if draw:
        # display results
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4),
                                 sharex=True, sharey=True)

        ax = axes.ravel()

        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('original', fontsize=20)

        ax[1].imshow(skeleton, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('skeleton', fontsize=20)


        x,y = zip(*ske)
        ax[2].scatter(x,y)
        ax[2].axis('off')
        ax[2].set_title('skeleton', fontsize=20)

        fig.tight_layout()
        plt.show()



    return ske

if __name__ == '__main__':
    # test()
    folder = "/home/lupin/Cinnamon/Flaxscanner/Dataset/CASIA/Offline/data_png/"
    file = "b_735-f.gnt_63.png"

    ske = ske_ext(folder + file)