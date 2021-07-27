# -*- coding: utf-8 -*-
"""画像を時系列で色を変えて重ね合わせる
* ２値化された時系列画像を用いて画像の経時変化を色付きで確認する

Todo:
    * 適応する画像の方向をどうやって揃えるか．移動方向をどうやって決めるか
"""
import numpy
import cv2

# read image
from PIL import Image, ImageSequence
from matplotlib import pyplot
from matplotlib.colors import Normalize  # Normalizeをimport


def plot_temporal_imgs(fig, ax, img_name, cmap_name='hsv', time_interval=5, tick_interval=5, contour=False):
    multi_img = Image.open(img_name)
    # Image -> numpy
    imgs = []
    for i, img in enumerate(ImageSequence.Iterator(multi_img)):
        im = numpy.array(img)
        if contour == True:
            im = cv2.Laplacian(im, cv2.CV_32F, ksize=3)
        imgs.append(im)
    for i, img in enumerate(imgs):
        imgs[i] = img/255
    imgs = numpy.array(imgs)
    # attach gradiation as a time elapsed
    cmap = pyplot.get_cmap(cmap_name)
    Cmap = []
    for i in range(1, len(imgs)+1):
        num = int(i*cmap.N//len(imgs))
        color = numpy.array(cmap(num))
        Cmap.append(color)
    rims = []
    gims = []
    bims = []
    for i, img in enumerate(imgs):
        rim = img*Cmap[i][0]
        gim = img*Cmap[i][1]
        bim = img*Cmap[i][2]
        rims.append(rim)
        gims.append(gim)
        bims.append(bim)
    rims = numpy.array(rims)
    gims = numpy.array(gims)
    bims = numpy.array(bims)
    max_rim = numpy.max(rims, axis=0)
    max_gim = numpy.max(gims, axis=0)
    max_bim = numpy.max(bims, axis=0)

    from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
    (h, w) = imgs[0].shape
    rgbArr = numpy.zeros((h, w, 3))
    rgbArr[..., 0] = max_rim
    rgbArr[..., 1] = max_gim
    rgbArr[..., 2] = max_bim
    ax.imshow(rgbArr)

    # for colorbar
    def add_colorbar(mapp, fig, ax, ticks, pad=0.1):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=pad)
        cbar = fig.colorbar(mapp, ax=ax, cax=cax,
                            orientation="vertical", ticks=ticks)
        cbar.outline.set_linewidth(0.05)
        cbar.ax.tick_params(length=0)
        return cax, cbar
    cb_min, cb_max = 0, (len(imgs)-1)*time_interval
    ticks = numpy.arange(cb_min, cb_max+tick_interval, tick_interval)
    cbar_im = numpy.array([[0, 0], [len(imgs), len(imgs)]])
    mapp = pyplot.pcolormesh(cbar_im, norm=Normalize(
        vmin=cb_min, vmax=cb_max), cmap=cmap)
    _, cbar = add_colorbar(mapp, fig, ax, pad=0.1, ticks=ticks)

    cbar.ax.set_yticklabels(ticks)
    cbar.ax.set_title('[sec]')
    ax.set_aspect(1/ax.get_data_ratio())


if __name__ == '__main__':
    img_name = "for_overlay.tif"
    fig, ax = pyplot.subplots()
    time_interval = 1
    plot_temporal_imgs(fig, ax, img_name,
                       time_interval=time_interval, contour=False)
    pyplot.show()
