# -*- coding: utf-8 -*-

import numpy as np
import os
import glob
from PIL import Image, ImageOps
from tqdm import tqdm
import argparse

desc = """
This is a programm to transform fish image to planar image.
It's only take `png` images
"""
parser = argparse.ArgumentParser(description=desc)
parser.add_argument(
    '--dirname',
    help='Image dir name',
    type=str,  # 引数は文字列として扱われるので，数字の場合は明示する
    default="../test_image/"
)
parser.add_argument(
    '--save',
    help='Save dir name',
    type=str,  # 引数は文字列として扱われるので，数字の場合は明示する
    default="./out/"
)
args = parser.parse_args()


def bilinear_interpolate(im, x, y):
    """バイナリ補間
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def exec_convert_fisheye_to_plane(ifname):
    '''
    座標は一回わかればいいはずなので、この関数では
    元画像での座標表現を返す(im_x, im_y)
    '''

    png_img = Image.open(ifname)
    png_img = np.array(png_img, dtype=np.float32)

    # 画像サイズ
    im_siz = png_img.shape
    print(im_siz)
    im_nx = im_siz[0]
    im_ny = im_siz[1]

    # 画像の中心座標
    im_cx = im_nx // 2
    im_cy = im_ny // 2

    # 展開画像の座標設定
    # 画像サイズ
    pro_nx = 1024
    pro_ny = 1024
    # 画像の中心座標
    pro_cx = pro_nx / 2.
    pro_cy = pro_ny / 2.

    # 投影計算
    # 仮定する雲の高さ(km)
    assume_z = 5.
    # 画像1枚がカバーする幅 (km)
    wide_km = 24.
    # 1pixelあたりのkm幅
    dx = wide_km / pro_nx

    # dindgenの代わりにmeshgridという関数を用いた
    tmp_y, tmp_x = np.meshgrid(np.arange(pro_nx, dtype=np.float32),
                               np.arange(pro_ny, dtype=np.float32))

    tmp_x = -(tmp_x - pro_cx) * dx
    tmp_y = -(tmp_y - pro_cy) * dx

    # 高度の配列
    tmp_z = np.zeros((pro_nx, pro_ny), dtype=np.float32)
    tmp_z[:, :] = assume_z

    # 各画素から空へ伸びていく単位ベクトルに変換
    tmp_total = np.sqrt(tmp_x**2 + tmp_y**2 + tmp_z**2)
    tmp_x /= tmp_total
    tmp_y /= tmp_total
    tmp_z /= tmp_total

    # 画像に画素値を挿入 `rad2deg`を用いた
    fisheye_lon = np.rad2deg(np.arctan2(tmp_y, tmp_x))
    fisheye_lat = np.rad2deg(np.arcsin(tmp_z))

    # 元画像でのピクセル刻み
    deg_per_pix = 1. / 8.

    # 元画像での座標表現
    im_colat = (90. - fisheye_lat) / deg_per_pix  # ここは何をしてる？
    im_x = im_colat * np.cos(np.deg2rad(fisheye_lon)) + im_cx
    im_y = im_colat * np.sin(np.deg2rad(fisheye_lon)) + im_cy
    return im_x, im_y


def main():
    # パスの取得
    ifldname = args.dirname
    ofldname = args.save
    pro_nx = 1024
    pro_ny = 1024

    # ディレクトリ内の画像リストを獲得する
    ls_im = glob.glob(os.path.join(ifldname, "*.png"))

    # もし保存するディレクトリがなかったら作る
    if not os.path.exists(ofldname):
        os.makedirs(ofldname)

    # 魚眼画像での座標表現をか獲得する
    im_x, im_y = exec_convert_fisheye_to_plane(ls_im[0])

    ls_out = []
    for path in tqdm(ls_im):
        png_img = Image.open(path)
        png_img = np.array(png_img, dtype=np.float32)

        pro_image = np.zeros((pro_nx, pro_ny, 3))
        for i in range(3):
            pro_image[:, :, i] = bilinear_interpolate(png_img[:, :, i],
                                                      im_x, im_y)

        # ひっくり返す
        tmp = Image.fromarray(np.uint8(pro_image))
        tmp = tmp.rotate(90)
        tmp = ImageOps.mirror(tmp)
        tmp = tmp.resize((144, 144))
        tmp.save(os.path.join(ofldname, os.path.basename(path)))
        pro_image = np.array(tmp)
        ls_out.append(pro_image)


if __name__ == '__main__':
    main()
