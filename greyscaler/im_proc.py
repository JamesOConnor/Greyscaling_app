from PIL import Image, ImageCms, ExifTags
import numpy as np
import sklearn.decomposition as dc
import cv2

def open_and_orient(fp_im):
    try:
        im = Image.open(fp_im)
        if hasattr(im, '_getexif'):  # only present in JPEGs
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            e = im._getexif()  # returns None if no EXIF data
            if e is not None:
                exif = dict(e.items())
                orientation = exif[orientation]

                if orientation == 3:
                    im = im.transpose(Image.ROTATE_180)
                elif orientation == 6:
                    im = im.transpose(Image.ROTATE_270)
                elif orientation == 8:
                    im = im.transpose(Image.ROTATE_90)
    except:
        im = Image.open(fp_im)
    return im


def avgLabPCAPCAwLabPCA_custom_coefs(x):
    out = np.zeros_like(x).astype(float)
    IMG_Lab = cv2.cvtColor(x, cv2.COLOR_RGB2Lab)
    for n in range(1, 4):
        f = 0.25 + (n - 1) * 0.2;
        coefs = [1 - f, f / 2, f / 2]
        Lab_Columns = np.reshape(IMG_Lab * coefs, (1 - IMG_Lab.shape[0] * IMG_Lab.shape[1], 3))
        pc = dc.pca.PCA(n_components=1)
        pc.fit(Lab_Columns)
        transformed_Data = pc.transform(Lab_Columns)
        transformed_Data = np.reshape(transformed_Data, (IMG_Lab.shape[0], IMG_Lab.shape[1], 1))
        b = (transformed_Data - np.min(transformed_Data)) / (np.max(transformed_Data) - np.min(transformed_Data))
        out[:, :, n - 1] = b[:, :, 0]
    IMG_GreyALL_Columns = np.reshape(out, (out.shape[0] * out.shape[1], 3))
    pc.fit(IMG_GreyALL_Columns)
    transformed_Data2 = pc.transform(IMG_GreyALL_Columns)
    transformed_Data2 = np.reshape(transformed_Data2, (IMG_Lab.shape[0], IMG_Lab.shape[1], 1))
    IMG_Grey1 = (transformed_Data2 - np.min(transformed_Data2)) / (
        np.max(transformed_Data2) - np.min(transformed_Data2))
    IMG_Grey1 = IMG_Grey1[:, :, 0]
    Lab_Columns = np.reshape(IMG_Lab, (IMG_Lab.shape[0] * IMG_Lab.shape[1], 3))
    pc.fit(Lab_Columns)
    transformed_Data3 = pc.transform(Lab_Columns)
    transformed_Data3 = np.reshape(transformed_Data2, (IMG_Lab.shape[0], IMG_Lab.shape[1], 1))
    IMG_Grey2 = (transformed_Data3 - np.min(transformed_Data3)) / (
        np.max(transformed_Data3) - np.min(transformed_Data3))
    IMG_Grey2 = IMG_Grey2[:, :, 0]
    IMG_Grey = (IMG_Grey1 + IMG_Grey2) / 2;
    return IMG_Grey


def avgYCrCbPCAPCAwLabPCA(x):
    out = np.zeros_like(x).astype(float)
    IMG_Lab = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)
    for n in range(1, 4):
        f = 0.25 + (n - 1) * 0.2;
        coefs = [1 - f, f / 2, f / 2]
        Lab_Columns = np.reshape(IMG_Lab * coefs, (1 - IMG_Lab.shape[0] * IMG_Lab.shape[1], 3))
        pc = dc.pca.PCA(n_components=1)
        pc.fit(Lab_Columns)
        transformed_Data = pc.transform(Lab_Columns)
        transformed_Data = np.reshape(transformed_Data, (IMG_Lab.shape[0], IMG_Lab.shape[1], 1))
        b = (transformed_Data - np.min(transformed_Data)) / (np.max(transformed_Data) - np.min(transformed_Data))
        out[:, :, n - 1] = b[:, :, 0]
    IMG_GreyALL_Columns = np.reshape(out, (out.shape[0] * out.shape[1], 3))
    pc.fit(IMG_GreyALL_Columns)
    transformed_Data2 = pc.transform(IMG_GreyALL_Columns)
    transformed_Data2 = np.reshape(transformed_Data2, (IMG_Lab.shape[0], IMG_Lab.shape[1], 1))
    IMG_Grey1 = (transformed_Data2 - np.min(transformed_Data2)) / (
        np.max(transformed_Data2) - np.min(transformed_Data2))
    IMG_Grey1 = IMG_Grey1[:, :, 0]
    Lab_Columns = np.reshape(IMG_Lab, (IMG_Lab.shape[0] * IMG_Lab.shape[1], 3))
    pc.fit(Lab_Columns)
    transformed_Data3 = pc.transform(Lab_Columns)
    transformed_Data3 = np.reshape(transformed_Data2, (IMG_Lab.shape[0], IMG_Lab.shape[1], 1))
    IMG_Grey2 = (transformed_Data3 - np.min(transformed_Data3)) / (
        np.max(transformed_Data3) - np.min(transformed_Data3))
    IMG_Grey2 = IMG_Grey2[:, :, 0]
    IMG_Grey = (IMG_Grey1 + IMG_Grey2) / 2;
    return IMG_Grey


def pca(x, component=0):
    out = np.zeros_like(x).astype(float)
    x = np.array(x)
    IMG_Colour_Columns = np.reshape(x, (int(x.shape[0]) * int(x.shape[1]), 3));
    pc = dc.pca.PCA(n_components=3)
    pc.fit(IMG_Colour_Columns)
    transformed_Data = pc.transform(IMG_Colour_Columns)
    transformed_Data = np.reshape(transformed_Data, x.shape);
    transformed_Data = transformed_Data[:, :, component];
    IMG_Grey = (transformed_Data - np.min(transformed_Data)) / (np.max(transformed_Data) - np.min(transformed_Data));
    return IMG_Grey
