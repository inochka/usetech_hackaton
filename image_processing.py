import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import io
from PIL import Image
import pandas as pd





in_folder = 'in_pics/'
out_folder = 'out_pics/'


def create_path_if_absence():
    Path("/in_pics").mkdir(parents=True, exist_ok=True)
    Path("/out_pics").mkdir(parents=True, exist_ok=True)



def get_contour_of_image(pic_name):
    out_pic_name = "binary_" + pic_name

    inputImage = cv2.imread(in_folder + '/' + pic_name + '.png')
    imgFloat = inputImage.astype(np.cfloat) / 255.
    kChannel = 1 - np.max(imgFloat, axis=2)
    kChannel = (255 * kChannel).astype(np.uint8)

    binaryThresh = 190
    _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_BINARY)


    def areaFilter(minArea, inputImage):
    # Perform an area filter on the binary blobs:
        componentsNumber, labeledImage, componentStats, componentCentroids = \
            cv2.connectedComponentsWithStats(inputImage, connectivity=4)
        remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]
        filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

        return filteredImage

    minArea = 100
    binaryImage = areaFilter(minArea, binaryImage)

    kernelSize = 3
    opIterations = 2
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)
    cv2.imwrite(out_folder + '/' + out_pic_name + '.png', cv2.bitwise_not(binaryImage))


def find_coords_of_exits(pic_name):

    in_img_name = pic_name
    bin_img_name = 'binary_' + in_img_name
    tmp_img_name = 'exit'
    out_img_name = bin_img_name + '_' + tmp_img_name

    img_rgb = cv2.imread(in_folder + in_img_name + '.png')
    tmp_rgb = cv2.imread(in_folder + tmp_img_name + '.png')
    bin_rgb = cv2.imread(out_folder + bin_img_name + '.png')

    elements = list()


    for template in [tmp_rgb, np.flip(tmp_rgb, axis=1)]:
        w, h = template.shape[:-1]
        res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
        threshold = .8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):

            #input = '(' + str(pt[0]) + ',' + str(pt[1]) + ')'
            input = ( pt[0],pt[1] )
            elements.append(input)

            cv2.rectangle(bin_rgb, pt, (pt[0] + 1*w, pt[1] + 1*h), (0, 0, 255), -1)

    #with open('coord_out/exits_coords.npy', 'wb') as f:
        #np.save(f, elements)

    np.save("exits_coords", np.array(elements))

    #np.save('coord_out/exits_coords.npy', np.array(elements, dtype=object), allow_pickle=True)

    #list_to_file(elements, "exit_coords")

    #cv2.imwrite(out_folder + out_img_name + '_found' + '.png', bin_rgb)

    return np.asarray(elements)


def find_coords_of_ladders(pic_name):

    in_img_name = pic_name
    bin_img_name = 'binary_' + in_img_name
    tmp_img_name = 'ladder'
    out_img_name = bin_img_name + '_' + tmp_img_name

    img_rgb = cv2.imread(in_folder + in_img_name + '.png')
    tmp_rgb = cv2.imread(in_folder + tmp_img_name + '.png')
    bin_rgb = cv2.imread(out_folder + bin_img_name + '.png')

    elements = list()


    for template in [tmp_rgb, np.flip(tmp_rgb, axis=1)]:
        w, h = template.shape[:-1]
        res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
        threshold = .8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            input = '(' + str(pt[0]) + ',' + str(pt[1]) + ')'

            elements.append(input)

            cv2.rectangle(bin_rgb, pt, (pt[0] + 1*w, pt[1] + 1*h), (0, 0, 255), -1)


    with open('coord_out/ladders_coords.npy', 'wb') as f:
        np.save(f, elements)

    # np.save('coord_out/ladders_coords.npy', np.array(elements, dtype=object), allow_pickle=True)

    #list_to_file(elements, "ladders_coords")
    #cv2.imwrite(out_folder + out_img_name + '_found' + '.png', bin_rgb)

    return np.asarray(elements)


def list_to_file(elements, file_name):
    with open('coord_out/' + file_name + '.txt', 'w') as f:
        f.write("%s" % '[')
        if(len(elements) > 1):
            for element in elements[:-1]:
                f.write("%s" % element)
                f.write("%s" % ',')
        f.write("%s" % elements[-1])
        f.write("%s" % ']')



def shrink_binary_img(pic_name):


    bin_img_name = pic_name
    tmp_img_name = 'ladder'
    out_img_name = bin_img_name + '_' + tmp_img_name

    def thricing(iimg):
        h_x = 3
        h_y = 3


        l_y, l_x = np.shape(iimg)
        img_c = np.ones( (l_y // h_y, l_x // h_x) )
        for iy, ix in np.ndindex(img_c.shape):
            color = np.mean(iimg[iy * h_y : iy * h_y + h_y, ix * h_x : ix * h_x + h_x])
            if color < 255*0.8:
                img_c[iy, ix] = 0
            else:
                img_c[iy, ix] = 255

        return img_c


    def invert(iimg):
        for iy, ix in np.ndindex(img_c.shape):
            img_c[iy, ix] = 1 - img_c[iy, ix]


    img = cv2.imread(out_folder + bin_img_name + '.png')
    iimg = np.mean(img, axis=2)

    img_c = thricing(iimg)
    img_c = thricing(img_c)



    with open('coord_out/' + 'plan_binary.npy', 'wb') as f:
        np.save(f, 1 - img_c / 255)


def shrink_triry_img(pic_name):

    bin_img_name = pic_name
    tmp_img_name = 'ladder'
    out_img_name = bin_img_name + '_' + tmp_img_name

    sample_img = cv2.imread(out_folder + bin_img_name + '.png')


    img = sample_img.copy()
    height, width, depth = img.shape
    img[0:height, 0:width, 0:depth] = 0 # DO THIS INSTEAD
    return img


def bin_img_to_array(pic_name, folder_name):
    image_string = open(folder_name + '/' + pic_name + '.png', 'rb').read()
    img = Image.open(io.BytesIO(image_string))
    arr = np.asarray(img)

    df = pd.DataFrame(1 - arr//255)
    df.to_csv('coord_out' + pic_name + "_array" + '.csv')