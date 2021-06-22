import tensorflow as tf
from tensorflow.contrib import slim
from scipy.misc import imread, imsave, imresize
import os, random
import numpy as np

class ImageData:

    def __init__(self, img_size, channels, augment_flag=False):
        self.img_size = img_size
        self.channels = channels
        self.augment_flag = augment_flag

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.img_size, self.img_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag:
            if self.img_size < 256:
                augment_size = 256
            else :
                augment_size = self.img_size + 30
            p = random.random()
            if p > 0.5:
                img = augmentation(img, augment_size)

        return img


def load_test_data(image_path, size=256):
    img = imread(image_path, mode='L')
    # img = imresize(img, [size, size])
    Shape = img.shape
    h = Shape[0]
    w = Shape[1]
    # h_num = int(np.floor(h / size))
    # w_num = int(np.floor(w / size))
    #
    # #
    # # img = img[0:size, 0:size,:]
    #
    # num = 0
    # for i in range(h_num):
    #     for j in range(w_num):
    #         if num == 0:
    #             sub_imgs = np.expand_dims(img[size * i: size * (i + 1), size * j:size * (j + 1), :], axis=0)
    #         else:
    #             sub_imgs = np.concatenate((sub_imgs, np.expand_dims(img[size * i: size * (i + 1), size * j:size * (j + 1), :], axis=0)), axis=0)
    #         num += 1
    # # img = np.expand_dims(img, axis=0)
    # sub_imgs = preprocessing(sub_imgs)
    # print(sub_imgs.shape)
    img = preprocessing(img)
    return img, h, w


def preprocessing(x):
    x = x / 127.5 - 1  # -1 ~ 1
    return x

def augmentation(image, aug_img_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    # image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [aug_img_size, aug_img_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image

def save_images(images, size, image_path):
    return mimsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def mimsave(images, size, path):
    return imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def rgb2ycbcr(img):
    img_255 = img * 255
    y = 0.257 * img_255[:, :, :, 0] + 0.564 * img_255[:, :, :, 1] + 0.098 * img_255[:, :, :, 2] + 16
    cb = -0.148 * img_255[:, :, :, 0] - 0.291 * img_255[:, :, :, 1] + 0.439 * img_255[:, :, :, 2] + 128
    cr = 0.439 * img_255[:, :, :, 0] - 0.368 * img_255[:, :, :, 1] - 0.071 * img_255[:, :, :, 2] + 128
    y = tf.expand_dims(y, -1)
    cb = tf.expand_dims(cb, -1)
    cr = tf.expand_dims(cr, -1)
    return y/255, cb/128, cr/128


def Per_LOSS(batchimg):
    _, h, w, c = batchimg.get_shape().as_list()
    fro_2_norm = tf.reduce_sum(tf.square(batchimg), axis=[1, 2, 3])
    loss = fro_2_norm / (h * w * c)
    return loss
