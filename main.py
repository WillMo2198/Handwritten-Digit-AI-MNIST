from random import randint
from tqdm import tqdm
from PIL import Image
import numpy as np
import os


class ANN:
    def __init__(self):
        self.y = np.empty((10,))
        self.weights1 = np.random.uniform(low=-1., high=1., size=(784, 64))
        self.weights2 = np.random.uniform(low=-1., high=1., size=(64, 10))
        self.list = []

    @staticmethod
    def s(x, deriv=False):
        if deriv:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))

    def train(self, var):
        if var == 0:
            self.y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif var == 1:
            self.y = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif var == 2:
            self.y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif var == 3:
            self.y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif var == 4:
            self.y = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif var == 5:
            self.y = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif var == 6:
            self.y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif var == 7:
            self.y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif var == 8:
            self.y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif var == 9:
            self.y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        for x in range(600):
            inputs = np.asarray(self.list)
            hidden = self.s(np.dot(inputs, self.weights1))
            output = self.s(np.dot(hidden, self.weights2))
            output.flatten()
            output_error = self.y - output
            output_delta = output_error * self.s(output, deriv=True)
            print(inputs)
            hidden_error = output_delta.dot(self.weights2.T)
            hidden_delta = hidden_error * self.s(hidden, deriv=True)
            self.weights2 += hidden.T.dot(output_delta)
            self.weights1 += inputs.T.dot(hidden_delta)

    def test(self, image):
        inputs = image
        hidden = self.s(np.dot(inputs, np.load('weights1.npy')))
        outputs = self.s(np.dot(hidden, np.load('weights2.npy')))
        print(np.argmax(outputs))


ann = ANN()
dir0 = 'mnist_png/training/0/'
dir1 = 'mnist_png/training/1/'
dir2 = 'mnist_png/training/2/'
dir3 = 'mnist_png/training/3/'
dir4 = 'mnist_png/training/4/'
dir5 = 'mnist_png/training/5/'
dir6 = 'mnist_png/training/6/'
dir7 = 'mnist_png/training/7/'
dir8 = 'mnist_png/training/8/'
dir9 = 'mnist_png/training/9/'
img0 = os.listdir(dir0)
img1 = os.listdir(dir1)
img2 = os.listdir(dir2)
img3 = os.listdir(dir3)
img4 = os.listdir(dir4)
img5 = os.listdir(dir5)
img6 = os.listdir(dir6)
img7 = os.listdir(dir7)
img8 = os.listdir(dir8)
img9 = os.listdir(dir9)
max0 = len(img0)
max1 = len(img1)
max2 = len(img2)
max3 = len(img3)
max4 = len(img4)
max5 = len(img5)
max6 = len(img6)
max7 = len(img7)
max8 = len(img8)
max9 = len(img9)
f0 = 0
f1 = 0
f2 = 0
f3 = 0
f4 = 0
f5 = 0
f6 = 0
f7 = 0
f8 = 0
f9 = 0
count = 0

train = input('Would you like to train? ')

if train in ['y', 'Y', 'yes', 'Yes', 'true', 'True']:
    for i in tqdm(range(max0+max1+max2+max3+max4+max5+max6+max7+max8+max9)):
        rand1 = randint(0, 9)
        if count > 10:
            count = 0
        if rand1 == 0 and f0 == max0:
            rand1 = randint(0, 9)
            count += 1
            continue
        elif rand1 == 1 and f1 == max1:
            rand1 = randint(0, 9)
            count += 1
            continue
        elif rand1 == 2 and f2 == max2:
            rand1 = randint(0, 9)
            count += 1
            continue
        elif rand1 == 3 and f3 == max3:
            rand1 = randint(0, 9)
            count += 1
            continue
        elif rand1 == 4 and f4 == max4:
            rand1 = randint(0, 9)
            count += 1
            continue
        elif rand1 == 5 and f5 == max5:
            rand1 = randint(0, 9)
            count += 1
            continue
        elif rand1 == 6 and f6 == max6:
            rand1 = randint(0, 9)
            count += 1
            continue
        elif rand1 == 7 and f7 == max7:
            rand1 = randint(0, 9)
            count += 1
            continue
        elif rand1 == 8 and f8 == max8:
            rand1 = randint(0, 9)
            count += 1
            continue
        elif rand1 == 9 and f9 == max9:
            rand1 = randint(0, 9)
            count += 1
            continue
        img = str
        if rand1 == 0:
            img = dir0 + img0[f0]
            f0 += 1
        elif rand1 == 1:
            img = dir1 + img1[f1]
            f1 += 1
        elif rand1 == 2:
            img = dir2 + img2[f2]
            f2 += 1
        elif rand1 == 3:
            img = dir3 + img3[f3]
            f3 += 1
        elif rand1 == 4:
            img = dir4 + img4[f4]
            f4 += 1
        elif rand1 == 5:
            img = dir5 + img5[f5]
            f5 += 1
        elif rand1 == 6:
            img = dir6 + img6[f6]
            f6 += 1
        elif rand1 == 7:
            img = dir7 + img7[f7]
            f7 += 1
        elif rand1 == 8:
            img = dir8 + img8[f8]
            f8 += 1
        elif rand1 == 9:
            img = dir9 + img9[f9]
            f9 += 1
        img = Image.open(img).convert('L')
        width, height = img.size
        data = np.asarray(list(img.getdata()))
        if np.unique(data).shape[0] == 1:
            pass
        else:
            data = (data - np.min(data)) / np.ptp(data)
        ann.list.append(data)
        ann.train(rand1)
        ann.list = []
    np.save('weights1.npy', ann.weights1)
    np.save('weights2.npy', ann.weights2)

else:
    while True:
        file = input('Enter file path: ')
        img = Image.open(file).convert('L')
        width, height = img.size
        img_data = np.asarray(list(img.getdata()))
        if np.unique(img_data).shape[0] == 1:
            pass
        else:
            img_data = (img_data - np.min(img_data)) / np.ptp(img_data)
        ann.test(img_data)
        cont = input('Would you like to contiue? ')
        if cont in ['y', 'Y', 'yes', 'Yes', 'true', 'True']:
            continue
        else:
            break
