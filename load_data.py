import os
import numpy as np
import nibabel as nib
from PIL import Image
from keras.utils import Sequence
#from skimage.io import imread


def load_data(nr_of_channels, batch_size=1, 
              nr_A_train_imgs=None, nr_B_train_imgs=None,
              nr_A_test_imgs=None, nr_B_test_imgs=None,
              generator=False, D_model=None, use_multiscale_discriminator=False, use_supervised_learning=False, REAL_LABEL=1.0, subfolder='MR_3D'):

    train_A_path = os.path.join('data', subfolder, 'trainA')
    train_B_path = os.path.join('data', subfolder, 'trainB')
    test_A_path = os.path.join('data', subfolder, 'testA')
    test_B_path = os.path.join('data', subfolder, 'testB')

    train_A_image_names = sorted(os.listdir(train_A_path))
    train_B_image_names = sorted(os.listdir(train_B_path))
    test_A_image_names = sorted(os.listdir(test_A_path))
    test_B_image_names = sorted(os.listdir(test_B_path))

    print(train_A_image_names)


    if generator:
        return data_sequence(train_A_path, train_B_path, train_A_image_names, train_B_image_names, nr_of_channels, batch_size=batch_size)  # D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL)
    else:
        train_A_images = create_image_array(train_A_image_names, train_A_path, nr_of_channels)
        train_B_images = create_image_array(train_B_image_names, train_B_path, nr_of_channels)
        test_A_images = create_image_array(test_A_image_names, test_A_path, nr_of_channels)
        test_B_images = create_image_array(test_B_image_names, test_B_path, nr_of_channels)
        return {"train_A_images": train_A_images,  "train_A_image_names": train_A_image_names,
                "train_B_images": train_B_images,  "train_B_image_names": train_B_image_names,
                "test_A_images": test_A_images,  "test_A_image_names": test_A_image_names,
                "test_B_images": test_B_images,  "test_B_image_names": test_B_image_names,

}


def create_image_array(image_list, image_path, nr_of_channels):
    image_array = []
    for image_name in image_list:
        print(image_name)
        #if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
        if nr_of_channels == 1:  # Gray scale image -> MR image
            #image = np.array(Image.open(os.path.join(image_path, image_name)))
            img = nib.load(os.path.join(image_path, image_name))
            image = img.get_fdata()
            image = np.pad(image,((1,0),(1,2),(1,0)), mode='constant')

            image = image[:, :, :, np.newaxis]

        else:                   # RGB image -> 3 channels
            #image = np.array(Image.open(os.path.join(image_path, image_name)))
            print("loading image_name")
            img = nib.load(os.path.join(image_path, image_name))
            image = img.get_fdata()
            image = np.pad(image,((1,0),(1,2),(1,0)), mode='constant')
        image = normalize_array(image)
        image_array.append(image)

    return np.array(image_array)
  
  
  # If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
def normalize_array(array):
    array = 2*(array - array.min())/(array.max() - array.min()) - 1

    #array = array / 127.5 - 1
    return array


class data_sequence(Sequence):

    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B, nr_of_channels, batch_size=1):  # , D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL):
        self.batch_size = batch_size
        self.train_A = []
        self.train_B = []
        for image_name in image_list_A:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_A.append(os.path.join(trainA_path, image_name))
        for image_name in image_list_B:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_B.append(os.path.join(trainB_path, image_name))

    def __len__(self):
        return int(max(len(self.train_A), len(self.train_B)) / float(self.batch_size))

    def __getitem__(self, idx):  # , use_multiscale_discriminator, use_supervised_learning):if loop_index + batch_size >= min_nr_imgs:
        if idx >= min(len(self.train_A), len(self.train_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.train_A) <= len(self.train_B):
                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.train_A[i])
                batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.train_B[i])
                batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]

        real_images_A = create_image_array(batch_A, trainA_path, nr_of_channels)
        real_images_B = create_image_array(batch_B, trainB_path, nr_of_channels)

        return real_images_A, real_images_B  # input_data, target_data


if __name__ == '__main__':
    load_data()
