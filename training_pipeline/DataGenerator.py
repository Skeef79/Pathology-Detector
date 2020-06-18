import imgaug as ia
import imgaug.augmenters as iaa
import tensorflow as tf
import numpy as np
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)



def get_default_imgaug():
	
	# Возвращает набор аугментаций imgaug'а, который по нашей договоренности является дефолтным (наиболее используемым) 
	# Набор аугментаций уже непосредствено можно применять к данным
	# Вызывается конструктором DataGenerator, если в конструктор DataGenerator не передан свой набор аугментаций.
	# По большей части повторяет пример из документации: https://imgaug.readthedocs.io/en/latest/source/examples_basics.html#a-simple-and-common-augmentation-sequence
	
	sometimes = lambda aug: iaa.Sometimes(0.5, aug)
	return iaa.Sequential(
                [
                   
                    iaa.Fliplr(0.5), # horizontally flip 50% of all images
                    iaa.Flipud(0.2), # vertically flip 20% of all images

                    
                    sometimes(iaa.Crop(percent=(0, 0.1))),

                    
                    sometimes(iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-45, 45),
                        shear=(-16, 16),
                        order=[0, 1],
                        cval=(0, 255),
                        mode=ia.ALL
                    )),

                   
                    iaa.SomeOf((0, 5),
                        [
                            
                            sometimes(
                                iaa.Superpixels(
                                    p_replace=(0, 1.0),
                                    n_segments=(20, 200)
                                )
                            ),

                           
                            iaa.OneOf([
                                iaa.GaussianBlur((0, 3.0)),
                                iaa.AverageBlur(k=(2, 7)),
                                iaa.MedianBlur(k=(3, 11)),
                            ]),

                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                            # Same as sharpen, but for an embossing effect.
                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                            sometimes(iaa.OneOf([
                                iaa.EdgeDetect(alpha=(0, 0.7)),
                                iaa.DirectedEdgeDetect(
                                    alpha=(0, 0.7), direction=(0.0, 1.0)
                                ),
                            ])),

                            iaa.AdditiveGaussianNoise(
                                loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                            ),

                           
                            iaa.OneOf([
                                iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                iaa.CoarseDropout(
                                    (0.03, 0.15), size_percent=(0.02, 0.05),
                                    per_channel=0.2
                                ),
                            ]),

                            
                            iaa.Invert(0.05, per_channel=True), # invert color channels

                            # Add a value of -10 to 10 to each pixel.
                            iaa.Add((-10, 10), per_channel=0.5),

                            # Change brightness of images (50-150% of original value).
                            iaa.Multiply((0.5, 1.5), per_channel=0.5),

                            # Improve or worsen the contrast of images.
                            #iaa.LinearContrast((0.5, 2.0), per_channel=0.5), ПОЧЕМУ ТО НЕ РОБИТ

                            # Convert each image to grayscale and then overlay the
                            # result with the original with random alpha. I.e. remove
                            # colors with varying strengths.
                            iaa.Grayscale(alpha=(0.0, 1.0)),

                            # In some images move pixels locally around (with random
                            # strengths).
                            sometimes(
                                iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                            ),

                            # In some images distort local areas with varying strength.
                            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                        ],
                        # do all of the above augmentations in random order
                        random_order=True
                    )
                ],
                # do all of the above augmentations in random order
                random_order=True
                )

def get_default_albumentations():
	# Аналог get_default_imgaug c использованием albumentations вместо imgaug.
	# https://albumentations.readthedocs.io/en/latest/examples.html
	return Compose([
                    #RandomRotate90(),
                    #Flip(),
                    #Transpose(),
                    OneOf([
                        IAAAdditiveGaussianNoise(),
                        GaussNoise(),
                    ], p=0.2),
                    OneOf([
                        MotionBlur(p=0.2),
                        MedianBlur(blur_limit=3, p=0.1),
                        Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                    OneOf([
                        OpticalDistortion(p=0.3),
                        GridDistortion(p=0.1),
                        IAAPiecewiseAffine(p=0.3),
                    ], p=0.2),
                    OneOf([
                        CLAHE(clip_limit=2),
                        IAASharpen(),
                        IAAEmboss(),
                        RandomBrightnessContrast(),
                    ], p=0.3),
                    HueSaturationValue(p=0.3),
                ], p=0.9)
def load_img(path):

    # Функция, которая подгружает .npy в память по его пути.
    
    img = np.load(path).astype('uint8')
    #img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_AREA)
    return img

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, images_paths, labels, batch_size=8, dim = (128,128,3), shuffle=True,augment = False, seq=None, seed=1, albumentations=True):
        
        # Конструктор

        self.images_paths = images_paths # Пути к картинкам, которые сохранены в .npy и ненормализованы
        self.labels = labels # Разметка картинок
        self.batch_size = batch_size # Количество подгружаемых картинок за один вызов __get_item__
        self.dim = dim # Размерность картинок. Нигде не используется, но без него не работает из-за наследования от tf.keras.utils.Sequence (мб не от этого, я хз)
        self.shuffle = shuffle # Рандомить ли порядок картинок для каждой новой эпохи. True или False.
        self.augment = augment # Использовать аугментацию. True или False.
        self.albumentations = albumentations # Использовать по умолчанию albumentations. True или False. (Нужно поменять название параметра, а то непонятно)
        
        self.on_epoch_end()

        ia.seed(seed)
        
        if seq!=None:
        	# Если есть свой набор аугментаций, то используем его
            self.seq = seq
        else:
        	# Иначе ставим дефолтный
            if self.albumentations == True:
                self.seq = get_default_albumentations()
            else:
                self.seq = get_default_imgaug()


    def __len__(self):
    	# Перегрузка len()
        return int(len(self.images_paths) / self.batch_size)
      
    def on_epoch_end(self):

    	# Функция, которую вызываем в конце эпохи.

        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):

    	# Перегрузка []

    	# Находим нидексы картинок из этого батча
        indexes = self.indexes[index * self.batch_size:(index+1)*self.batch_size]

        # y - лейблы, x - подгруженные картинки
        y = np.array([self.labels[i] for i in indexes])
        x = np.array([load_img(self.images_paths[i]) for i in indexes])
        

        if self.augment:
            x = self.augmentor(x)
        
        # Нормализуем картинки
        x = x.astype('float32')
        x-=x.min()
        x/=x.max()
        
        return x,y
    
    def augmentor(self,x):
    	# Аугментируем картинки
        
        if str(type(self.seq)) == "<class 'imgaug.augmenters.meta.Sequential'>":
            # imgaug
            print(x.shape)
            print(x)
            return self.seq(images=x)
        elif str(type(self.seq)) == "<class 'albumentations.core.composition.Compose'>":
        	# Albumentations
            for i in range(len(x)):
                x[i] = self.seq(image=x[i])['image']
            return x
    