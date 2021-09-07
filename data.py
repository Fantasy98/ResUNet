from albumentations import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import *


def augment_data(images, masks, save_path, augment=True):
    """
    Data augmentation using albumentations.
    These datasets have various image sizes. For simplicity, we resize all images
    to a fixed size of (288, 384) in accordance with the model input shape.

    See: https://github.com/albumentations-team/albumentations
    """
    size = (384, 288)  # [W, H]
    crop_size = (288 - 32, 384 - 32)  # [H, W]

    for image, mask in tqdm(zip(images, masks), total=len(images)):
        image_name = get_filename(image)
        mask_name = get_filename(mask)

        x, y = read_data(image, mask)

        if augment:
            # Choose the minimum of crop size and image shape in case of overflow
            aug = CenterCrop(p=1, height=min(crop_size[0], x.shape[0]), width=min(crop_size[1], x.shape[1]))
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            aug = RandomCrop(p=1, height=min(crop_size[0], x.shape[0]), width=min(crop_size[1], x.shape[1]))
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = HorizontalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = VerticalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = RandomRotate90(p=1)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            aug = ShiftScaleRotate(p=1)
            augmented = aug(image=x, mask=y)
            x6 = augmented['image']
            y6 = augmented['mask']

            aug = Transpose(p=1)
            augmented = aug(image=x, mask=y)
            x7 = augmented['image']
            y7 = augmented['mask']

            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x8 = augmented['image']
            y8 = augmented['mask']

            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x9 = augmented['image']
            y9 = augmented['mask']

            # Grayscale
            x10 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            y10 = y

            aug = CenterCrop(p=1, height=min(crop_size[0], x.shape[0]), width=min(crop_size[1], x.shape[1]))
            augmented = aug(image=x10, mask=y10)
            x11 = augmented['image']
            y11 = augmented['mask']

            aug = HorizontalFlip(p=1)
            augmented = aug(image=x10, mask=y10)
            x12 = augmented['image']
            y12 = augmented['mask']

            aug = VerticalFlip(p=1)
            augmented = aug(image=x10, mask=y10)
            x13 = augmented['image']
            y13 = augmented['mask']

            aug = RandomBrightnessContrast(p=1)
            augmented = aug(image=x, mask=y)
            x14 = augmented['image']
            y14 = augmented['mask']

            aug = RandomGamma(p=1)
            augmented = aug(image=x, mask=y)
            x15 = augmented['image']
            y15 = augmented['mask']

            aug = HueSaturationValue(p=1)
            augmented = aug(image=x, mask=y)
            x16 = augmented['image']
            y16 = augmented['mask']

            aug = CLAHE(p=1)
            augmented = aug(image=x, mask=y)
            x17 = augmented['image']
            y17 = augmented['mask']

            aug = Blur(p=1)
            augmented = aug(image=x, mask=y)
            x18 = augmented['image']
            y18 = augmented['mask']

            aug = MotionBlur(p=1)
            augmented = aug(image=x, mask=y)
            x19 = augmented['image']
            y19 = augmented['mask']

            aug = MedianBlur(p=1)
            augmented = aug(image=x, mask=y)
            x20 = augmented['image']
            y20 = augmented['mask']

            aug = GaussianBlur(p=1)
            augmented = aug(image=x, mask=y)
            x21 = augmented['image']
            y21 = augmented['mask']

            aug = GaussNoise(p=1)
            augmented = aug(image=x, mask=y)
            x22 = augmented['image']
            y22 = augmented['mask']

            aug = RGBShift(p=1)
            augmented = aug(image=x, mask=y)
            x23 = augmented['image']
            y23 = augmented['mask']

            aug = ChannelShuffle(p=1)
            augmented = aug(image=x, mask=y)
            x24 = augmented['image']
            y24 = augmented['mask']

            aug = CoarseDropout(p=1, max_holes=8, max_height=32, max_width=32)
            augmented = aug(image=x, mask=y)
            x25 = augmented['image']
            y25 = augmented['mask']

            images = [
                x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
                x21, x22, x23, x24, x25
            ]
            masks = [
                y, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10,
                y11, y12, y13, y14, y15, y16, y17, y18, y19, y20,
                y21, y22, y23, y24, y25
            ]

        else:
            images = [x]
            masks = [y]

        for idx, (i, m) in enumerate(zip(images, masks)):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{image_name}_{idx}.png"
            tmp_mask_name = f"{mask_name}_{idx}.png"

            image_path = os.path.join(save_path, "images/", tmp_image_name)
            mask_path = os.path.join(save_path, "masks/", tmp_mask_name)

            i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)  # Convert to BGR color space before calling #cv2.imwrite
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)


def load_data(path, split=0.1):
    """
    Load the data and split them into random train, validation and test subsets.
    """
    images, masks = load_dataset(path)

    test_size = int(split * len(images))

    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def augment(dataset):
    """
    Apply augmentation to the training dataset while keeping the validation and test datasets as they are.
    """
    data_path = f"dataset/{dataset}/"
    aug_data_path = f"aug_data/{dataset}/"

    for dir in ["train", "valid", "test"]:
        for subdir in ["images", "masks"]:
            create_dirs(os.path.join(aug_data_path, dir, subdir))

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(data_path)

    augment_data(train_x, train_y, save_path=os.path.join(aug_data_path, "train/"), augment=True)
    augment_data(valid_x, valid_y, save_path=os.path.join(aug_data_path, "valid/"), augment=False)
    augment_data(test_x, test_y, save_path=os.path.join(aug_data_path, "test/"), augment=False)


if __name__ == "__main__":
    np.random.seed(42)

    datasets = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir-SEG"]

    for dataset in datasets:
        augment(dataset=dataset)
