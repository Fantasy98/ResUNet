from glob import glob

from tensorflow.keras.metrics import Recall, Precision, MeanIoU
from tensorflow.keras.optimizers import Nadam
from tqdm import tqdm

from utils import *


def read_image(x):
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = np.clip(image - np.median(image) + 127, 0, 255)
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def read_mask(y):
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask


def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


def evaluate_normal(model, x_data, y_data):
    for i, (x, y) in tqdm(enumerate(zip(x_data, y_data)), total=len(x_data)):
        x = read_image(x)
        y = read_mask(y)
        _, h, w, _ = x.shape

        y_pred = model.predict(x)[0]
        y_pred = (y_pred > 0.5) * 255.0

        line = np.ones((h, 10, 3)) * 255.0

        all_images = [
            x[0] * 255.0, line,
            mask_to_3d(y) * 255.0, line,
            mask_to_3d(y_pred)
        ]
        result = np.concatenate(all_images, axis=1)

        cv2.imwrite(f"results/{i}.png", result)


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    model_path = "logs/ckpt/timestamp.h5"
    create_dir("results/")

    batch_size = 8
    lr = 1e-5

    optimizer = Nadam(learning_rate=lr)
    metrics = [
        dice_coef,
        MeanIoU(num_classes=2),
        Recall(),
        Precision()
    ]

    test_path = "aug_data/test/"

    test_x = sorted(glob(os.path.join(test_path, "image", "*.jpg")))
    test_y = sorted(glob(os.path.join(test_path, "mask", "*.jpg")))

    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x) // batch_size)

    if len(test_x) % batch_size != 0:
        test_steps += 1

    model = load_model_weight(model_path)
    model.evaluate(test_dataset, steps=test_steps)
    evaluate_normal(model, test_x, test_y)
