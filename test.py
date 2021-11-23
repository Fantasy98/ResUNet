from tqdm import tqdm

from utils import *


def mask_to_3c(mask):
    """
    Convert a single-channel mask to 3-channel (256, 256, 1) -> (256, 256, 3).
    """
    mask = np.repeat(mask, 3, axis=-1)
    return mask


def write_result(model, test_x, test_y, save_path):
    """
    Write the image, ground truth and prediction into one single image.
    """
    make_dirs(save_path)

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x, y = read_and_normalize_data(x, y)

        x = np.expand_dims(x, axis=0)  # Convert from [H, W, C] to [N, H, W, C]
        y = np.expand_dims(y, axis=-1)  # Convert from [H, W] to [H, W, C]

        y_pred = K.round(model.predict(x)[0])  # Round predictions to 0 or 1 element-wise

        h = x.shape[1]
        sep_line = np.ones((h, 10, 3))

        all_images = [
            x[0], sep_line,
            mask_to_3c(y), sep_line,
            mask_to_3c(y_pred)
        ]
        all_images = [*map(lambda x: x * 255., all_images)]

        result = np.concatenate(all_images, axis=1)
        cv2.imwrite(os.path.join(save_path, f"{i}.png"), result)


def evaluate0(model_path, test_dataset_path, save_path, cross_dataset):
    """
    Evaluate the model and write the results.
    """
    test_x, test_y = load_dataset(test_dataset_path, cross_dataset=cross_dataset)

    test_dataset = tf_dataset(test_x, test_y, batch_size=batch_size, epochs=1)
    test_steps = len(test_x) // batch_size

    if len(test_x) % batch_size != 0:
        test_steps += 1

    model = load_model_weights(model_path)
    model.evaluate(test_dataset, steps=test_steps)
    write_result(model, test_x, test_y, save_path)


def evaluate(model_path, training_dataset):
    """
    Evaluate the model, including cross-dataset evaluation for its generalizability,
    in which case the test dataset is different from the training one.

    Args:
        model_path: Path from which to load the trained model.
        training_dataset: Dataset on which the model is trained.
    """
    model_name = get_filename(model_path)
    datasets = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir-SEG"]

    for test_dataset in datasets:
        if test_dataset == training_dataset:
            cross_dataset = False
            test_dataset_path = f"aug_data/{test_dataset}/test/"
        else:
            cross_dataset = True
            test_dataset_path = f"dataset/{test_dataset}/"

        save_path = f"results/{model_name}/{training_dataset}_X_{test_dataset}/"
        evaluate0(model_path, test_dataset_path=test_dataset_path, save_path=save_path, cross_dataset=cross_dataset)


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    batch_size = 8
    model_path = "logs/CVC-ClinicDB/ckpt/model.h5"  # Replace with your model path
    evaluate(model_path, training_dataset="CVC-ClinicDB")
