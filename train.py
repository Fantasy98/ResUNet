from tensorflow.keras.callbacks import *
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import *

from models.model import build_model
from utils import *


def train(training_dataset):
    """
    Train a model.
    Dealing with the input shape of a model is a thorny problem thanks to various image
    sizes of datasets. For the sake of simplicity, we use a fixed shape of (256, 256, 3) for all
    training processes.

    Args:
        training_dataset: Dataset on which to train the model.
    """
    train_path = f"aug_data/{training_dataset}/train/"
    valid_path = f"aug_data/{training_dataset}/valid/"

    train_x, train_y = load_dataset(train_path, cross_dataset=False)
    valid_x, valid_y = load_dataset(valid_path, cross_dataset=False)

    train_dataset = tf_dataset(train_x, train_y, batch_size=batch_size, epochs=epochs)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size=batch_size, epochs=epochs)

    shape = (256, 256, 3)
    model = build_model(shape)

    optimizer = Nadam(learning_rate=learning_rate)
    metrics = [
        dice_coef,
        MeanIoU(num_classes=2),
        Recall(),
        Precision()
    ]

    model.compile(loss=dice_loss, optimizer=optimizer, metrics=metrics)

    log_path = f"logs/{training_dataset}"
    model_name = model.name

    ckpt_path = f"{log_path}/ckpt/{model_name}.h5"
    csv_path = f"{log_path}/csv/{model_name}.csv"
    log_dir = f"{log_path}/fit/{model_name}"
    make_dirs(f"{log_path}/csv/")

    callbacks = [
        ModelCheckpoint(ckpt_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(factor=0.1, patience=5),
        CSVLogger(csv_path),
        TensorBoard(log_dir=log_dir),
        EarlyStopping(patience=20, restore_best_weights=False)
    ]

    train_steps = len(train_x) // batch_size
    valid_steps = len(valid_x) // batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    history = model.fit(train_dataset,
                        validation_data=valid_dataset,
                        steps_per_epoch=train_steps,
                        validation_steps=valid_steps,
                        callbacks=callbacks,
                        epochs=epochs,
                        shuffle=False)

    return history


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    batch_size = 8
    epochs = 250
    learning_rate = 1e-3

    train(training_dataset="CVC-ClinicDB")
