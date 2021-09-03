from datetime import datetime

from tensorflow.keras.callbacks import *
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Nadam

from model import build_model
from utils import *


def train(training_dataset):
    train_path = f"aug_data/{training_dataset}/train/"
    valid_path = f"aug_data/{training_dataset}/valid/"

    train_x, train_y = load_dataset(train_path, cross_dataset=False)
    train_x, train_y = shuffle_data(train_x, train_y)  # Shuffle
    valid_x, valid_y = load_dataset(valid_path, cross_dataset=False)

    train_dataset = tf_dataset(train_x, train_y, batch_size=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size=batch_size)

    shape = (288, 384, 3)
    model = build_model(shape)

    optimizer = Nadam(learning_rate=learning_rate)
    metrics = [
        dice_coef,
        MeanIoU(num_classes=2),
        Recall(),
        Precision()
    ]

    model.compile(loss=dice_loss, optimizer=optimizer, metrics=metrics)

    callbacks = [
        ModelCheckpoint(ckpt_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(factor=0.1, patience=20),
        CSVLogger(csv_path),
        TensorBoard(log_dir=log_dir),
        EarlyStopping(patience=20, restore_best_weights=False)
    ]

    train_steps = (len(train_x) // batch_size)
    valid_steps = (len(valid_x) // batch_size)

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    model.fit(train_dataset,
              validation_data=valid_dataset,
              steps_per_epoch=train_steps,
              validation_steps=valid_steps,
              callbacks=callbacks,
              epochs=epochs,
              shuffle=False)


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    # Place the logs in a timestamped subdirectory to allow easy selection of different training runs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    ckpt_path = "logs/ckpt/" + timestamp + ".h5"
    csv_path = "logs/csv/" + timestamp + ".csv"
    log_dir = "logs/fit/" + timestamp
    create_dir("logs/csv/")

    batch_size = 4
    epochs = 250
    learning_rate = 1e-5

    train(training_dataset="CVC-ClinicDB")
