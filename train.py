from datetime import datetime
from glob import glob

from tensorflow.keras.callbacks import *
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Nadam

from model import build_model
from utils import *

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    train_path = "aug_data/train/"
    valid_path = "aug_data/valid/"

    # Training
    train_x = sorted(glob(os.path.join(train_path, "image", "*.jpg")))
    train_y = sorted(glob(os.path.join(train_path, "mask", "*.jpg")))

    # Shuffling
    train_x, train_y = shuffling(train_x, train_y)

    # Validation
    valid_x = sorted(glob(os.path.join(valid_path, "image", "*.jpg")))
    valid_y = sorted(glob(os.path.join(valid_path, "mask", "*.jpg")))

    # Place the logs in a timestamped subdirectory to allow easy selection of different training runs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    ckpt_path = "logs/ckpt/" + timestamp + ".h5"
    csv_path = "logs/csv/" + timestamp + ".csv"
    log_dir = "logs/fit/" + timestamp
    create_dir("logs/csv/")

    batch_size = 4
    epochs = 250
    lr = 1e-5

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    shape = (288, 384, 3)
    model = build_model(shape)

    optimizer = Nadam(learning_rate=lr)
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
