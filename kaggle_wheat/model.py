import tensorflow as tf

def custom_loss(y_true, y_pred):
    prob_loss = tf.keras.losses.binary_crossentropy(
        y_true[:, :, :, 0],
        y_pred[:, :, :, 0]
    )

    bboxes_mask = tf.where(
        y_true[:, :, :, 0] == 0,
        0.5,
        5.0
    )

    xy_loss = tf.keras.losses.MSE(
        y_true[:, :, :, 1:3],
        y_pred[:, :, :, 1:3]
    )

    wh_loss = tf.keras.losses.MSE(
        y_true[:, :, :, 3:5],
        y_pred[:, :, :, 3:5]
    )

    xy_loss = xy_loss * bboxes_mask
    wh_loss = wh_loss * bboxes_mask

    return prob_loss + xy_loss + wh_loss

def create_model():
    custom_model_head = tf.keras.models.Sequential([
        tf.keras.layers.ZeroPadding2D(padding=(1, 1), input_shape = (8, 8, 2048)),
        tf.keras.layers.Conv2D(2048, (3, 3), strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(2048, (3, 3), strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(2048, (3, 3), strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(1028, (3, 3), strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(8, (3, 3), strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(5, (3, 3), strides=(1, 1), padding='same', activation='sigmoid'),
        ])

    backbone = tf.keras.applications.ResNet152V2(
        include_top=False, weights='imagenet', input_shape=(256, 256, 3),
    )
    backbone.trainable = False

    model = tf.keras.Sequential([backbone, custom_model_head])
    return model

def compile_model(model):
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimiser,
        loss=custom_loss
    )
    return model

def get_model_and_callbacks():
    model = create_model()
    model = compile_model(model)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True),
    ]
    return model, callbacks