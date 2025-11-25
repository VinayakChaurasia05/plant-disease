import tensorflow as tf

def build_plant_disease_model(input_shape=(128, 128, 3), num_classes=38):
    cnn = tf.keras.models.Sequential()

    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Dropout(0.5))

    cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Dropout(0.6))

    cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    cnn.add(tf.keras.layers.Dropout(0.25))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=1500, activation='relu'))
    cnn.add(tf.keras.layers.Dropout(0.4))

    cnn.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    cnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return cnn


def main():
    model = build_plant_disease_model()
    model.summary()
    # Save an untrained model with the same architecture used in the script.
    model.save('trained_plant_disease_model.keras')
    print("Saved untrained model to 'trained_plant_disease_model.keras'.")


if __name__ == '__main__':
    main()
