import numpy as np
import tensorflow.keras as keras

from smartsim.ml.tf import DataGenerator


training_generator = DataGenerator(smartredis_cluster=False, verbose=True)
model = keras.applications.MobileNetV2(weights=None, classes=training_generator.num_classes)
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

print("starting training")

for epoch in range(100):
    print(f"Epoch {epoch}")
    model.fit(training_generator, steps_per_epoch=None, 
              epochs=epoch+1, initial_epoch=epoch, batch_size=training_generator.batch_size,
              verbose=2)
