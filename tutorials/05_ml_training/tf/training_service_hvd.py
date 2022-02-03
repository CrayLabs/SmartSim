import tensorflow.keras as keras
import tensorflow as tf

from smartsim.ml.tf import DynamicDataGenerator

import horovod.tensorflow.keras as hvd

# HVD initialization
hvd.init()
hvd_rank = hvd.rank()
hvd_size = hvd.size()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


training_generator = DynamicDataGenerator(cluster=False, init_samples=True, replica_rank=hvd_rank, num_replicas=hvd_size)
model = keras.applications.MobileNetV2(weights=None, classes=training_generator.num_classes)

opt = keras.optimizers.Adam(0.001 * hvd.size())
# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)
model.compile(optimizer=opt, loss="mse", metrics=["mae"])
callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

print("Starting training")

for epoch in range(100):
    model.fit(training_generator, steps_per_epoch=None, callbacks=callbacks,
              epochs=epoch+1, initial_epoch=epoch, batch_size=training_generator.batch_size,
              verbose=2 if hvd_rank==0 else 0)
