import numpy as np
from smartredis import Client, Dataset
from smartsim.ml import TrainingDataUploader

import numpy as np
from tqdm import tqdm

from steady_state import fd2d_heat_steady_test01

def augment_batch(samples, targets):
    """Augment samples and targets
    
    by exploiting rotational and axial symmetry. Each sample is 
    rotated and reflected to obtain 8 valid samples. The same
    transformations are applied to targets.

    Samples and targets must be 4-dimensional batches,
    following NWHC ordering.

    :param samples: Samples to augment
    :param targets: Targets to augment

    :returns: Tuple of augmented samples and targets
    """
    batch_size = samples.shape[0]
    augmented_samples = np.empty((batch_size*8, *samples.shape[1:]))
    augmented_targets = np.empty_like(augmented_samples)

    aug = 0
    augmented_samples[batch_size*aug:batch_size*(1+aug), :, :, :] = samples
    augmented_targets[batch_size*aug:batch_size*(1+aug), :, :, :] = targets

    aug = 1
    samples = np.rot90(samples, k=1, axes=[1,2])
    targets = np.rot90(targets, k=1, axes=[1,2])
    augmented_samples[batch_size*aug:batch_size*(1+aug), :, :, :] = samples
    augmented_targets[batch_size*aug:batch_size*(1+aug), :, :, :] = targets

    aug = 2
    samples = np.rot90(samples, k=1, axes=[1,2])
    targets = np.rot90(targets, k=1, axes=[1,2])
    augmented_samples[batch_size*aug:batch_size*(1+aug), :, :, :] = samples
    augmented_targets[batch_size*aug:batch_size*(1+aug), :, :, :] = targets

    aug = 3
    samples = np.rot90(samples, k=1, axes=[1,2])
    targets = np.rot90(targets, k=1, axes=[1,2])
    augmented_samples[batch_size*aug:batch_size*(1+aug), :, :, :] = samples
    augmented_targets[batch_size*aug:batch_size*(1+aug), :, :, :] = targets

    aug = 4
    samples = np.flip(samples, 1)
    targets = np.flip(targets, 1)
    augmented_samples[batch_size*aug:batch_size*(1+aug), :, :, :] = samples
    augmented_targets[batch_size*aug:batch_size*(1+aug), :, :, :] = targets

    aug = 5
    samples = np.rot90(samples, k=1, axes=[1,2])
    targets = np.rot90(targets, k=1, axes=[1,2])
    augmented_samples[batch_size*aug:batch_size*(1+aug), :, :, :] = samples
    augmented_targets[batch_size*aug:batch_size*(1+aug), :, :, :] = targets

    aug = 6
    samples = np.rot90(samples, k=1, axes=[1,2])
    targets = np.rot90(targets, k=1, axes=[1,2])
    augmented_samples[batch_size*aug:batch_size*(1+aug), :, :, :] = samples
    augmented_targets[batch_size*aug:batch_size*(1+aug), :, :, :] = targets

    aug = 7
    samples = np.rot90(samples, k=1, axes=[1,2])
    targets = np.rot90(targets, k=1, axes=[1,2])
    augmented_samples[batch_size*aug:batch_size*(1+aug), :, :, :] = samples
    augmented_targets[batch_size*aug:batch_size*(1+aug), :, :, :] = targets

    return augmented_samples, augmented_targets

def simulate(steps, size):
    """Run multiple simulations and upload results
    
    both as tensors and as augmented samples for training.

    :param steps: Number of simulations to run
    :param size: lateral size of the discretized domain
    """
    batch_size = 50
    samples = np.zeros((batch_size,size,size,1)).astype(np.single)
    targets = np.zeros_like(samples).astype(np.single)
    client = Client(None, False)

    training_data_uploader = TrainingDataUploader(cluster=False, verbose=True)
    training_data_uploader.publish_info()

    for i in tqdm(range(steps)):
        
        u_init, u_steady = fd2d_heat_steady_test01(samples.shape[1], samples.shape[2])
        u_init = u_init.astype(np.single)
        u_steady = u_steady.astype(np.single)
        dataset = create_dataset(i, u_init, u_steady)
        client.put_dataset(dataset)

        samples[i%batch_size, :, :, 0] = u_init
        targets[i%batch_size, :, :, 0] = u_steady

        if (i+1)%batch_size == 0:
            augmented_samples, augmented_targets = augment_batch(samples, targets)
            training_data_uploader.put_batch(augmented_samples, augmented_targets)


def create_dataset(idx, u_init, u_steady):
    """Create SmartRedis Dataset containing multiple NumPy arrays
    to be stored at a single key within the database"""
    dataset = Dataset(f"sim_data_{idx}")
    dataset.add_tensor("u_steady", np.expand_dims(u_steady, axis=[0,-1]))
    dataset.add_tensor("u_init", np.expand_dims(u_init, axis=[0,-1]))
    return dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finite Difference Simulation")
    parser.add_argument('--steps', type=int, default=4000,
                        help='Number of simulations to run')
    parser.add_argument('--size', type=int, default=100,
                        help='Size of sample side, each sample will be a (size, size, 1) image')
    args = parser.parse_args()
    simulate(args.steps, size=args.size)