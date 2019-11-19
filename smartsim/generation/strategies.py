
# Generation Strategies

from itertools import product

# create permutations of all parameters
# single model if parameters only have one value
def _create_all_permutations(param_names, param_values):
    perms = list(product(*param_values))
    all_permutations = []
    for p in perms:
        temp_model = dict(zip(param_names, p))
        all_permutations.append(temp_model)
    return all_permutations

def _step_values(param_names, param_values):
    permutations = []
    for p in zip(*param_values):
        permutations.append(dict(zip(param_names, p)))
    return permutations

def _random_permutations(param_names, param_values, n_models):
    import random
    # first, check if we've requested more values than possible.
    perms = list(product(*param_values))
    if n_models >= len(perms):
        return _create_all_permutations(param_names, param_values)
    else:
        permutations = []
        permutation_strings = set()
        while len(permutations) < n_models:
            model_dict = dict(zip(param_names, map(lambda x: x[random.randint(0,len(x)-1)], param_values)))
            if str(model_dict) not in permutation_strings:
                permutation_strings.add(str(model_dict))
                permutations.append(model_dict)
        return permutations