import os
import matplotlib.pyplot as plt
from statistics import median
import pandas as pd
import numpy as np
from pprint import pprint

def make_hist_plot(data, title, fname):
    x = plt.hist(data, color = 'blue', edgecolor = 'black', bins = 500)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('MPI Ranks')
    med = median(data)
    min_ylim, max_ylim = plt.ylim()
    plt.axvline(med, color='red', linestyle='dashed', linewidth=1)
    plt.text(med, max_ylim*0.9, '   Median: {:.2f}'.format(med))
    plt.savefig(fname)
    plt.clf()

def make_stats(run_name, timing_dict):
    data = dict()
    for k, v in timing_dict.items():
        if len(v) > 1:
            array = np.array(v)
            arr_min = np.min(array)
            arr_max = np.max(array)
            arr_mean = np.mean(array)

            data["_".join((k, "min"))] = arr_min
            data["_".join((k, "mean"))] = arr_mean
            data["_".join((k, "max"))] = arr_max
        else:
            data[k] = np.float(v[0])
        data["Name"] = run_name
    return data

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--name", type=str, default="infer-sess")
    args = parser.parse_args()

    files = os.listdir(args.path)

    function_times = {}
    function_rank_times = {}

    for file in files:
        if '.csv' in file:
            fp = os.path.join(args.path, file)
            with open(fp) as f:
                for i, line in enumerate(f):
                    vals = line.split(',')
                    if vals[1] in function_times.keys():
                        function_times[vals[1]].append(float(vals[2]))
                    else:
                        function_times[vals[1]] = [float(vals[2])]
            os.remove(fp)
        else:
            print(file)

    num_run = len(function_times['run_model'])
    print(f'there are {num_run} values in the run_model entries')
    print('Max {0}'.format(max(function_times['client()'])))
    print('Min {0}'.format(min(function_times['client()'])))

    make_hist_plot(function_times['client()'], 'client()', 'client_constructor_dist.pdf')
    make_hist_plot(function_times['run_script'], 'run_script()', 'run_script.pdf')
    make_hist_plot(function_times['run_model'], 'run_model()', 'run_model.pdf')
    make_hist_plot(function_times['put_tensor'], 'put_tensor()', 'put_tensor.pdf')
    make_hist_plot(function_times['get_tensor'], 'get_tensor()', 'get_tensor.pdf')
    make_hist_plot(function_times['main()'], 'main()', 'main.pdf')

    # get stats
    data = make_stats(args.name, function_times)
    data_df = pd.DataFrame(data, index=[0])
    file_name = ".".join((args.name, "csv"))
    data_df.to_csv(file_name)