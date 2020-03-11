#!/bin/env python
import smartsim
import numpy as np
import matplotlib.pyplot as plt
import time
cluster = smartsim.clients.Client(cluster=True)
cluster.setup_connections()
nranks = 1440

def retrieve_all_ranks(key_suffix, nranks, dtype="float64"):
  retrieve_d = {}
  for rank in range(nranks):
      rank_id= f'{rank:06d}'
      key = f'{rank_id}_{key_suffix}'
      retrieve_d[rank_id] = cluster.get_data(f'{rank_id}_{key_suffix}',dtype, wait=True)
  return retrieve_d

start_total = time.time()

for iter in range(1):
    start_iter = time.time()
    
    meta_rank = retrieve_all_ranks(f'rank-meta',nranks)
    print(f"retrieve_all_ranks: {time.time()-start_iter}")
    starti_rank  = np.array([int(meta[0]) for meta in meta_rank.values()])
    startj_rank  = np.array([int(meta[2]) for meta in meta_rank.values()])
    glob_starti_rank = starti_rank - min(starti_rank)
    glob_startj_rank = startj_rank - min(startj_rank)
    
    timestamp = '63292754700.00'
    t1 = time.time()
    h_rank = retrieve_all_ranks(f'{timestamp}_h',nranks)
    print(f"retreive_all_ranks h: {time.time()-t1}")
    
    ni_rank = np.array( [ h.shape[-1] for h in h_rank.values()] )
    nj_rank = np.array( [ h.shape[-2] for h in h_rank.values()] )
    nk_rank = np.array( [ h.shape[-3] for h in h_rank.values()] )
    
    glob_endi_rank = glob_starti_rank + ni_rank - 1
    glob_endj_rank = glob_startj_rank + nj_rank - 1
    
    ni_glob = max(glob_endi_rank) + 1 
    nj_glob = max(glob_endj_rank) + 1
    nk_glob = max(nk_rank)
    
    h_glob = np.zeros([nk_glob,nj_glob,ni_glob])
    
    for rank in range(nranks):
        si = glob_starti_rank[rank]
        sj = glob_startj_rank[rank]
        ei = glob_endi_rank[rank] + 1
        ej = glob_endj_rank[rank] + 1
    
        h_glob[:,sj:ej,si:ei] = h_rank[f'{rank:06d}']
    iter_time = time.time() - start_iter
    print(f"Time elapsed in iteration: {iter_time}")

print(f"Array size: {h_glob.shape}")
print(f"Total time elapsed: {time.time() - start_total}")
# Don't plot for now because we primarily care about the timings
#plt.pcolormesh(h_glob[30,:,:],vmin=0,vmax=200,cmap=plt.cm.RdBu_r)
#plt.show()
