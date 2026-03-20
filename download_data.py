import os
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
import numpy as np

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org',
    password='international',
    cache_dir="./data"
)

def download_data(eid):
    _, pnames = one.eid2pid(eid) # extract probe names
    for pname in pnames:
        sl = SpikeSortingLoader(eid=eid, pname=pname, one=one)
        sl.load_spike_sorting(good_units=True)
        
if __name__ == "__main__":
    eids = np.load("eids.npy")
    
    for eid in eids:
        download_data(eid)