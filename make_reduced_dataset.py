import os
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from tqdm import tqdm
import numpy as np
from tqdm import tqdm

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org',
    password='international',
    cache_dir="./data"
)

def make_session_dataset(eid, probe_name):
    # load curated units
    sl = SpikeSortingLoader(eid=eid, pname=probe_name, one=one)
    spikes, clusters, channels = sl.load_spike_sorting(good_units=True)
    clusters = sl.merge_clusters(spikes, clusters, channels)
    
    if (not hasattr(clusters, "x") or len(clusters.x) == 0 or
    not hasattr(clusters, "atlas_id") or len(clusters.atlas_id) == 0):
        return None
    
    # Safety check: for standard IBL outputs cluster_id should match row index
    cluster_ids = np.asarray(clusters.cluster_id).astype(int)
    assert np.array_equal(cluster_ids, np.arange(len(cluster_ids))), \
        f"cluster_id != row index for eid={eid}, probe={probe_name}"

    # Accepted neurons must come from the filtered spikes object
    accepted_ids = np.unique(np.asarray(spikes.clusters).astype(int)) 
    
    # make neuron dataset
    data = {}
    for neuron_id in accepted_ids:
        temp = {}
        firing_rate = float(clusters.firing_rate[neuron_id])
        if firing_rate <= 0.05: continue
        
        temp["firing_rate"] = firing_rate
        temp["neuron_spike_times"] = spikes.times[spikes.clusters == neuron_id]
        temp["n_spikes"] = len(temp["neuron_spike_times"])
        temp["acronym"] = str(clusters.acronym[neuron_id])
        temp["atlas_id"] = int(clusters.atlas_id[neuron_id])
        temp["coord"] = np.array([
            clusters.x[neuron_id], 
            clusters.y[neuron_id], 
            clusters.z[neuron_id]
        ]).astype(float)
        temp["hemisphere"] = "L" if temp["coord"][0] < 0 else "R"
        temp["histology_final"] = sl.histology in {"resolved", "alf"}
        data[int(neuron_id)] = temp
        
    return data

if __name__ == "__main__":
    eids = np.load("eids.npy")

    os.makedirs("output", exist_ok=True)

    for eid in tqdm(eids):
        session_data = {
            "session_id": str(eid),
            "neural_data": {}
        }
            
        _, pnames = one.eid2pid(eid)
        
        for probe_name in pnames:
            probe_data = make_session_dataset(eid, probe_name)
            if probe_data is not None:
                session_data["neural_data"][probe_name] = probe_data
            
        np.save(f"output/{eid}.npy", session_data)