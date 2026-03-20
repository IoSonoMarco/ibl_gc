import os
from one.api import ONE
import numpy as np

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    
    one = ONE(
        base_url='https://openalyx.internationalbrainlab.org',
        password='international',
        cache_dir="./data"
    )

    eids_1 = one.search(project='brainwide', datasets='spikes.times.npy')
    eids_2 = one.search(tag="Brainwidemap")
    eids = list(set(eids_1).intersection(eids_2))
    np.save("eids", np.array(eids).astype(str))