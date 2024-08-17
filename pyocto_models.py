# %% Imports and Configuration
import numpy as np
import pandas as pd
import pyocto
import tqdm

from src.clustering.dataset import (GaMMAPickFormat, GaMMAStationFormat,
                                    PhasePicksDataset)
from src.clustering.models import run_pyocto
from src.clustering.utils import ClusterStatistics, plot_arrivals

velocity_model = pyocto.VelocityModel0D(
    p_velocity=5.5,  # 5.5,
    s_velocity=2.7,  # 2.7,
    tolerance=0.1,
)

associator = pyocto.OctoAssociator.from_area(
    lat=(46.4, 46.6),
    lon=(8.4, 8.6),
    zlim=(-1, -2),
    # lat=(46,47),
    # lon=(8,9),
    # zlim=(-1,-3),

    time_before=0.1,  # 300,

    velocity_model=velocity_model,

    # n_picks=4,
    # n_p_and_s_picks=4,
    n_p_picks=4,
    n_s_picks=4,

    min_node_size=5e-1,  # 10.0,
    min_node_size_location=0.75e-1,  # 1.5,

    pick_match_tolerance=1.5,  # 1.5,

    min_interevent_time=0.0001,  # 3.0,
    exponential_edt=False,
    edt_pick_std=1.0,  # 1.0,
    max_pick_overlap=4,  # 4,

    refinement_iterations=3,  # 3,
    time_slicing=0.5,  # 1200.0,
    node_log_interval=0,  # 0,

    location_split_depth=6,  # 6,
    location_split_return=4,  # 4,
    min_pick_fraction=0.25
)

# %% Run PyOcto
statistics = ClusterStatistics()

ds = PhasePicksDataset(
    root_dir='data/raw',
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=GaMMAPickFormat(),
    station_transform=GaMMAStationFormat()
)

for sample in tqdm.tqdm(ds):
    events, labels_pred = run_pyocto(sample.x, ds.stations, associator)

    statistics.add(sample.y.to_numpy(),
                   labels_pred,
                   len(sample.y.unique())-1,
                   len(np.unique(labels_pred))-1)

print(f"PyOcto ARI: {statistics.ari()}, Accuray: {statistics.accuracy()}, "
      f"Precision: {statistics.precision()}, Recall: {statistics.recall()}")
print(f"PyOcto discovered {statistics.perc_eq()}% of the events correctly.")

# %% Plot Results
associations = sample.x.copy().join(ds.stations.set_index('id'), on='id')
associations['dx'] = PhasePicksDataset.get_distance(
    associations, ['x(km)', 'y(km)', 'z(km)'])*1000
associations['time'] = pd.to_datetime(
    associations['timestamp'], unit='ns').values.astype(int)

plot_arrivals(associations[['dx', 'time']],
              events[['dx', 'time']],
              events[['dx', 'time']],
              sample.y.to_numpy(), labels_pred)

# %%
