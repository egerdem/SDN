import pandas as pd
from IPython.display import display

# Ordered method labels based on mapping
methods = [
    "HO-SDN (N=3)", "HO-SDN (N=2)", "SDN Original (c=1)",
    "SW-SDN (c=-3)", "SW-SDN (c=-2)", "SW-SDN (c=2)", "SW-SDN (c=3)",
    "SW-SDN (c=4)", "SW-SDN (c=5)", "SW-SDN (c=6)", "SW-SDN (c=7)"
]

# Fill values manually extracted from the images
room_a = [
    [1.467477, 5.709701, 0.91],
    [2.296093, 4.635993, 0.93],
    [2.323439, 3.152451, 0.93],
    [0.303824, 6.978518, 0.90],
    [0.553081, 5.329914, 0.91],
    [1.910303, 3.349412, 0.93],
    [1.129958, 4.007765, 0.91],
    [0.518003, 5.127509, 0.91],
    [0.423950, 6.708644, 0.90],
    [0.643943, 8.751171, 0.90],
    [0.840376, 11.255088, 0.89]
]

room_w = [
    [1.071433, 24.948321, 1.63],
    [0.959834, 20.553142, 1.63],
    [0.845186, 13.994011, 1.62],
    [0.095281, 48.703960, 1.61],
    [0.178597, 33.798040, 1.61],
    [0.650259, 15.697244, 1.62],
    [0.324976, 21.552767, 1.62],
    [0.111660, 31.560577, 1.61],
    [0.127018, 45.720676, 1.61],
    [0.200977, 64.033064, 1.61],
    [0.252359, 86.497740, 1.61]
]

room_j = [
    [1.360128, 24.016110, 0.92],
    [1.546182, 16.949303, 0.92],
    [0.836875, 11.249598, 0.91],
    [0.214999, 40.479702, 0.90],
    [0.155589, 28.011671, 0.90],
    [0.591526, 12.542914, 0.91],
    [0.242231, 17.276566, 0.90],
    [0.223353, 25.450556, 0.90],
    [0.350950, 37.064882, 0.90],
    [0.439162, 52.119545, 0.90],
    [0.494580, 70.614545, 0.90]
]

# Create DataFrame
df = pd.DataFrame({
    "Method": methods,
    "Room A RMSE": [r[0] for r in room_a],
    "Room A Energy": [r[1] for r in room_a],
    "Room A RT60": [r[2] for r in room_a],
    "Room W RMSE": [r[0] for r in room_w],
    "Room W Energy": [r[1] for r in room_w],
    "Room W RT60": [r[2] for r in room_w],
    "Room J RMSE": [r[0] for r in room_j],
    "Room J Energy": [r[1] for r in room_j],
    "Room J RT60": [r[2] for r in room_j],
})

import ace_tools as tools; tools.display_dataframe_to_user(name="RMSE, Energy, and RT60 Table", dataframe=df)
