import numpy as np

#position = np.array([381.98, 24.32, 290.61])

position = np.array([81.6930770874023, 0.595929861068726, 48.8013076782227])

#marker = np.array([379.6, 24.43, 288.9])

marker = np.array([47.45000076,  3.05375004, 36.11249924])

distance_to_next_marker = np.linalg.norm(position - marker)

print(distance_to_next_marker)


