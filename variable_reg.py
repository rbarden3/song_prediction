from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

X = [[0.904, 0.813, 4.0, -7.105, 0.0, 0.121, 0.0311, 0.00697, 0.0471, 0.81, 125.461, 226864.0, 4.0],
 [0.774, 0.838, 5.0, -3.914, 0.0, 0.114, 0.0249, 0.025, 0.242, 0.924, 143.04, 198800.0, 4.0],
 [0.664, 0.758, 2.0, -6.583, 0.0, 0.21, 0.00238, 0.0, 0.0598, 0.701, 99.259, 235933.0, 4.0],
 [0.892, 0.714, 4.0, -6.055, 0.0, 0.141, 0.201, 0.000234, 0.0521, 0.817, 100.972, 267267.0, 4.0],
 [0.853, 0.606, 0.0, -4.596, 1.0, 0.0713, 0.0561, 0.0, 0.313, 0.654, 94.759, 227600.0, 4.0],
 [0.881, 0.788, 2.0, -4.669, 1.0, 0.168, 0.0212, 0.0, 0.0377, 0.592, 104.997, 250373.0, 4.0],
 [0.662, 0.507, 5.0, -8.238, 1.0, 0.118, 0.257, 0.0, 0.0465, 0.676, 86.412, 223440.0, 4.0],
 [0.57, 0.821, 2.0, -4.38, 1.0, 0.267, 0.178, 0.0, 0.289, 0.408, 210.857, 225560.0, 4.0],
 [0.713, 0.678, 5.0, -3.525, 0.0, 0.102, 0.273, 0.0, 0.149, 0.734, 138.009, 271333.0, 4.0],
 [0.727, 0.974, 4.0, -2.261, 0.0, 0.0664, 0.103, 0.000532, 0.174, 0.965, 79.526, 235213.0, 4.0],
 [0.808, 0.97, 10.0, -6.098, 0.0, 0.0506, 0.0569, 6.13e-05, 0.154, 0.868, 114.328, 242293.0, 4.0],
 [0.71, 0.553, 4.0, -4.722, 0.0, 0.0292, 0.00206, 5.48e-05, 0.0469, 0.731, 99.005, 211693.0, 4.0],
 [0.66, 0.666, 9.0, -4.342, 1.0, 0.0472, 0.0759, 0.0, 0.0268, 0.933, 89.975, 214227.0, 4.0],
 [0.687, 0.71, 9.0, -5.84, 1.0, 0.0522, 0.0283, 3.71e-06, 0.0689, 0.886, 79.235, 216880.0, 4.0],
 [0.803, 0.454, 8.0, -4.802, 0.0, 0.0294, 0.352, 0.0, 0.0655, 0.739, 99.99, 192213.0, 4.0],
 [0.775, 0.731, 8.0, -5.446, 1.0, 0.134, 0.189, 0.0, 0.129, 0.821, 131.103, 256427.0, 4.0],
 [0.487, 0.9, 0.0, -4.417, 1.0, 0.0482, 6.79e-05, 0.0, 0.358, 0.484, 149.937, 204000.0, 4.0],
 [0.846, 0.482, 1.0, -6.721, 0.0, 0.129, 0.0246, 0.0, 0.393, 0.212, 100.969, 229867.0, 4.0],
 [0.705, 0.796, 7.0, -6.845, 1.0, 0.267, 0.0708, 0.0, 0.388, 0.864, 166.042, 210453.0, 4.0],
 [0.771, 0.685, 1.0, -4.639, 1.0, 0.0567, 0.00543, 0.00157, 0.0537, 0.683, 88.997, 230200.0, 4.0],
 [0.717, 0.733, 4.0, -4.985, 1.0, 0.0427, 0.0398, 0.0, 0.136, 0.713, 119.996, 292307.0, 4.0],
 [0.835, 0.687, 5.0, -3.18, 1.0, 0.184, 0.101, 0.0, 0.132, 0.828, 94.059, 272533.0, 4.0],
 [0.728, 0.801, 4.0, -3.636, 1.0, 0.0752, 0.00349, 0.000195, 0.0907, 0.813, 119.989, 193043.0, 4.0],
 [0.571, 0.89, 9.0, -1.6, 1.0, 0.0395, 0.00509, 0.0, 0.0769, 0.751, 110.958, 234147.0, 4.0],
 [0.536, 0.612, 4.0, -5.847, 1.0, 0.272, 0.119, 0.0, 0.209, 0.57, 86.768, 229040.0, 4.0],
 [0.659, 0.869, 11.0, -5.858, 1.0, 0.046, 0.00357, 0.0, 0.302, 0.811, 106.966, 201960.0, 4.0],
 [0.619, 0.87, 2.0, -4.956, 1.0, 0.501, 0.41, 1.37e-06, 0.0571, 0.94, 188.772, 219773.0, 4.0],
 [0.624, 0.976, 8.0, -5.355, 1.0, 0.0494, 0.004, 1.18e-05, 0.376, 0.514, 142.008, 199120.0, 4.0],
 [0.615, 0.711, 11.0, -5.507, 1.0, 0.0779, 0.0444, 0.0, 0.145, 0.711, 144.036, 221253.0, 4.0],
 [0.673, 0.683, 1.0, -5.693, 1.0, 0.115, 0.522, 0.0, 0.235, 0.713, 171.86, 232000.0, 4.0],
 [0.652, 0.698, 10.0, -4.667, 0.0, 0.042, 0.00112, 0.000115, 0.0886, 0.47, 96.021, 202067.0, 4.0],
 [0.423, 0.94, 1.0, -4.012, 0.0, 0.0635, 0.00166, 0.0, 0.178, 0.505, 149.934, 206520.0, 4.0],
 [0.706, 0.751, 9.0, -6.323, 1.0, 0.0708, 0.173, 0.0, 0.168, 0.195, 91.031, 182307.0, 4.0],
 [0.669, 0.822, 11.0, -4.288, 1.0, 0.043, 0.0339, 0.000142, 0.231, 0.43, 120.011, 277107.0, 4.0],
 [0.829, 0.627, 1.0, -3.928, 1.0, 0.0759, 0.00663, 0.0, 0.0939, 0.72, 120.048, 187133.0, 4.0],
 [0.709, 0.745, 4.0, -6.437, 0.0, 0.0738, 0.0225, 5.2e-05, 0.154, 0.576, 126.027, 234360.0, 4.0],
 [0.58, 0.75, 4.0, -4.421, 1.0, 0.194, 0.159, 0.0, 0.274, 0.728, 86.938, 229107.0, 4.0],
 [0.447, 0.848, 2.0, -6.175, 1.0, 0.222, 0.033, 7.45e-05, 0.722, 0.485, 172.247, 203760.0, 4.0],
 [0.356, 0.924, 1.0, -3.74, 1.0, 0.0808, 0.00101, 0.0, 0.0953, 0.232, 148.017, 222587.0, 4.0],
 [0.442, 0.893, 0.0, -4.878, 1.0, 0.0505, 0.00844, 0.0, 0.529, 0.712, 148.119, 168000.0, 4.0],
 [0.938, 0.735, 7.0, -6.382, 1.0, 0.0434, 0.00952, 0.0, 0.0998, 0.55, 103.7, 229360.0, 4.0],
 [0.719, 0.839, 5.0, -5.235, 1.0, 0.0257, 0.00241, 0.0, 0.228, 0.636, 129.97, 220920.0, 4.0],
 [0.465, 0.954, 10.0, -4.251, 1.0, 0.044, 0.000346, 5.38e-06, 0.573, 0.458, 143.85, 193653.0, 4.0],
 [0.73, 0.848, 5.0, -5.262, 0.0, 0.101, 0.0494, 0.0, 0.133, 0.559, 65.025, 213973.0, 4.0],
 [0.559, 0.845, 11.0, -3.871, 1.0, 0.0379, 0.222, 0.0, 0.164, 0.304, 94.915, 237493.0, 4.0],
 [0.725, 0.732, 9.0, -6.594, 1.0, 0.156, 0.0501, 1.8e-06, 0.104, 0.515, 93.029, 201230.0, 4.0],
 [0.695, 0.903, 9.0, -5.092, 0.0, 0.162, 0.0137, 4.32e-05, 0.126, 0.811, 138.985, 190453.0, 4.0],
 [0.747, 0.878, 8.0, -3.533, 1.0, 0.0282, 0.343, 0.0, 0.0872, 0.557, 106.004, 208333.0, 4.0],
 [0.485, 0.823, 1.0, -2.816, 1.0, 0.0362, 0.00981, 0.0, 0.116, 0.555, 91.005, 189187.0, 4.0],
 [0.414, 0.936, 2.0, -2.407, 1.0, 0.0758, 0.00136, 0.0, 0.369, 0.74, 170.229, 242413.0, 4.0],
 [0.35, 0.909, 2.0, -4.204, 1.0, 0.0774, 0.0024, 0.0, 0.163, 0.314, 166.866, 220133.0, 4.0]]

y = [0.423, 0.94, 1.0, -4.012, 0.0, 0.0635, 0.00166, 0.0, 0.178, 0.505, 149.934, 206520.0, 4.0]

model.fit(X, y, sample_weight=None)