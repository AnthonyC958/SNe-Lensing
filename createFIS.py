import MICE
import pickle
import numpy as np
import matplotlib.pyplot as plt

# with open("random_cones1_new.pickle", "rb") as pickle_in:
#     lenses = pickle.load(pickle_in)
#     print(lenses.keys())
#
# plt.plot(0, 0)
# plt.show()

with open(f"random_cones_new.pickle", "rb") as pickle_in:
    lenses = pickle.load(pickle_in)
with open("MICE_SN_data.pickle", "rb") as pickle_in:
    SN_data = pickle.load(pickle_in)
SN_data_fis = {}

for key, item in lenses.items():
    print(key)
    FIS_indices = np.where(item["WEIGHT"] == 1.0)
    for num, w in enumerate(item["WEIGHT"]):
        if w != 1.0:
            lenses[key].pop(f"Shell{num+1}")
    SN_data_fis[key] = {"mu_diff": SN_data["mu_diff"][FIS_indices],
                        "SNZ": SN_data["SNZ"][FIS_indices],
                        "SNkappa": SN_data["SNkappa"][FIS_indices],
                        "SNRA": SN_data["SNRA"][FIS_indices],
                        "SNDEC": SN_data["SNDEC"][FIS_indices],
                        "SNMU": SN_data["SNMU"][FIS_indices],
                        "SNMU_ERR": SN_data["SNMU_ERR"][FIS_indices]}

pickle_out = open(f"random_cones_new_fis.pickle", "wb")
pickle.dump(lenses, pickle_out)
pickle_out.close()

print(SN_data_fis["Radius20.0"]['mu_diff'])
pickle_out = open(f"MICE_SN_data_fis.pickle", "wb")
pickle.dump(SN_data_fis, pickle_out)
pickle_out.close()

