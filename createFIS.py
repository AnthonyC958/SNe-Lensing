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
#  ################################################ create FIS ################################################  #
with open(f"random_cones_new.pickle", "rb") as pickle_in:
    lenses = pickle.load(pickle_in)
#
# for key, item in lenses.items():
#     print(key)
#     for num, w in enumerate(item["WEIGHT"]):
#         if w != 1.0:
#             lenses[key].pop(f"Shell{num+1}")
#
# pickle_out = open(f"random_cones_new_fis.pickle", "wb")
# pickle.dump(lenses, pickle_out)
# pickle_out.close()


#  ############################################ create FIS SNe Data ############################################  #
with open("MICE_SN_data.pickle", "rb") as pickle_in:
    SN_data = pickle.load(pickle_in)

SN_data_fis = {}
for key, item in lenses.items():
    SN_data_fis[key] = {"mu_diff": SN_data["mu_diff"][item["WEIGHT"] == 1.0],
                        "SNZ": SN_data["SNZ"][item["WEIGHT"] == 1.0],
                        "SNkappa": SN_data["SNkappa"][item["WEIGHT"] == 1.0],
                        "SNRA": SN_data["SNRA"][item["WEIGHT"] == 1.0],
                        "SNDEC": SN_data["SNDEC"][item["WEIGHT"] == 1.0],
                        "SNMU": SN_data["SNMU"][item["WEIGHT"] == 1.0],
                        "SNMU_ERR": SN_data["SNMU_ERR"][item["WEIGHT"] == 1.0]}

print(SN_data_fis["Radius20.0"]['mu_diff'])
pickle_out = open(f"MICE_SN_data_fis.pickle", "wb")
pickle.dump(SN_data_fis, pickle_out)
pickle_out.close()

