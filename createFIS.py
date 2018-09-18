import MICE
import pickle
import numpy as np
import Convergence
import matplotlib.pyplot as plt

RADII = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
         0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0,
         2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
         4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5,
         5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25,
         7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0,
         9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75,
         11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5,
         12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.5,
         15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0,
         18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0,
         24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]

# with open("random_cones1_new.pickle", "rb") as pickle_in:
#     lenses = pickle.load(pickle_in)
#     print(lenses.keys())
#
# plt.plot(0, 0)
# plt.show()

# with open(f"random_cones_new.pickle", "rb") as pickle_in:
#     lenses = pickle.load(pickle_in)
# with open("MICE_SN_data.pickle", "rb") as pickle_in:
#     SN_data = pickle.load(pickle_in)
# SN_data_fis = {}
#
# for key, item in lenses.items():
#     print(key)
#     FIS_indices = np.where(item["WEIGHT"] == 1.0)
#     print(len(FIS_indices[0]))
#     for num, w in enumerate(item["WEIGHT"]):
#         if w != 1.0:
#             lenses[key].pop(f"Shell{num+1}")
#     SN_data_fis[key] = {"mu_diff": SN_data["mu_diff"][FIS_indices],
#                         "SNZ": SN_data["SNZ"][FIS_indices],
#                         "SNkappa": SN_data["SNkappa"][FIS_indices],
#                         "SNRA": SN_data["SNRA"][FIS_indices],
#                         "SNDEC": SN_data["SNDEC"][FIS_indices],
#                         "SNMU": SN_data["SNMU"][FIS_indices],
#                         "SNMU_ERR": SN_data["SNMU_ERR"][FIS_indices]}
#
# pickle_out = open(f"random_cones_new_fis.pickle", "wb")
# pickle.dump(lenses, pickle_out)
# pickle_out.close()
#
# print(SN_data_fis["Radius20.0"]['mu_diff'])
# pickle_out = open(f"MICE_SN_data_fis.pickle", "wb")
# pickle.dump(SN_data_fis, pickle_out)
# pickle_out.close()

with open(f"lenses.pickle", "rb") as pickle_in:
    lenses = pickle.load(pickle_in)
# print(lenses['Radius12.0']["SN356"]["IPWEIGHT"])

for radius in RADII:
    print(radius)
    for key, item in lenses[f'Radius{radius}'].items():
        lenses[f'Radius{radius}'][key]['IPWEIGHT'] = []
        for z, ra, dec in zip(item['Zs'], item["RAs"], item["DECs"]):
            # print(item["SNRA"], item["SNDEC"])
            # print(item["RAs"], item["DECs"])
            theta = (((ra - item["SNRA"])**2 + (dec - item["SNDEC"])**2)**0.5*np.pi/180)
            Dpara = Convergence.b_comoving(0, z, OM=0.27, OL=0.73, h=0.738, n=201)[-1] * 1000.0
            limperp = 0.2*np.pi/180 * Dpara
            Dperp = theta * Dpara
            # print(theta*180/np.pi*60, (1/Dperp + 0.1 - 1/limperp))
            lenses[f'Radius{radius}'][key]['IPWEIGHT'].append(1/Dperp + 0.1 - 1/limperp)

pickle_out = open(f"lenses.pickle", "wb")
pickle.dump(lenses, pickle_out)
pickle_out.close()
