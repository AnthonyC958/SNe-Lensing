import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import Convergence
from labellines import labelLines

matplotlib.use('TkAgg')
H0 = 70.0
c = 2.998E5
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
colours = [[0, 150/255, 100/255], [225/255, 149/255, 0], [207/255, 0, 48/255], 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5']
grey = [0.75, 0.75, 0.75]


def vz(z):
    q0 = -0.55
    j0 = -1
    return c * z * (1 + 0.5 * (1 - q0) * z - (1 - q0 - 3 * q0 ** 2 + j0) * (z ** 2) / 6)


v = c*0.1
zs = np.linspace(0, 1.5, 1001)
for num, theta in enumerate(np.array(RADII)[[0, 15, 27, 41, 57, 69, 76, 83]]):
    theta_rad = theta/60 * np.pi/180.
    Dperp = theta_rad * Convergence.comoving(zs, OM=0.25, OL=0.75, h=0.7)*1000.0
    plt.plot(zs, Dperp, color=colours[0], alpha=(1 - num/9.0), label=f"{theta}'", linewidth=1.5)
labelLines(plt.gca().get_lines(), xvals=[1.375, 1.35, 1.32, 1.28, 1.23, 1.185, 1.11, 1.0],
           zorder=2.5, fontsize=12, align=False)
plt.plot([0.6, 0.6], [-5, 45], linestyle='--', color=grey)
plt.text(0.4, 30, 'SDSS', color=grey, fontsize=16)
plt.ylim([-2, 42])
plt.xlabel('$z$')
plt.ylabel('Perpendicular Distance (Mpc)')
plt.show()

n = 41
mag = np.ones(n)
magdeg = np.zeros(n)
magdegplus = np.zeros(n)
for i in np.arange(n):
    magdeg[i] = mag[i] * random.gauss(1.0, 0.1)
    magdegplus[i] = mag[i] + random.gauss(0.0, 0.1)

plt.plot(mag, '.')
# plt.plot(magdeg,'x')
# plt.plot(magdegplus,'+')

plt.errorbar(np.arange(n), magdegplus, np.ones(n)*0.1, fmt='.')

# plt.errorbar(np.arange(n),magdeg,np.ones(11)*0.1,'.')
plt.show()


