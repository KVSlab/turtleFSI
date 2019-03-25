import numpy as np
import matplotlib.pyplot as plt

compare = "dt-0.1_theta-1.0/"
#min_j_const = np.loadtxt(open("./CSM_results/CSM-4/alfa_const/" + compare))
min_j_biharmonic = np.loadtxt(open("../CSM_results/CSM-4/biharmonic_bc1/"+ compare + "min_J.txt"))
time =             np.loadtxt(open("../CSM_results/CSM-4/biharmonic_bc1/" + compare + "time.txt"))
#pressure = np.loadtxt(open("pressure.txt"))


#plt.plot(time, min_j_const,"--", label =("Harmonic"))
#plt.plot(time, min_j_biharmonic,"-", label =("Biharmonic"))
#plt.plot(time, min_j_const,":", label =("Harmonic"))
plt.plot(time, min_j_biharmonic,"-.", label =("Biharmonic_bc1"))

#plt.plot(time,pressure, label =("pressure"))
#plt.axis([0, 10, -1.2, 5])
axes = plt.gca()
legend = axes.legend(loc='upper right', shadow=True)
plt.title("Min_j")
plt.show()
