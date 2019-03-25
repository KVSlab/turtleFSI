import numpy as np
import matplotlib.pyplot as plt

FSI = 1; dt = 0.5 ; theta = 0.5 ; extra = "alfa_const"
Drag = np.loadtxt(open("./FSI_fresh_results/FSI-"+str(FSI)+"/"+str(extra)+"/dt-"+str(dt)+"_theta-"+str(theta)+"/Drag.txt"))
Lift = np.loadtxt(open("./FSI_fresh_results/FSI-"+str(FSI)+"/"+str(extra)+"/dt-"+str(dt)+"_theta-"+str(theta)+"/Lift.txt"))

time = np.loadtxt(open("./FSI_fresh_results/FSI-"+str(FSI)+"/"+str(extra)+"/dt-"+str(dt)+"_theta-"+str(theta)+"/Time.txt"))
#pressure = np.loadtxt(open("pressure.txt"))

plt.figure(1)
axes = plt.gca()
legend = axes.legend(loc='upper right', shadow=True)

plt.plot(time, Drag,"--", label =("FSI-"+str(FSI)))
plt.title("Drag")
plt.figure(2)
plt.plot(time, Drag,"--", label =("FSI-"+str(FSI)))
plt.title("Lift")
#plt.axis([0, 10, -1.2, 5])
axes = plt.gca()
legend = axes.legend(loc='upper right', shadow=True)
plt.show()
