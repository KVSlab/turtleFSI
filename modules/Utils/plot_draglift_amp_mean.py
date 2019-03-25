import numpy as np
import matplotlib.pyplot as plt

FSI = 2; dt = 0.01 ; theta = 0.51 ; extra = "biharmonic"

#Drag1 = np.loadtxt(open("./FSI_fresh_results/FSI-"+str(FSI)+"/"+str(extra)+"/dt-"+str(dt)+"_theta-"+str(theta)+"/Drag.txt"))
#Lift1 = np.loadtxt(open("./FSI_fresh_results/FSI-"+str(FSI)+"/"+str(extra)+"/dt-"+str(dt)+"_theta-"+str(theta)+"/Lift.txt"))
#disx = np.loadtxt(open("./FSI_fresh_results/FSI-"+str(FSI)+"/"+str(extra)+"/dt-"+str(dt)+"_theta-"+str(theta)+"/dis_x.txt"))
#disy = np.loadtxt(open("./FSI_fresh_results/FSI-"+str(FSI)+"/"+str(extra)+"/dt-"+str(dt)+"_theta-"+str(theta)+"/dis_y.txt"))

#time1 = np.loadtxt(open("./FSI_fresh_results/FSI-"+str(FSI)+"/"+str(extra)+"/dt-"+str(dt)+"_theta-"+str(theta)+"/Time.txt"))

Drag = np.loadtxt(open("./abacus_results/FSI-3/Drag.txt"))
Lift = np.loadtxt(open("./abacus_results/FSI-3/Lift.txt"))
disy = np.loadtxt(open("./abacus_results/FSI-3/dis_y.txt"))
disx = np.loadtxt(open("./abacus_results/FSI-3/dis_x.txt"))
time2 = np.loadtxt(open("./abacus_results/FSI-3/time.txt"))
#/FSI_fresh_results/FSI-2/laplace_bc1/dt-0.01_theta-0.51/refine=0_v_deg=2_d_deg=2_p_deg=1
#abacus_results/refine=0_v_deg=2_d_deg=2_p_deg=1
#pressure = np.loadtxt(open("pressure.txt"))
#print len(Drag)
mean = 0.5*(np.max(disy[800:900]) + np.min(disy[800:900]))
amplitude = 0.5*(np.max(disy[800:900]) - np.min(disy[800:900]) )
print("dis_y: ",mean, "+", amplitude)
mean = 0.5*(np.max(disx[800:900]) + np.min(disx[800:900]))
amplitude = 0.5*(np.max(disx[800:900]) - np.min(disx[800:900]) )
print("dis_x: ",mean, "+", amplitude)
mean = 0.5*(np.max(Lift[800:900]) + np.min(Lift[800:900]))
amplitude = 0.5*(np.max(Lift[800:900]) - np.min(Lift[800:900]) )
print("Lift: ",mean, "+", amplitude)
mean = 0.5*(np.max(Drag[800:900]) + np.min(Drag[800:900]))
amplitude = 0.5*(np.max(Drag[800:900]) - np.min(Drag[800:900]) )
print("Drag: ",mean, "+", amplitude)

plt.figure(1)
#plt.plot(time1, Drag1,"r", label =("biharmonic"))
plt.plot(time2, Drag,"b", label =("harmonic"))
plt.axis([8, 8.5, 430, 485])

plt.title("Drag")
plt.figure(2)
#plt.plot(time1, Lift1,"r", label =("biharmonic"))
plt.plot(time2, Lift,"b", label =("harmonic"))
plt.axis([8, 8.5, -150, 200])

plt.title("Lift")
#plt.figure(2)
#plt.plot(time, Lift,"b", label =("FSI-"+str(FSI)))
#plt.title("Lift")
plt.figure(3)
plt.plot(disx ,"g", label =("FSI-"+str(FSI)))
plt.title("Displacement x")
plt.figure(4)
plt.plot(disy,"y", label =("FSI-"+str(FSI)))
plt.title("Displacement y")
axes = plt.gca()
legend = axes.legend(loc='upper right', shadow=True)
plt.show()
