import numpy as np
from tabulate import tabulate

def convergence(E_u, E_p, N, dt):
    print(N, dt)
    if len(N) > len(dt):
        check = N; opp = dt
    else:
        check = dt; opp = N

    if check == N:
        print()
        print("#################################### - ERROR/CON SPACE - ####################################\n")

    else:
        print()
        print("#################################### - ERROR/CON TIME - ####################################\n")

    time = [i for i in range(len(E_u))]
    for E in [E_u, E_p]:
        print()
        print("#################################### - L2 NORM - ####################################\n")
        table = []
        headers = ["N" if opp is N else "dt"]
        li = []
        li.append(str(opp[0]))
        for i in range(len(E)):
            li.append("%e" % E[i])

        table.append(li)
        print(check)
        for i in range(len(check)):
            headers.append("dt = %g" % check[i] if check is dt else "N = %g" % check[i])
        print(tabulate.tabulate(table, headers, tablefmt="fancy_grid"))

        print()


        print()
        print("############################### - CONVERGENCE RATE - ###############################\n")

        table = []
        headers = ["N" if opp is N else "dt"]
        #for i in range(len(N)):
        li = []
        li.append(str(opp[0]))
        for i in range(len(E) - 1):
            error = E[(i+1)] / E[i]
            h_ = check[(i+1)] / check[i]
            conv = np.log(error)/np.log(h_) #h is determined in main solve method
            li.append(conv)
        table.append(li)
        for i in range(len(check)-1):
            headers.append("%g to %g" % (check[i], check[i+1]))
        print(tabulate.tabulate(table, headers, tablefmt="fancy_grid"))
        #time = []; E = []; h = []
