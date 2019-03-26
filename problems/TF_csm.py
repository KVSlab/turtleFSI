# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.


from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys

refi = 0
mesh_name = "base2"
mesh_file = Mesh("problems/mesh/" + mesh_name + ".xml")
# for i in range(refi):
#    mesh_file = refine(mesh_file)

# Parameters
common = {"mesh": mesh_file,
          "v_deg": 2,  # Velocity degree
          "p_deg": 1,  # Pressure degree
          "d_deg": 2,  # Deformation degree
          "T": 30,  # End time [s]
          "dt": 0.01,  # Time step [s]
          "rho_f": 1.0E3,  # Fluid density [kg/m3]
          "mu_f": 1.0,  # Fluid dynamic viscosity [Pa.s]
          "rho_s": Constant(1.0E3),  # Solid density[kg/m3]
          "mu_s": Constant(0.5E6),  # Solid shear modulus or 2nd Lame Coef. (CSM1:0.5E6, CSM2:2.0E6, CSM3:0.5E6) [Pa]
          "nu_s": Constant(0.4),  # Solid Poisson ratio [-]
          "Um": 1.0,  # Max. velocity inlet [m/s]
          "D": 0.1,  # Turek flag specific
          "H": 0.41,  # Turek flag specific
          "L": 2.5,  # Turek flag specific
          "step": 1,  # save every step
          "checkpoint": 1}  # checkpoint every step

g = 2  # gravity [m/s2]
vars().update(common)
lamda_s = nu_s*2*mu_s/(1 - 2.*nu_s)  # Solid Young's modulus [Pa]

for coord in mesh.coordinates():
    if coord[0] == 0.6 and (0.199 <= coord[1] <= 0.2001):  # to get the point [0.2,0.6] end of bar
        print(coord)
        break
# BOUNDARIES

#NOS = AutoSubDomain(lambda x: "on_boundary" and( near(x[1],0) or near(x[1], 0.41)))
Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0], 2.5)))
Wall = AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.41) or near(x[1], 0)))
Bar = AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.21)) or near(x[1], 0.19) or near(x[0], 0.6))
Circle = AutoSubDomain(lambda x: "on_boundary" and (((x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2) < 0.0505*0.0505)))
Barwall = AutoSubDomain(lambda x: "on_boundary" and (((x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2) < 0.0505*0.0505) and x[1] >= 0.19 and x[1] <= 0.21 and x[0] > 0.2))

Allboundaries = DomainBoundary()

boundaries = MeshFunction("size_t", mesh_file, 1)
boundaries.set_all(0)
Allboundaries.mark(boundaries, 1)
Wall.mark(boundaries, 2)
Inlet.mark(boundaries, 3)
Outlet.mark(boundaries, 4)
Bar.mark(boundaries, 5)
Circle.mark(boundaries, 6)
Barwall.mark(boundaries, 7)
# plot(boundaries,interactive=True)

ds = Measure("ds", subdomain_data=boundaries)
dS = Measure("dS", subdomain_data=boundaries)
n = FacetNormal(mesh_file)

Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24 <= x[0] <= 0.6)  # only the "flag" or "bar"
domains = MeshFunction("size_t", mesh_file, 2)
domains.set_all(1)
Bar_area.mark(domains, 2)  # Overwrites structure domain
dx = Measure("dx", subdomain_data=domains)
#plot(domains,interactive = True)
dx_f = dx(1, subdomain_data=domains)
dx_s = dx(2, subdomain_data=domains)
dis_x = []
dis_y = []
Drag_list = []
Lift_list = []
Time_list = []
Det_list = []

#dvp_file = XDMFFile(mpi_comm_world(), "FSI_fresh_checkpoints/CSM-1/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.xdmf")


if checkpoint == "results/TF_csm/checkpoints/CSM-1/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.h5":
    sys.exit(0)
else:
    dvp_file = HDF5File(mpi_comm_world(), "results/TF_csm/checkpoints/CSM-1/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.h5", "w")


def initiate(t, T, F_solid_linear, args, theta, mesh_file, rho_s, psi, extype,
             dx_s, d_deg, p_deg, v_deg, dt, P, dvp_, Time_list, Det_list, **semimp_namespace):

    #gravity = Constant((0, -2*rho_s))
    #F_solid_linear -= inner(gravity, psi)*dx_s
    def F_(U):
        return Identity(len(U)) + grad(U)

    def J_(U):
        return det(F_(U))

    if args.extravar == "alfa":
        path = "results/TF_csm/"+str(args.extravar) + "_" + str(args.extype) + "/dt-"+str(dt)+"_theta-"+str(theta)
    else:
        path = "results/TF_csm/"+str(args.extravar) + "_" + str(args.bitype) + "/dt-"+str(dt)+"_theta-"+str(theta)

    u_file = XDMFFile(mpi_comm_world(), path + "/velocity.xdmf")
    d_file = XDMFFile(mpi_comm_world(), path + "/d.xdmf")
    for tmp_t in [u_file, d_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["rewrite_function_mesh"] = False
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    d_file.write(d)
    u_file.write(v)

    #dg = FunctionSpace(mesh_file, "DG", 0)
    det_func = Function(P)
    Det = project(J_(d), P)
    det_func.vector().zero()
    det_func.vector().axpy(1, Det.vector())

    Time_list.append(t)
    dsx = d(coord)[0]
    dsy = d(coord)[1]
    dis_x.append(dsx)
    dis_y.append(dsy)

    Det_list.append((det_func.vector().get_local()).min())

    return dict(u_file=u_file, d_file=d_file, det_func=det_func, path=path)


def create_bcs(DVP, args, boundaries,  **semimp_namespace):

    # Fluid velocity conditions
    u_inlet = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 3)
    u_wall = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 2)
    u_circ = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 6)  # No slip on geometry in fluid
    u_barwall = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 7)  # No slip on geometry in fluid

    p_outlet = DirichletBC(DVP.sub(2), (0.0), boundaries, 4)

    # Assemble boundary conditions
    bcs = [u_wall, u_inlet, u_circ, u_barwall,
           p_outlet]

    # if DVP.num_sub_spaces() == 4:
    if args.bitype == "bc1":
        d_wall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 2)
        d_inlet = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 3)
        d_outlet = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 4)
        d_circle = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 6)
        d_barwall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 7)  # No slip on geometry in fluid
        for i in [d_wall, d_inlet, d_outlet, d_circle, d_barwall]:
            bcs.append(i)

    if args.bitype == "bc2":
        w_wall = DirichletBC(DVP.sub(0).sub(1), (0.0), boundaries, 2)
        w_inlet = DirichletBC(DVP.sub(0).sub(0), (0.0), boundaries, 3)
        w_outlet = DirichletBC(DVP.sub(0).sub(0), (0.0), boundaries, 4)
        w_circle = DirichletBC(DVP.sub(0).sub(1), (0.0), boundaries, 6)
        w_barwall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 7)  # No slip on geometry in fluid

        d_wall = DirichletBC(DVP.sub(0).sub(1), (0.0), boundaries, 2)
        d_inlet = DirichletBC(DVP.sub(0).sub(0), (0.0), boundaries, 3)
        d_outlet = DirichletBC(DVP.sub(0).sub(0), (0.0), boundaries, 4)
        d_circle = DirichletBC(DVP.sub(0).sub(1), (0.0), boundaries, 6)
        d_barwall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 7)

        for i in [w_wall, w_inlet, w_outlet, w_circle, w_barwall,
                  d_wall, d_inlet, d_outlet, d_circle, d_barwall]:
            bcs.append(i)

    return dict(bcs=bcs)


def pre_solve(**semimp_namespace):
    return {}


def after_solve(t, path, det_func, P, DVP, dvp_, n, coord, dis_x, dis_y, Det_list,
                counter, dvp_file, u_file, d_file, **semimp_namespace):

    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)

    if counter % step == 0:

        d = dvp_["n"].sub(0, deepcopy=True)
        v = dvp_["n"].sub(1, deepcopy=True)
        d_file.write(d, t)
        u_file.write(v, t)

    def F_(U):
        return Identity(len(U)) + grad(U)

    def J_(U):
        return det(F_(U))

    def sigma_f_new(v, p, d, mu_f):
        return -p*Identity(len(v)) + mu_f*(grad(v)*inv(F_(d)) + inv(F_(d)).T*grad(v).T)

    #Det = project(J_(d), DVP.sub(0).collapse())
    Det = project(J_(d), P)
    det_func.vector().zero()
    det_func.vector().axpy(1, Det.vector())
    Det_list.append((det_func.vector().get_local()).min())

    Time_list.append(t)
    dsx = d(coord)[0]
    dsy = d(coord)[1]
    dis_x.append(dsx)
    dis_y.append(dsy)
    if MPI.rank(mpi_comm_world()) == 0:
        print("dis_x/dis_y : %g %g " % (dsx, dsy))

    return {}


def post_process(path, T, dt, Det_list, dis_x, dis_y, Time_list,
                 args, v_deg, p_deg, d_deg, **semimp_namespace):

    theta = args.theta
    f_scheme = args.fluidvar
    s_scheme = args.solidvar
    e_scheme = args.extravar
    if MPI.rank(mpi_comm_world()) == 0:

        f = open(path+"/report.txt", 'w')
        f.write("""CSM-1 EXPERIMENT
        T = %(T)g\ndt = %(dt)g\nv_deg = %(d_deg)g\nv_deg = %(v_deg)g\np_deg = %(p_deg)g\n
    theta = %(theta)s\nf_vari = %(f_scheme)s\ns_vari = %(s_scheme)s\ne_vari = %(e_scheme)s\n""" % vars())
        #f.write("""Runtime = %f """ % fintime)
        f.close()
        np.savetxt(path + '/Min_J.txt', Det_list, delimiter=',')
        np.savetxt(path + '/Time.txt', Time_list, delimiter=',')
        np.savetxt(path + '/dis_x.txt', dis_x, delimiter=',')
        np.savetxt(path + '/dis_y.txt', dis_y, delimiter=',')

    return {}