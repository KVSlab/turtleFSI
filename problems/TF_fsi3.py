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
mesh_name = "base0"
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
          "rho_s": Constant(1E3),  # Solid density (FSI1:1E3, FSI2:10E3, FSI3:1E3) [kg/m3]
          "mu_s": Constant(2.0E6),  # Solid shear modulus or 2nd Lame Coef. (FSI1:0.5E6, FSI2:0.5E6, FSI3:2.0E6) [Pa]
          "nu_s": Constant(0.4),  # Solid Poisson ratio [-]
          "Um": 2.0,  # Max. velocity inlet (FSI1:0.2, FSI2:1.0, FSI3:2.0) [m/s]
          "D": 0.1,  # Turek flag specific
          "H": 0.41,  # Turek flag specific
          "L": 2.5,  # Turek flag specific
          "step": 1,  # save every step
          "checkpoint": 1}  # checkpoint every step

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

# Fluid properties


class Inlet(Expression):
    def __init__(self, Um, **kwargs):
        self.t = 0
        self.Um = Um

    def eval(self, value, x):
        value[0] = 0.5*(1-np.cos(self.t*np.pi/2))*1.5*self.Um*x[1]*(H-x[1])/((H/2.0)**2)
        value[1] = 0

    def value_shape(self):
        return (2,)


inlet = Inlet(Um, degree=v_deg)

if checkpoint == "results/TF_fsi/checkpoints/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.h5":
    sys.exit(0)
else:
    dvp_file = HDF5File(mpi_comm_world(), "results/TF_fsi/checkpoints/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.h5", "w")


def initiate(P, v_deg, d_deg, p_deg, dt, theta, dvp_, args, Det_list, refi, mesh_file, mesh_name, **semimp_namespace):

    exva = args.extravar
    extype = args.extype
    bitype = args.bitype
    solver = args.solver
    if args.extravar == "alfa":
	print(args)
        path = "results/TF_fsi/%(exva)s_%(extype)s_%(solver)s/dt-%(dt)g_theta-%(theta)g/%(mesh_name)s_refine_%(refi)d_v_deg_%(v_deg)s_d_deg_%(d_deg)s_p_deg_%(p_deg)s" % vars()
    # if args.extravar == "biharmonic" or args.extravar == "laplace" or args.extravar == "elastic":
    else:
        path = "results/TF_fsi/%(exva)s_%(bitype)s/dt-%(dt)g_theta-%(theta)g/%(mesh_name)s_refine_%(refi)d_v_deg_%(v_deg)s_d_deg_%(d_deg)s_p_deg_%(p_deg)s" % vars()

    u_file = XDMFFile(mpi_comm_world(), path + "/velocity.xdmf")
    d_file = XDMFFile(mpi_comm_world(), path + "/d.xdmf")
    p_file = XDMFFile(mpi_comm_world(), path + "/pressure.xdmf")
    for tmp_t in [u_file, d_file, p_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["rewrite_function_mesh"] = False
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)
    d_file.write(d)
    u_file.write(v)
    p_file.write(p)

    return dict(u_file=u_file, d_file=d_file, p_file=p_file, path=path)


def create_bcs(DVP, args, dvp_, n, k, Um, H, boundaries, inlet, **semimp_namespace):
    print("Create bcs")
    # Fluid velocity conditions
    u_inlet = DirichletBC(DVP.sub(1), inlet, boundaries, 3)
    u_wall = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 2)
    u_circ = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 6)  # No slip on geometry in fluid
    u_barwall = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 7)  # No slip on geometry in fluid

    # Pressure Conditions
    p_out = DirichletBC(DVP.sub(2), 0, boundaries, 4)

    # Assemble boundary conditions
    bcs = [u_wall, u_inlet, u_circ, u_barwall, p_out]

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

    return dict(bcs=bcs, inlet=inlet)


def pre_solve(t, inlet, **semimp_namespace):
    if t < 2:
        inlet.t = t
    else:
        inlet.t = 2

    return dict(inlet=inlet)


def after_solve(t, P, DVP, dvp_, n, coord, dis_x, dis_y, Drag_list, Lift_list,
                Det_list, counter, dvp_file, u_file, p_file, d_file, **semimp_namespace):

    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)

    if counter % step == 0:

        d = dvp_["n"].sub(0, deepcopy=True)
        v = dvp_["n"].sub(1, deepcopy=True)
        p = dvp_["n"].sub(2, deepcopy=True)
        p_file.write(p, t)
        d_file.write(d, t)
        u_file.write(v, t)

    def F_(U):
        return (Identity(len(U)) + grad(U))

    def J_(U):
        return det(F_(U))

    def sigma_f_new(v, p, d, mu_f):
        return -p*Identity(len(v)) + mu_f*(grad(v)*inv(F_(d)) + inv(F_(d)).T*grad(v).T)

    Det = project(J_(d), P)
    Det_list.append((Det.vector().get_local()).min())

    Dr = -assemble((sigma_f_new(v, p, d, mu_f)*n)[0]*ds(6))
    Li = -assemble((sigma_f_new(v, p, d, mu_f)*n)[1]*ds(6))
    Dr += -assemble((sigma_f_new(v("+"), p("+"), d("+"), mu_f)*n("+"))[0]*dS(5))
    Li += -assemble((sigma_f_new(v("+"), p("+"), d("+"), mu_f)*n("+"))[1]*dS(5))
    Drag_list.append(Dr)
    Lift_list.append(Li)
    Time_list.append(t)

    det_func = Function(P)
    Det = project(J_(d), P)
    det_func.vector().zero()
    det_func.vector().axpy(1, Det.vector())
    Det_list.append((det_func.vector().get_local()).min())

    dsx = d(coord)[0]
    dsy = d(coord)[1]
    dis_x.append(dsx)
    dis_y.append(dsy)
    if MPI.rank(mpi_comm_world()) == 0:
        print("LIFT = %g,  DRAG = %g" % (Li, Dr))
        print("dis_x/dis_y : %g %g " % (dsx, dsy))

    return {}


def post_process(path, T, dt, Det_list, dis_x, dis_y, Drag_list, Lift_list, Time_list,
                 args, simtime, v_deg, p_deg, d_deg, dvp_file, **semimp_namespace):

    theta = args.theta
    f_scheme = args.fluidvar
    s_scheme = args.solidvar
    e_scheme = args.extravar

    if MPI.rank(mpi_comm_world()) == 0:
        print("IN POSTPRO", path)
        f = open(path+"/report.txt", 'w')
        f.write("""FSI2 EXPERIMENT
        T = %(T)g\ndt = %(dt)g\nv_deg = %(d_deg)g\nv_deg = %(v_deg)g\np_deg = %(p_deg)g\n
    theta = %(theta)s\nf_vari = %(f_scheme)s\ns_vari = %(s_scheme)s\ne_vari = %(e_scheme)s\n time = %(simtime)g""" % vars())
        #f.write("""Runtime = %f """ % fintime)
        f.close()
        np.savetxt(path + '/Min_J.txt', Det_list, delimiter=',')
        np.savetxt(path + '/Lift.txt', Lift_list, delimiter=',')
        np.savetxt(path + '/Drag.txt', Drag_list, delimiter=',')
        np.savetxt(path + '/Time.txt', Time_list, delimiter=',')
        np.savetxt(path + '/dis_x.txt', dis_x, delimiter=',')
        np.savetxt(path + '/dis_y.txt', dis_y, delimiter=',')

        plt.figure(1)
        plt.plot(Time_list, dis_x)
        plt.ylabel("Displacement x")
        plt.xlabel("Time")
        plt.grid()
        plt.savefig(path + "/dis_x.png")
        plt.figure(2)
        plt.plot(Time_list, dis_y)
        plt.ylabel("Displacement y")
        plt.xlabel("Time")
        plt.grid()
        plt.savefig(path + "/dis_y.png")
        plt.figure(3)
        plt.plot(Time_list, Drag_list)
        plt.ylabel("Drag")
        plt.xlabel("Time")
        plt.grid()
        plt.savefig(path + "/drag.png")
        plt.figure(4)
        plt.plot(Time_list, Lift_list)
        plt.ylabel("Lift")
        plt.xlabel("Time")
        plt.grid()
        plt.savefig(path + "/lift.png")
        # plt.figure(5)
        # plt.plot(Time_list,Det_list);plt.ylabel("Min_Det(F)");plt.xlabel("Time");plt.grid();
        #plt.savefig(path + "/Min_J.png")

    return {}
