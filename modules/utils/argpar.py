import argparse
from argparse import RawTextHelpFormatter


def parse():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('Parameters')
    group.add_argument("-problem", help="Set problem to solve", default="TF_fsi")
    group.add_argument("-fluidvar", help="Set variationalform for fluid", default="thetaCN")
    group.add_argument("-solidvar", help="Set variationalform for solid", default="thetaCN")
    group.add_argument("-extravar", help="Set variationalform for extrapolation", default="laplace")
    group.add_argument("-tag", help="tag name for storing file", default="tag")
    group.add_argument("-solver", help="Set type of solver to be used", default="naive")
    group.add_argument("-p_deg", type=int, help="Set degree of pressure", default=1)
    group.add_argument("-v_deg", type=int, help="Set degree of velocity", default=2)
    group.add_argument("-d_deg", type=int, help="Set degree of deformation", default=2)
    group.add_argument("-theta", type=float,  help="Theta Parameter for ex/imp/CN", default=0.5)
    group.add_argument("-extype", help="extrapolation constant (const, smallc, det)", default="const")
    group.add_argument("-bitype", help="Different BC for extrapol (bc1, bc2)", default="bc1")
    group.add_argument("-T", type=float, help="Set end time", default=None)
    group.add_argument("-dt", type=float, help="Set timestep", default=None)
    group.add_argument("-r", "--refiner",   action="count", help="Mesh-refiner using built-in FEniCS method refine(Mesh)")

    return parser.parse_args()
