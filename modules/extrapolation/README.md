# Motivation 
In FSI simulations, the fluid mesh cells must conform to the deformation of the solid domain. If the fluid domain cells
are poorly constructed, or fluid domain is acted upon by medium/large deformations, the FSI simulations might diverge
due to crossing fluid mesh cells.

One strategy to avoid entanglement is to solve an additional equation for "stiffening" the fluid mesh, often
refered to as a mesh lifting operator. The overall goal of a mesh lifting operator, is 
