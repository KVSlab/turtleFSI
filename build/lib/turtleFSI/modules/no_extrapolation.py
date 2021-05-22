# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.


def extrapolate_setup(**namespace):
    """
    Do not move mesh.
    If only solving for solid or fluid, use this to also remove the mesh lifting
    operator.
    """

    return {}
