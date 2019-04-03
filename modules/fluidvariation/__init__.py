def setup(**namespace):
    """Set up all equations to be solved."""
    return {}


def create_bcs(**namespace):
    """Set up boundary conditions"""
    return {}

# Execute before solving in timeloop
def pre_solve(**namespace):
    """Execute before each solve in timeloop"""
    return {}

def sourceterm(**namespace):
    """Get defined sourceterm, mainly for MMS"""
    return {}
