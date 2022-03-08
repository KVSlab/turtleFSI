__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
"""
This module contains functionality for efficiently probing a Function many times.
"""
import cppimport
from mpi4py.MPI import COMM_WORLD as comm
from numpy import zeros, squeeze, save

probe11 = cppimport.imp('turtleFSI.utils.probe.probe11')


# Give the compiled classes some additional pythonic functionality
class Probes(probe11.Probes):

    def __call__(self, *args):
        return self.eval(*args)

    def __len__(self):
        return self.local_size()

    def __iter__(self):
        self.i = 0
        return self

    def __getitem__(self, i):
        return self.get_probe_id(i), self.get_probe(i)

    def __next__(self):
        try:
            p = self[self.i]
        except:
            raise StopIteration
        self.i += 1
        return p

    next = __next__

    def array(self, N=None, filename=None, component=None, root=0):
        """Dump data to numpy format on root processor for all or one snapshot."""
        is_root = comm.Get_rank() == root
        size = self.get_total_number_probes() if is_root else len(self)
        comp = self.value_size() if component is None else 1
        if not N is None:
            z = zeros((size, comp))
        else:
            z = zeros((size, comp, self.number_of_evaluations()))

        # Get all values
        if len(self) > 0:
            if not N is None:
                for k in range(comp):
                    if is_root:
                        ids = self.get_probe_ids()
                        z[ids, k] = self.get_probes_component_and_snapshot(k, N)
                    else:
                        z[:, k] = self.get_probes_component_and_snapshot(k, N)
            else:
                for i, (index, probe) in enumerate(self):
                    j = index if is_root else i
                    if not N is None:
                        z[j, :] = probe.get_probe_at_snapshot(N)
                    else:
                        for k in range(self.value_size()):
                            z[j, k, :] = probe.get_probe_sub(k)

        # Collect values on root
        recvfrom = comm.gather(len(self), root=root)
        if is_root:
            for j, k in enumerate(recvfrom):
                if comm.Get_rank() != j:
                    ids = comm.recv(source=j, tag=101)
                    z0 = comm.recv(source=j, tag=102)
                    z[ids, :] = z0[:, :]
        else:
            ids = self.get_probe_ids()
            comm.send(ids, dest=root, tag=101)
            comm.send(z, dest=root, tag=102)

        if is_root:
            if filename:
                if not N is None:
                    save(filename + "_snapshot_" + str(N), z)
                else:
                    save(filename + "_all", z)
            return squeeze(z)
