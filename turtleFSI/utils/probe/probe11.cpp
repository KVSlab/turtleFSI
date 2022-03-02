/*cppimport
<%
from dolfin.jit.jit import dolfin_pc
setup_pybind11(cfg)
cfg['sources'] = ['Probe.cpp', 'Probes.cpp']
cfg['libraries'] = ['dolfin']
cfg['include_dirs'] = dolfin_pc['include_dirs']
cfg['library_dirs'] = dolfin_pc['library_dirs']
%>
*/

#include "Probe.h"
#include "Probes.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace dolfin;
namespace py = pybind11;

PYBIND11_MODULE(probe11, m)
{
    py::class_<Probe, std::shared_ptr<Probe>>(m, "Probe")
        .def(py::init([](const py::array_t<double> x, const py::object v){
            auto _v = v.attr("_cpp_object").cast<const FunctionSpace&>();
            const Array<double> _x(x.size(), const_cast<double*>(x.data()));
            return Probe(_x, _v);
        }))
        //.def(py::init<const py::array_t<double>, std::shared_ptr<const FunctionSpace>>())
        .def("eval", [](Probe& self, py::object v){
            auto _v = v.attr("_cpp_object").cast<const Function&>();
            self.eval(_v);
        })
        .def("get_probe_sub", &Probe::get_probe_sub)
        .def("get_probe_at_snapshot", &Probe::get_probe_at_snapshot)
        .def("get_probe_component_and_snapshot", &Probe::get_probe_component_and_snapshot)
        .def("dump", (void (Probe::*)(std::size_t, std::string, std::size_t)) &Probe::dump, py::arg("i"), py::arg("filename"), py::arg("id")=0, "dump probes")
        .def("dump", (void (Probe::*)(std::string, std::size_t)) &Probe::dump, py::arg("filename"), py::arg("id")=0, "dump probes")
        .def("value_size", &Probe::value_size)
        .def("number_of_evaluations", &Probe::number_of_evaluations)
        .def("coordinates", &Probe::coordinates)
        .def("erase_snapshot", &Probe::erase_snapshot)
        .def("clear", &Probe::clear)
        .def("restart_probe", [](Probe& self, const py::array_t<double> u, std::size_t num_evals){
            const Array<double> _u(u.size(), const_cast<double*>(u.data()));
            self.restart_probe(_u, num_evals);
        })
        .def("restart_probe", [](Probe& self, const py::array_t<double> u){
            const Array<double> _u(u.size(), const_cast<double*>(u.data()));
            self.restart_probe(_u);
        });

    py::class_<Probes, std::shared_ptr<Probes>>(m, "Probes")
        .def(py::init([](const py::array_t<double> x, const py::object v){
            auto _v = v.attr("_cpp_object").cast<const FunctionSpace&>();
            const Array<double> _x(x.size(), const_cast<double*>(x.data()));
            return Probes(_x, _v);
        }))
        .def("eval", [](Probes& self, py::object v){
            auto _v = v.attr("_cpp_object").cast<const Function&>();
            self.eval(_v);
        })
        .def("dump", (void (Probes::*)(std::size_t, std::string)) &Probes::dump, py::arg("i"), py::arg("filename"), "dump probes")
        .def("dump", (void (Probes::*)(std::string)) &Probes::dump, py::arg("filename"), "dump probes")
        .def("add_positions", [](Probes& self, const py::array_t<double> x, const py::object v){
            auto _v = v.attr("_cpp_object").cast<const FunctionSpace&>();
            const Array<double> _x(x.size(), const_cast<double*>(x.data()));
            self.add_positions(_x, _v);
        })
        .def("get_probe", (std::shared_ptr<Probe> (Probes::*)(std::size_t)) &Probes::get_probe)
        .def("get_probe_id", &Probes::get_probe_id)
        .def("get_probe_ids", &Probes::get_probe_ids)
        .def("get_probes_component_and_snapshot", &Probes::get_probes_component_and_snapshot)
        .def("local_size", &Probes::local_size)
        .def("value_size", &Probes::value_size)
        .def("get_total_number_probes", &Probes::get_total_number_probes)
        .def("number_of_evaluations", &Probes::number_of_evaluations)
        .def("erase_snapshot", &Probes::erase_snapshot)
        .def("clear", &Probes::clear)
        .def("set_probes_from_ids", &Probes::set_probes_from_ids);
}
