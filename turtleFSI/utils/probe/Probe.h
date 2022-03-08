#ifndef __PROBE_H
#define __PROBE_H

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <fstream>
#include <boost/lexical_cast.hpp>

namespace dolfin
{

  class Probe
  {

  public:

    Probe(const Array<double>& x, const FunctionSpace& V);

    Probe(const Probe& p);

    virtual ~Probe();

    // evaluate the probe.
    virtual void eval(const Function& u);

    // Return all probe values for one component
    std::vector<double> get_probe_sub(std::size_t i);

    // Return one snapshot of the probe values for all components
    std::vector<double> get_probe_at_snapshot(std::size_t i);

    // Return one probe values for given component at given snapshot
    double get_probe_component_and_snapshot(std::size_t comp, std::size_t i);

    // dump component i to filename. id is optional probe identifier
    void dump(std::size_t i, std::string filename, std::size_t id=0);

    // dump all components to filename. id is optional probe identifier
    void dump(std::string filename, std::size_t id=0);

    // Return number of components in probe (one for scalar, ...)
    std::size_t value_size() {return _value_size_loc;};

    // Return number of evaluations of probe
    std::size_t number_of_evaluations() {return _num_evals;};

    // Return coordinates of probe
    std::vector<double> coordinates();

    // Erase one snapshot of probe
    virtual void erase_snapshot(std::size_t i);

    // Reset probe by deleting all previous evaluations
    virtual void clear();

    // Reset probe values for entire tensor
    virtual void restart_probe(const Array<double>& u, std::size_t num_evals)
      {dolfin_error("Probe.h","restart probe", "Restart only for StatisticsProbe");};

    void restart_probe(const Array<double>& u);

  protected:

    std::vector<std::vector<double> > basis_matrix;

    std::vector<double> coefficients;

    std::vector<double> vertex_coordinates;

    std::vector<double> _x;

    std::shared_ptr<const FiniteElement> _element;

    std::size_t _value_size_loc, _num_evals;

    std::vector<std::vector<double> > _probes;

    std::unique_ptr<Cell> dolfin_cell;

    ufc::cell ufc_cell;

  };

}

#endif
