#include <gz/sim/Server.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <pybind11/pybind11.h>

// Expose native methods missing from gz.sim8, only for offline fixture resets.
PYBIND11_MODULE(fixture_bridge, m) {
  m.def("run_paused_once", [](gz::sim::Server &s) {
    return s.RunOnce(true);
  }, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("release_velocity", [](gz::sim::EntityComponentManager &ecm, gz::sim::Entity link) {
    return ecm.RemoveComponent<gz::sim::components::AngularVelocityCmd>(link) &&
           ecm.RemoveComponent<gz::sim::components::LinearVelocityCmd>(link);
  });
}
