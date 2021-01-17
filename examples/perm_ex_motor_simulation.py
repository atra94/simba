import matplotlib.pyplot as plt
import numba as nb
import numpy as np

from perm_ex_motor_sim_components import RotationalMechanicalLoad,\
    PermanentlyExcitedDCMotor, QuadraticLoadTorque, PController
from simba.basic_components import Sub, TFunction
from simba.core import System

import time

start = time.time()


def reference(t):
    return np.array([100.0], dtype=np.float32) if t > 10.0 else np.array([50.0], dtype=np.float32)


# Initialize Components
reference_generation = TFunction(reference)
sub = Sub()
p_controller = PController(p_gain=100.0)
motor = PermanentlyExcitedDCMotor()
load_torque = QuadraticLoadTorque()
load = RotationalMechanicalLoad()

# Connect Components
sub(in1=reference_generation.outputs['Out'], in2=load.outputs['omega'])
p_controller(error=sub.outputs['Out'])
motor(u=p_controller.outputs['action'], omega=load.outputs['omega'])
load(t=motor.outputs['T'], t_l=load_torque.outputs['T_L'])
load_torque(omega=load.outputs['omega'])
print('Init Time:', time.time() - start )

# Initialize and compile the dynamic system
system = System((reference_generation, sub, p_controller, motor, load_torque, load))
system.compile(numba_compile=True)
system_equation = system.system_equation

print('Compilation Time', time.time() - start)
# Simulate the system equation
system_equation(0.0, np.array([0.0, 0.0], dtype=np.float32))
print('One Call', time.time() - start)
state = np.zeros(system.state_length, dtype=np.float32)
step_size_tau = 1e-4
simulation_time = 40.0
all_states = np.zeros((int(simulation_time/step_size_tau), system.state_length), dtype=np.float32)
ts = np.linspace(0, simulation_time, int(simulation_time/step_size_tau), dtype=np.float32)

#signature = nb.none(nb.typeof(ts), nb.typeof(all_states), nb.typeof(state), nb.typeof(system_equation))

#@nb.njit(signature)
def simulation(ts, all_states, state, system_equation):
    for i, t in enumerate(ts):
        all_states[i] = state
        state += system_equation(t, state) * step_size_tau


simulation(ts, all_states, state, system_equation)
print('Whole Time', time.time() - start)
# Plot the results
plt.plot(ts, all_states[:, 1])
plt.show()
