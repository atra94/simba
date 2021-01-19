import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from perm_ex_motor_sim_components import RotationalMechanicalLoad,\
    PermanentlyExcitedDCMotor, QuadraticLoadTorque, PController
from simba.basic_components import Sub, TFunction
from simba.core import System

import time

start = time.time()


def reference(t):
    return np.array([100.0]) if t > 10.0 else np.array([50.0])


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
system_equation(0.0, np.array([0.0, 0.0]))
print('One Call', time.time() - start)
state = np.zeros(system.state_length, dtype=float)
step_size_tau = 1e-4
simulation_time = 20.0

ts = np.linspace(0, simulation_time, int(simulation_time/step_size_tau), dtype=float)

results = solve_ivp(system_equation, (0, 40.0), np.array([0.0, 0.0], dtype=float), t_eval=ts)

print('Whole Time', time.time() - start)
# Plot the results
plt.plot(ts, results.y[1])
plt.show()
