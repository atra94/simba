import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from scipy.integrate import solve_ivp, ode

from perm_ex_motor_sim_components import RotationalMechanicalLoad,\
    PermanentlyExcitedDCMotor, QuadraticLoadTorque, PController, PIController, DiscreteTimePIController
from simba.basic_components import Sub, TFunction
from simba.core import System

import time

start = time.time()

amplitude = nb.float64([100.0])
step_size_tau = 1e-4
simulation_time = 40.0
half = nb.float64([.5])
full = nb.float64([1.])


def reference(t_):
    #if t_ < 0.5 * simulation_time:
    #    return half * amplitude
    #else:
    #    return full * amplitude
    return amplitude * np.cos(t_)


# Initialize Components
reference_generation = TFunction(reference)
sub = Sub()
p_controller = PIController(p_gain=1.0, i_gain=2.0)
motor = PermanentlyExcitedDCMotor()
load_torque = QuadraticLoadTorque()
load = RotationalMechanicalLoad()

# Connect Components
sub(in1=reference_generation.outputs['Out'], in2=load.outputs['omega'])
p_controller(error=sub.outputs['Out'])
motor(u=p_controller.outputs['action'], omega=load.outputs['omega'])
load(t=motor.outputs['T'], t_l=load_torque.outputs['T_L'])
load_torque(omega=load.outputs['omega'])
print('Init Time:', time.time() - start)

sys_outputs = (
    load.outputs['omega'],
    reference_generation.outputs['Out'],
    motor.outputs['T'],
)
# Initialize and compile the dynamic system
system = System((reference_generation, sub, p_controller, motor, load_torque, load), system_outputs=sys_outputs)
system.compile(numba_compile=True)
system_equation = system.system_equation

print('Compilation Time', time.time() - start)
# Simulate the system equation
system_equation(0.0, np.array([0.0, 0.0]))
print('One Call', time.time() - start)
state = np.zeros(system.state_length, dtype=float)


states = []

ts = np.linspace(0, simulation_time, int(simulation_time/step_size_tau), dtype=float)
state = np.array([0.0, 0.0])
o = ode(system_equation)
o.set_integrator('dopri5', first_step=step_size_tau/10)
counter = 0
t_ = []


def solout(t, y):
    t_.append(t)
    global counter
    counter += 1


o.set_solout(solout)
o.set_initial_value(state, 0.0)
for i, t in enumerate(ts):
    states.append(state)
    state = o.integrate(t)
    #dev = system.system_equation(t, state)
    #state = state + dev * step_size_tau
    if not o.successful():
        print(i)
        o.set_initial_value(state, t)

states = np.array(states).T.reshape(-1, len(ts))
#o.integrate(simulation_time)
print(o.successful())
print(counter)
print('Whole Time', time.time() - start)
# Plot the results
plt.plot(states[0], marker='*')
plt.plot(states[1]*0.5, marker='*')
plt.show()
