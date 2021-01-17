import matplotlib.pyplot as plt
import numpy as np

from perm_ex_motor_sim_components import RotationalMechanicalLoad,\
    PermanentlyExcitedDCMotor, QuadraticLoadTorque, PController
from simba.basic_components import Sub, TFunction
from simba.core import System


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

# Initialize and compile the dynamic system
system = System((reference_generation, sub, p_controller, motor, load_torque, load))
system.compile(numba_compile=False)

# Simulate the system equation
system_equation = system.system_equation

state = np.zeros(system.state_length)
step_size_tau = 1e-4
simulation_time = 20.0
all_states = np.zeros((int(simulation_time/step_size_tau), system.state_length))
ts = np.linspace(0, simulation_time, int(simulation_time/step_size_tau))
for i, t in enumerate(ts):
    all_states[i] = state
    state = state + system_equation(t, state) * step_size_tau

# Plot the results
plt.plot(ts, all_states[:, 1])
plt.show()
