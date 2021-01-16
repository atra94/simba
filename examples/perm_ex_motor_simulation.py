import matplotlib.pyplot as plt
import numpy as np

from perm_ex_motor_sim_components import RotationalMechanicalLoad,\
    PermanentlyExcitedDCMotor, QuadraticLoadTorque, PController
from simba.basic_components import Sub, TFunction
from simba.core import System


def reference(t):
    return 100 if t > 10.0 else 50.0


# Initialize Components
reference_generation = TFunction(reference)
sub = Sub()
p_controller = PController()
motor = PermanentlyExcitedDCMotor()
load_torque = QuadraticLoadTorque()
load = RotationalMechanicalLoad()

# Connect Components
sub(in1=load.outputs['omega'], in2=reference_generation.outputs['Out'])
p_controller(error=sub.outputs['Out'])
motor(u=p_controller.outputs['action'], omega=load.outputs['omega'])
load(t=motor.outputs['T'], t_l=load_torque.outputs['T_L'])


# Initialize and compile the dynamic system
system = System((reference_generation, sub, p_controller, motor, load_torque, load))
system.compile(numba_compile=False)

# Simulate the system equation
system_equation = system.system_equation

state = np.zeros(system.state_length)
step_size_tau = 1e-4

all_states = np.zeros((int(20/step_size_tau), system.state_length))
ts = enumerate(np.linspace(0, 20, int(20/step_size_tau)))
for i, t in ts:
    all_states[i] = state
    state = state + system_equation(t, state) * step_size_tau

# Plot the results

plt.plot(ts, all_states[:, 0])
plt.show()
