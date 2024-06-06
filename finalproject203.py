# Wind turbine parameters
J = 0.5  # Inertia (kg.m^2)
B = 0.01  # Damping coefficient (N.m.s)
K = 1.0  # Gain constant

def wind_turbine_dynamics(state, control_input, wind_speed):
    """
    Simulates the dynamics of the wind turbine.
    state: [rotor_speed, generator_torque]
    control_input: [blade_pitch_angle, generator_torque]
    wind_speed: current wind speed
    """
    rotor_speed, generator_torque = state
    blade_pitch_angle = control_input[0]

    # Simplified aerodynamic torque equation
    aerodynamic_torque = K * wind_speed * (1 - blade_pitch_angle / 90)

    # Rotor speed dynamics
    rotor_speed_dot = (aerodynamic_torque - generator_torque - B * rotor_speed) / J

    # Update state
    new_rotor_speed = rotor_speed + rotor_speed_dot
    new_generator_torque = control_input[1]  # Assume generator torque is directly controlled

    # Power output (simplified)
    power_output = generator_torque * rotor_speed

    return np.array([new_rotor_speed, new_generator_torque]), power_output

from scipy.optimize import minimize
import numpy as np

def mpc_control(state, wind_speed, prediction_horizon=10, Q=np.eye(2), R=np.eye(2)):
    """
    Model Predictive Control to determine the optimal control action.
    state: [rotor_speed, generator_torque]
    wind_speed: current wind speed
    prediction_horizon: number of steps to predict into the future
    Q: state weighting matrix
    R: control weighting matrix
    """
    def cost_function(control_inputs):
        cost = 0
        current_state = state.copy()
        control_inputs = np.reshape(control_inputs, (prediction_horizon, 2))  # Reshape control inputs to match the prediction horizon
        for i in range(prediction_horizon):
            control_input = control_inputs[i]
            next_state, _ = wind_turbine_dynamics(current_state, control_input, wind_speed)
            cost += np.dot((next_state - desired_state).T, Q).dot(next_state - desired_state) + np.dot(control_input, R).dot(control_input.T)
            current_state = next_state
        return cost

    initial_guess = np.zeros(prediction_horizon * 2)
    bounds = [(-10, 10)] * prediction_horizon + [(0, 10)] * prediction_horizon  # Assume blade pitch between -10 and 10, generator torque between 0 and 10

    result = minimize(cost_function, initial_guess, bounds=bounds)
    optimal_control = result.x[:2]  # Only return the first control input pair

    return optimal_control


# Define desired state for the turbine
desired_state = np.array([10, 5])  # Example desired state: [rotor_speed, generator_torque]

!pip install deap
from deap import base, creator, tools, algorithms

# Global parameters for the GA
initial_state = np.array([0, 0])  # Initial state of the turbine
wind_speed = 10  # Constant wind speed for simplicity

# Create the optimization problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)  # Q and R matrices are 2x2 and 1x1

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    Q = np.diag([individual[0], individual[0]])
    R = np.diag([individual[1], individual[1]])
    control_action = mpc_control(initial_state, wind_speed, Q=Q, R=R)
    final_state, power_output = wind_turbine_dynamics(initial_state, control_action, wind_speed)
    cost = np.linalg.norm(final_state - desired_state)
    return (cost,)


toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Genetic Algorithm parameters
population_size = 50
num_generations = 40
prob_crossover = 0.7
prob_mutation = 0.2

# Initialize population
population = toolbox.population(n=population_size)

# Run Genetic Algorithm
algorithms.eaSimple(population, toolbox, cxpb=prob_crossover, mutpb=prob_mutation, ngen=num_generations, verbose=True)


import matplotlib.pyplot as plt
# Simulation parameters
time_steps = 100

# Initialize metrics
states = [initial_state]
power_outputs = []
control_efforts = []

# Simulate the wind turbine with MPC
for t in range(time_steps):
    control_action = mpc_control(states[-1], wind_speed)
    new_state, power_output = wind_turbine_dynamics(states[-1], control_action, wind_speed)
    states.append(new_state)
    power_outputs.append(power_output)
    control_efforts.append(control_action)

# Convert to numpy arrays for easier analysis
states = np.array(states)
power_outputs = np.array(power_outputs)
control_efforts = np.array(control_efforts)

# Plot results
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(states[:, 0], label='Rotor Speed')
plt.plot(states[:, 1], label='Generator Torque')
plt.xlabel('Time Step')
plt.ylabel('State')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(power_outputs, label='Power Output')
plt.xlabel('Time Step')
plt.ylabel('Power Output (W)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(control_efforts[:, 0], label='Blade Pitch Angle')
plt.plot(control_efforts[:, 1], label='Generator Torque Control')
plt.xlabel('Time Step')
plt.ylabel('Control Effort')
plt.legend()

plt.tight_layout()
plt.show()

def assess_model(states, power_outputs, control_efforts, desired_state, time_steps):
    # Calculate total power output
    total_power_output = np.sum(power_outputs)

    # Calculate total control effort
    total_control_effort = np.sum(np.abs(control_efforts))

    # Calculate tracking error
    desired_states = np.tile(desired_state, (time_steps+1, 1))
    tracking_error = np.mean((states - desired_states)**2, axis=0)

    # Print metrics
    print(f"Total Power Output: {total_power_output:.2f} W")
    print(f"Total Control Effort: {total_control_effort:.2f}")
    print(f"Tracking Error (Rotor Speed): {tracking_error[0]:.2f}")
    print(f"Tracking Error (Generator Torque): {tracking_error[1]:.2f}")

    # Plot state trajectories
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(states[:, 0], label='Rotor Speed')
    plt.plot([desired_state[0]] * len(states), 'r--', label='Desired Rotor Speed')
    plt.xlabel('Time Step')
    plt.ylabel('Rotor Speed')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(states[:, 1], label='Generator Torque')
    plt.plot([desired_state[1]] * len(states), 'r--', label='Desired Generator Torque')
    plt.xlabel('Time Step')
    plt.ylabel('Generator Torque')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Assess MPC model
print("MPC Model Assessment:")
assess_model(states, power_outputs, control_efforts, desired_state, time_steps)


import numpy as np

# Wind turbine parameters
J = 0.5  # Inertia (kg.m^2)
B = 0.01  # Damping coefficient (N.m.s)
K = 1.0  # Gain constant

def wind_turbine_dynamics(state, control_input, wind_speed):
    """
    Simulates the dynamics of the wind turbine.
    state: [rotor_speed, generator_torque]
    control_input: [blade_pitch_angle, generator_torque]
    wind_speed: current wind speed
    """
    rotor_speed, generator_torque = state
    blade_pitch_angle = control_input[0]

    # Simplified aerodynamic torque equation
    aerodynamic_torque = K * wind_speed * (1 - blade_pitch_angle / 90)

    # Rotor speed dynamics
    rotor_speed_dot = (aerodynamic_torque - generator_torque - B * rotor_speed) / J

    # Update state
    new_rotor_speed = rotor_speed + rotor_speed_dot
    new_generator_torque = control_input[1]  # Assume generator torque is directly controlled

    # Power output (simplified)
    power_output = generator_torque * rotor_speed

    return np.array([new_rotor_speed, new_generator_torque]), power_output

from scipy.optimize import minimize

def mpc_control(state, wind_speed, prediction_horizon=10, Q=np.eye(2), R=np.eye(2)):
    """
    Model Predictive Control to determine the optimal control action.
    state: [rotor_speed, generator_torque]
    wind_speed: current wind speed
    prediction_horizon: number of steps to predict into the future
    Q: state weighting matrix
    R: control weighting matrix
    """
    def cost_function(control_inputs):
        cost = 0
        current_state = state.copy()
        control_inputs = np.reshape(control_inputs, (prediction_horizon, 2))  # Reshape control inputs to match the prediction horizon
        for i in range(prediction_horizon):
            control_input = control_inputs[i]
            next_state, _ = wind_turbine_dynamics(current_state, control_input, wind_speed)
            cost += np.dot((next_state - desired_state).T, Q).dot(next_state - desired_state) + np.dot(control_input, R).dot(control_input.T)
            current_state = next_state
        return cost

    initial_guess = np.zeros(prediction_horizon * 2)
    bounds = [(-10, 10)] * prediction_horizon + [(0, 10)] * prediction_horizon  # Assume blade pitch between -10 and 10, generator torque between 0 and 10

    result = minimize(cost_function, initial_guess, bounds=bounds)
    optimal_control = result.x[:2]  # Only return the first control input pair

    return optimal_control

# Define desired state for the turbine
desired_state = np.array([10, 5])  # Example desired state: [rotor_speed, generator_torque]

import matplotlib.pyplot as plt

# Simulation parameters
time_steps = 100
initial_state = np.array([0, 0])  # Initial state of the turbine
wind_speed = 100  # Constant wind speed for simplicity

# Initialize metrics
states = [initial_state]
power_outputs = []
control_efforts = []

# Simulate the wind turbine with MPC
for t in range(time_steps):
    control_action = mpc_control(states[-1], wind_speed)
    new_state, power_output = wind_turbine_dynamics(states[-1], control_action, wind_speed)
    states.append(new_state)
    power_outputs.append(power_output)
    control_efforts.append(control_action)

# Convert to numpy arrays for easier analysis
states = np.array(states)
power_outputs = np.array(power_outputs)
control_efforts = np.array(control_efforts)

# Plot results
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(states[:, 0], label='Rotor Speed')
plt.plot(states[:, 1], label='Generator Torque')
plt.xlabel('Time Step')
plt.ylabel('State')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(power_outputs, label='Power Output')
plt.xlabel('Time Step')
plt.ylabel('Power Output (W)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(control_efforts[:, 0], label='Blade Pitch Angle')
plt.plot(control_efforts[:, 1], label='Generator Torque Control')
plt.xlabel('Time Step')
plt.ylabel('Control Effort')
plt.legend()

plt.tight_layout()
plt.show()

def assess_model(states, power_outputs, control_efforts, desired_state, time_steps):
    # Calculate total power output
    total_power_output = np.sum(power_outputs)

    # Calculate total control effort
    total_control_effort = np.sum(np.abs(control_efforts))

    # Calculate tracking error
    desired_states = np.tile(desired_state, (time_steps+1, 1))
    tracking_error = np.mean((states - desired_states)**2, axis=0)

    # Print metrics
    print(f"Total Power Output: {total_power_output:.2f} W")
    print(f"Total Control Effort: {total_control_effort:.2f}")
    print(f"Tracking Error (Rotor Speed): {tracking_error[0]:.2f}")
    print(f"Tracking Error (Generator Torque): {tracking_error[1]:.2f}")

    # Plot state trajectories
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(states[:, 0], label='Rotor Speed')
    plt.plot([desired_state[0]] * len(states), 'r--', label='Desired Rotor Speed')
    plt.xlabel('Time Step')
    plt.ylabel('Rotor Speed')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(states[:, 1], label='Generator Torque')
    plt.plot([desired_state[1]] * len(states), 'r--', label='Desired Generator Torque')
    plt.xlabel('Time Step')
    plt.ylabel('Generator Torque')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Assess MPC model
print("MPC Model Assessment:")
assess_model(states, power_outputs, control_efforts, desired_state, time_steps)

