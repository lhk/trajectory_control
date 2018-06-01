# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

c = 10
m = 5
I = 3
l = 5
g = 9.81

T = 10


# functions to create the matrices Ax + Bu
def A():
    return np.array([[0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0]])


def B(theta, m, l, I):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([[-s / m, c / m],
                     [0, 0],
                     [c / m, s / m],
                     [0, 0],
                     [0, l / I],
                     [0, 0]])


# see : https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
def discretize_AB(mat_a, mat_b, T):
    num_states = mat_a.shape[1]
    num_controls = mat_b.shape[1]

    blocked = np.zeros(2 * (num_states + num_controls,))
    blocked[:num_states, :num_states] = mat_a
    blocked[:num_states, num_states:] = mat_b

    exp_blocked = scipy.linalg.expm(blocked * T)

    mat_a_d = exp_blocked[:num_states, :num_states]
    mat_b_d = exp_blocked[:num_states, num_states:]

    return mat_a_d, mat_b_d


discretize_AB(A(), B(1, m, l, I), T)

steps = 5

# trajectory is
# [dx, x, dy, y, dtheta, theta]

trajectory_start = np.array([0, 100, 0, 20, 0, 0]).reshape((6, 1))
trajectory_end = np.zeros((6,)).reshape((6, 1))

trajectory = trajectory_end - trajectory_start * np.linspace(0, 1, steps + 1).reshape((1, steps + 1)) + trajectory_start

trajectory.shape

import cvxpy as cvx

states = cvx.Variable(6, steps)
deltas = cvx.Variable(6, steps)
forces = cvx.Variable(2, steps)

for i in range(10):
    constraints = []

    # state transitions as constraint
    for t in range(steps):
        # for the trajectory
        current_state = trajectory[:, t]
        next_state = trajectory[:, t + 1]

        # read the current orientation,
        # set up linearization and discretize
        current_theta = current_state[5]
        mat_a = A()
        mat_b = B(current_theta, m, l, I)
        mat_a_d, mat_b_d = discretize_AB(mat_a, mat_b, T)

        # the next state is given by simulating the dynamics from the current trajectory position
        constraints.append(states[:, t] == mat_a_d @ current_state + mat_b_d * forces[:, t] + np.array(
            [0, 0, -T * g, -T ** 2 / 2 * g, 0, 0]))

        # the delta is given by the difference between this state and the predicted trajectory
        constraints.append(next_state == states[:, t] + deltas[:, t])

    # positions never below 0
    constraints.append(states[1, :] > 0)
    constraints.append(states[3, :] > 0)

    # forces always positive
    constraints.append(forces > 0)

    objective = cvx.Minimize(cvx.sum_squares(deltas) + cvx.sum_squares(forces))

    problem = cvx.Problem(objective, constraints)

    problem.solve()


    plt.figure(figsize=(10, 10))
    plt.title("toller plot")
    plt.plot(trajectory.T)
    # plt.plot(forces.value.T)
    plt.legend(["dx", "x", "dy", "y", "dtheta", "theta"])
    # plt.legend(["fp", "fq"])
    plt.show()


    trajectory[:, 1:] = states.value#trajectory[:, 1:] - deltas.value

plt.figure(figsize=(10, 10))
plt.title("toller plot")
plt.plot(deltas.value.T)
# plt.plot(forces.value.T)
plt.legend(["dx", "x", "dy", "y", "dtheta", "theta"])
# plt.legend(["fp", "fq"])
plt.show()

plt.figure(figsize=(10, 10))
plt.title("toller plot")
plt.plot(trajectory.T)
# plt.plot(forces.value.T)
plt.legend(["dx", "x", "dy", "y", "dtheta", "theta"])
# plt.legend(["fp", "fq"])
plt.show()

steps = 10000
states = np.zeros((6, steps + 1))
states[:, 0] = trajectory[:, 0]

forces = np.zeros((2, steps))
forces[0, :] = m * g
forces[1, :] = 0.1

for t in range(steps):
    # read the current orientation,
    # set up linearization and discretize
    current_theta = states[5, t]
    mat_a = A()
    mat_b = B(current_theta, m, l, I)
    mat_a_d, mat_b_d = discretize_AB(mat_a, mat_b, T)

    # the next state is given by simulating the dynamics from the current trajectory position
    states[:, t + 1] = mat_a_d @ states[:, t] + mat_b_d @ forces[:, t]  # - np.array([0, 0, T*g, T**2/2*g, 0, 0])

plt.figure(figsize=(10, 10))
plt.title("toller plot")
plt.plot(states.T[:, :])
# plt.plot(forces.value.T)
plt.legend(["dx", "x", "dy", "y", "dtheta", "theta"])
plt.show()
