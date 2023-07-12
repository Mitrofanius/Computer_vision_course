import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_manipulator(x, x_goal):
    plt.clf()
    x = x.detach().numpy()
    x_goal = x_goal.detach().numpy()
    plt.plot(x[0, :], x[1, :], 'o-', color='b', linewidth=5, mew=5)
    plt.text(x[0, 0], x[1, 0] - 0.2, 'x_base', color='k')
    plt.plot([x[0, 0] - 0.1, x[0, 0] + 0.1], [x[1, 0], x[1, 0]], '-', color='k', linewidth=15, mew=2)
    plt.plot(x_goal[0], x_goal[1], 'x', color='r', linewidth=5, mew=5)
    plt.text(x_goal[0], x_goal[1], 'x_goal', color='r')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-5, 5)
    plt.ylim(-1, 9)
    # plt.axis('equal')
    plt.grid()
    plt.pause(0.001)


def visualize_calibration(X: torch.Tensor, X_goals: torch.Tensor) -> None:
    plt.clf()
    X = X.detach().numpy()
    X_goals = X_goals.detach().numpy()
    for i in range(X.shape[0]):
        x = X[i]
        x_goal = X_goals[:, i]
        plt.plot(x[0, :], x[1, :], 'o-', color='b', linewidth=5, mew=5)
        plt.text(x[0, 0], x[1, 0] - 0.2, 'x_base', color='k')
        plt.plot([x[0, 0] - 0.1, x[0, 0] + 0.1], [x[1, 0], x[1, 0]], '-', color='k', linewidth=15, mew=2)
        plt.plot(x_goal[0], x_goal[1], 'x', color='r', linewidth=5, mew=5)
        plt.text(x_goal[0], x_goal[1], 'x_goal', color='r')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-5, 5)
    plt.ylim(-1, 9)
    # plt.axis('equal')
    plt.grid()
    plt.pause(0.001)


def generate_task_1(number_of_joints: int):
    # Generate task without base
    omega = (torch.rand(number_of_joints) - 0.5) / 1
    rho = torch.ones(number_of_joints, dtype=torch.float32, requires_grad=False)

    base = torch.zeros(2, dtype=torch.float32, requires_grad=False)

    # Generate random goal
    x_goal = torch.tensor((0, 3)) + 6 * (torch.rand(2, dtype=torch.float32) - 0.5)
    return omega, rho, base, x_goal


def generate_task_2():
    number_of_joints = 5
    X_goals = torch.load('./data/X_GOAL')
    omega = torch.load('./data/JOINTS')
    base = torch.tensor([0, 0], dtype=torch.float32, requires_grad=False)
    rho = torch.ones(number_of_joints, dtype=torch.float32, requires_grad=True)
    return omega, rho, base, X_goals
