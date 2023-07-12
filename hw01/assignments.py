from numpy import diag, diagonal
import torch


def dkt(omega: torch.Tensor, rho: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Direct Kinetic Task non-vectorized
    :param omega: A tensor of shape (n,) containing joint angles, where n is number of joints
    :param rho: A tensor of shape (n,) containing link lengths, where n is number of joints
    :param base: A tensor of shape (2) representing base angle and length with respect to origin
    :return: A tensor X of shape (2, n) containing joint positions, where n is number of joints
    """
    n = omega.shape[0]

    omega_base = base[0]
    rho_base = base[1]

    # Determine base position
    phi = omega_base
    pos_x = rho_base * torch.cos(phi)
    pos_y = rho_base * torch.sin(phi)

    # Add base position to list of joint positions
    x = [torch.stack((pos_x, pos_y))]

    for k in range(n):
        # Calculate joint position
        # TODO: Implement this loop
        # Hint: You can use the previous joint position to calculate the current joint position
        # -------------------------------------------------
        # START OF YOUR CODE
        phi   = phi + omega[k]
        pos_x = pos_x + rho[k] * torch.cos(phi)
        pos_y = pos_y + rho[k] * torch.sin(phi)
        # -------------------------------------------------

        # Add joint position to list of joint positions
        x.append(torch.stack((pos_x, pos_y)))

    return torch.stack(x, dim=1)


def dkt_vectorized(omega: torch.Tensor, rho: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Direct Kinetic Task vectorized
    :param omega: A tensor of shape (n,) containing joint angles, where n is number of joints
    :param rho: A tensor of shape (n,) containing link lengths, where n is number of joints
    :param base: A tensor of shape (2) representing base angle and length with respect to origin
    :return: A tensor X of shape (2, n) containing joint positions, where n is number of joints
    """

    # print(dkt(omega, rho, base))


    # Determine number of joints from input
    n = omega.shape[0] + 1

    # Add base to length and angle vectors
    omega = torch.cat((base[0].unsqueeze(0), omega)).unsqueeze(1)
    rho = torch.cat((base[0].unsqueeze(0), rho))
    phi = (torch.tril((torch.ones(n, n))) @ omega).squeeze(1)
    phi_trig = torch.stack((torch.cos(phi), torch.sin(phi)))
    delta = phi_trig @ torch.diag(rho)

    # TODO: Implement vectorized version of DKT
    # -------------------------------------------------
    # START OF YOUR CODE
    X = delta @ torch.triu(torch.ones(n, n))
    # -------------------------------------------------
    return X


def manipulator_loss(X: torch.Tensor, x_goal: torch.Tensor) -> torch.Tensor:
    """Manipulator loss
    :param X: A tensor of shape (2, n) containing joint positions, where n is number of joints
    :param x_goal: A tensor of shape (2,) representing goal position
    :return loss: A scalar representing the loss

    Tip: You can add another arguments to this function if you want to, but it is not necessary
    """

    # TODO: Implement loss function
    # -------------------------------------------------
    # START OF YOUR CODE
    loss = torch.sqrt(torch.sum((X[:, -1] - x_goal)**2))

    # -------------------------------------------------
    return loss


def calibration_loss(X: torch.Tensor, x_goal: torch.Tensor) -> torch.Tensor:
    """Calibration loss
    :param X: A tensor of shape (2, n) containing joint positions, where n is number of joints
    :param x_goal: A tensor of shape (2,) representing goal position
    :return loss: A scalar representing the loss

    Tip: You can add another arguments to this function if you want to, but it is not necessary
    """

    # TODO: Implement loss function
    # -------------------------------------------------
    # START OF YOUR CODE]
    # Couldn't come up with anything else
    loss = torch.sqrt(torch.sum((X[:, -1] - x_goal)**2))

    # -------------------------------------------------
    return loss
