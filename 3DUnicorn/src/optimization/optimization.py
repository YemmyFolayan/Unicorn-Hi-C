import numpy as np
import matplotlib.pyplot as plt
from utils.gradient_calculator import gradient_calculator
from utils.isconvergence import isconvergence
from utils.convert2xyz import convert2xyz
from utils.gradient_chain_calc import gradient_chain_calc
from utils.initialize_structure_c_style import initialize_structure_c_style


def optimization(n, MAX_ITERATION, LEARNING_RATE, smooth_factor, NEAR_ZERO, lstCons, maxIF, initial_structure=None):
    """
    Optimization function to initialize structure using C-style initialization, calculate objective,
    update variables using Adam optimizer, and loop until convergence.

    Args:
    - n (int): Number of points.
    - MAX_ITERATION (int): Maximum number of iterations.
    - LEARNING_RATE (float): Learning rate for updating variables.
    - smooth_factor (float): Smoothing factor for numerical stability.
    - NEAR_ZERO (float): Convergence threshold.
    - lstCons (ndarray): List of constraints.
    - maxIF (float): Maximum interaction frequency.
    - initial_structure (ndarray, optional): Predefined initial structure as a NumPy array.
    """
    # Check if an initial structure is provided
    if initial_structure is not None:
        print("Using provided initial structure.")
        coordinates = initial_structure
    else:
        # Initialize structure using dynamic cube size
        coordinates = initialize_structure_c_style(
            sequence_length=n,  # Number of points
            min_dist=0.8,       # Minimum distance
            max_dist=5.0        # Maximum distance
        )
    
    # Validate and refine structure if necessary
    if not validate_structure(coordinates, min_dist=0.8, max_dist=5.0):
        print("Initial structure does not satisfy constraints. Refining...")
        coordinates = refine_path_continuity(coordinates)
        #coordinates = simulated_annealing(coordinates, min_dist=0.8, max_dist=5.0)
    
    # Flatten coordinates for optimization
    variables = coordinates.flatten()
    len_ = len(variables)

    # Adam optimizer parameters
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8
    m = np.zeros(len_)
    v = np.zeros(len_)
    t = 0  # Time step

    # First calculation of Objective function
    v_, w, dl_dw, dw_dv = gradient_chain_calc(lstCons, variables, n)
    change, cost = gradient_calculator(lstCons, variables, dl_dw, dw_dv, maxIF, n)

    # Optimization loop
    count = 0
    converge = isconvergence(change, cost, NEAR_ZERO)

    while count < MAX_ITERATION and not converge:
        count += 1
        t += 1

        # Recalculate v, w, dl_dw, dw_dv and gradients
        v_, w, dl_dw, dw_dv = gradient_chain_calc(lstCons, variables, n)
        change, newobj = gradient_calculator(lstCons, variables, dl_dw, dw_dv, maxIF, n)

        # Update variables with Adam optimizer
        for i in range(len_):
            # Update biased first moment estimate
            m[i] = beta1 * m[i] + (1 - beta1) * change[i]
            # Update biased second raw moment estimate
            v[i] = beta2 * v[i] + (1 - beta2) * (change[i] ** 2)
            # Compute bias-corrected first moment estimate
            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)
            # Update the variables
            variables[i] += LEARNING_RATE * m_hat / (np.sqrt(v_hat) + epsilon)

        # Save the updated coordinates to a text file for tracking
        if count % 10 == 0 or count == MAX_ITERATION:  # Save every 10 iterations and the final iteration
            optimized_coordinates = variables.reshape(-1, 3)
            #np.savetxt(f"optimized_coordinates_iter_{count}.txt", optimized_coordinates, fmt="%.6f", delimiter=" ")
            #print(f"Optimized coordinates saved to 'optimized_coordinates_iter_{count}.txt'")

        # Check for convergence
        converge = isconvergence(change, cost, NEAR_ZERO)
        print(f"Iteration {count}, objective function: {newobj:.5f}")

        cost = newobj

    # Final save of the optimized structure
    optimized_coordinates = variables.reshape(-1, 3)
    optimized_coordinates = refine_path_continuity(optimized_coordinates)

    #np.savetxt("optimized_coordinates_final.txt", optimized_coordinates, fmt="%.6f", delimiter=" ")
    #print("Final optimized coordinates saved to 'optimized_coordinates_final.txt'")

    return variables, change, cost


def validate_structure(coordinates, min_dist, max_dist):
    for i in range(len(coordinates) - 1):
        dist = np.linalg.norm(coordinates[i + 1] - coordinates[i])
        if dist < min_dist or dist > max_dist:
            return False
    return True


def refine_path_continuity(coordinates, max_step=3):
    for i in range(1, len(coordinates)):
        step = coordinates[i] - coordinates[i - 1]
        step_norm = np.linalg.norm(step)
        if step_norm > max_step:
            coordinates[i] = coordinates[i - 1] + (step / step_norm) * max_step
    return coordinates
