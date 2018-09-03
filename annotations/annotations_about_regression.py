# Created by Lorenaps at 02/09/18
'''
    Implementando regressão linear com base no vídeo:
    https://www.youtube.com/watch?v=XdM6ER7zTLk
'''
import numpy as np


def compute_error_for_given_points(b, m, points):
    total_error = 0

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2

    return total_error / float(len(points))


def step_gradient(current_b, current_m, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    print("***************")
    print("Currents b = {0} and m = {1}".format(current_b, current_m))
    print("Error {0}".format(compute_error_for_given_points(current_b, current_m, points)))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((current_m * x) + current_b))
        m_gradient += -(2 / N) * x * (y - ((current_m * x) + current_b))

    # Se o gradiente é negativo queremos aumentar o valor do coeficiente.
    # Se positivo, queremos diminuir. Logo, subtrair a multiplicação entre
    # a taxa de aprendizado e o gradiente dos respectivos valores atuais dos
    # coeficientes garante esse comportamento.
    print(" --- Learning rate = {0} * b_gradiente = {1} = {2}".
          format(learning_rate, b_gradient, learning_rate * b_gradient))
    print(" --- Learning rate = {0} * m_gradiente = {1} = {2}".
          format(learning_rate, m_gradient, learning_rate * m_gradient))

    new_b = current_b - (learning_rate * b_gradient)
    new_m = current_m - (learning_rate * m_gradient)

    print("News b = {0} and m = {1}".format(new_b, new_m))
    return [new_b, new_m]


def gradient_descent_runner(points, learning_rate, starting_b, starting_m, num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)

    return [b, m]

def run():
    points = np.genfromtxt('../datasets/data.csv', delimiter=',')

    learning_rate = 0.0001

    # Coeficiente linear
    initial_b = 0

    # Coeficiente angular
    initial_m = 0

    num_iterations = 10

    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".
            format(initial_b, initial_m,
            compute_error_for_given_points(initial_b, initial_m, points)))

    print("Running...")

    [b, m] = gradient_descent_runner(points, learning_rate, initial_b, initial_m, num_iterations)

    print("After {0} iterations b = {1}, m = {2}, error = {3}".
            format(num_iterations, b, m,
            compute_error_for_given_points(b, m, points)))

if __name__ == '__main__':
    run()