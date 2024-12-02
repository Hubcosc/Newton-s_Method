from flask import Flask, render_template, request, jsonify
import numpy as np
import sympy as sp

app = Flask(__name__)


def newton_method(func, d_func, initial_guess, max_iterations=100, tolerance=1e-7):
    x_n = initial_guess
    for iteration in range(max_iterations):
        try:
            # Create a lambda function from the sympy expressions
            f = sp.lambdify(sp.symbols('x'), func, 'numpy')
            df = sp.lambdify(sp.symbols('x'), d_func, 'numpy')

            # Evaluate the function and its derivative
            f_x_n = f(x_n)
            d_f_x_n = df(x_n)

            if d_f_x_n == 0:
                return None, iteration  # Derivative is zero, no solution

            x_n1 = x_n - f_x_n / d_f_x_n

            if abs(x_n1 - x_n) < tolerance:
                return x_n1, iteration + 1

            x_n = x_n1

        except Exception as e:
            return None, f"Error evaluating the function or derivative: {str(e)}"

    return None, max_iterations


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    func = data['function']
    initial_guess = float(data['initial_guess'])
    max_iterations = int(data['max_iterations'])
    tolerance = float(data['tolerance'])

    x = sp.symbols('x')

    try:
        # Use sympy's own functions here
        sym_func = sp.sympify(func)
        d_func = sp.diff(sym_func, x)

        # Convert the derivative back to a string for use in eval
        d_func_str = str(d_func)

        root, result_info = newton_method(sym_func, d_func, initial_guess, max_iterations, tolerance)

        if root is not None:
            return jsonify({'result': f'Root found: {root} in {result_info} iterations.', 'derivative': str(d_func)})
        else:
            return jsonify(
                {'result': f'Maximum iterations reached. No solution found. {result_info}', 'derivative': str(d_func)})

    except Exception as e:
        return jsonify({'result': f'Error parsing function: {str(e)}', 'derivative': ''})


if __name__ == '__main__':
    app.run(debug=True)