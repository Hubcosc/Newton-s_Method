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

def round_result(result, rounding_type, rounding_value):
    """Round the result based on the rounding type and value."""
    if rounding_type == "dp":  # Decimal Places
        return round(result, rounding_value)
    elif rounding_type == "sf":  # Significant Figures
        if result == 0:
            return 0
        else:
            return round(result, rounding_value - int(np.floor(np.log10(abs(result)))) - 1)
    return result  # No rounding

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
    rounding_type = data.get('rounding_type', 'none')  # Default to 'none'
    rounding_value = int(data.get('rounding_value', 0))  # Default to 0 for rounding value

    x = sp.symbols('x')

    try:
        # Use sympy's own functions here
        sym_func = sp.sympify(func)
        d_func = sp.diff(sym_func, x)

        # Convert the derivative back to a string for use in eval
        d_func_str = str(d_func)

        root, result_info = newton_method(sym_func, d_func, initial_guess, max_iterations, tolerance)

        if root is not None:
            # Apply rounding if necessary
            rounded_root = round_result(root, rounding_type, rounding_value)
            return jsonify({'result': f'Root found: {rounded_root} in {result_info} iterations.', 'derivative': d_func_str})
        else:
            return jsonify({'result': f'Maximum iterations reached. No solution found. {result_info}', 'derivative': d_func_str})

    except Exception as e:
        return jsonify({'result': f'Error parsing function: {str(e)}', 'derivative': ''})

if __name__ == '__main__':
    app.run(debug=True)