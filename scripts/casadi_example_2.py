import casadi as ca
from os import system

# Define your CasADi function
def create_function():
    x = ca.MX.sym('x', 2, 1)  # 2x1 vector
    Q = ca.MX.sym('Q', 2, 2)  # 2x2 matrix, assumed positive definite
    c = ca.MX.sym('Q', 2, 1)  # 2x2 matrix, assumed positive definite
    xd = ca.MX.sym('xd', 2, 1)
    xe = x - xd
    # Define the quadratic cost function x^T * Q * x
    cost = (1/2)*ca.mtimes([xe.T, Q, xe]) + c.T@xe
    gradient = ca.gradient(cost, x)
    Hessian = ca.jacobian(gradient, x)
    delta = - ca.inv(Hessian)@gradient
    f = ca.Function('cost', [Q, c, x, xd], [cost], ['Q', 'c', 'x', 'xd'], ['cost'])  # Define CasADi function
    grad_f = ca.Function('gradient', [Q, c, x, xd], [gradient], ['Q', 'c', 'x', 'xd'], ['gradient'])  # Define CasADi function
    hess_f = ca.Function('hessian', [Q, c, x, xd], [Hessian], ['Q', 'c', 'x', 'xd'], ['hessian'])  # Define CasADi function
    delta_f = ca.Function('delta_x', [Q, c, x, xd], [delta], ['Q', 'c', 'x', 'xd'], ['delta'])
    return f, grad_f, hess_f, delta_f

# Create the function
quadratic_cost_function, gradient_f, hessian_f, delta_f = create_function()

# Use CodeGenerator to set up code generation with specific options
name = "my_function_cost"
name_1 = "my_function_gradient"
name_2 = "my_function_hessian"
name_3 = "my_function_delta"

# Add the function to the code generator

# Generate both C and header files with options
opts = dict(main=False, \
            mex=False, \
            with_header=True, \
            casadi_real = "float")

cname = quadratic_cost_function.generate(name, opts)
cname_gradient = gradient_f.generate(name_1, opts)
cname_hessian = hessian_f.generate(name_2, opts)
cname_delta = delta_f.generate(name_3, opts)

# Compile the generated C code with optimization flags
oname_O3 = name + '_Ofast.so'
oname_O3_gradient = name_1 + '_Ofast.so'
oname_O3_hessian = name_2 + '_Ofast.so'
oname_O3_delta = name_3 + '_Ofast.so'

print('Compiling with O3 optimization: ', oname_O3)
system(f'gcc -fPIC -shared -Ofast -march=native {cname} -o {oname_O3}')
system(f'gcc -fPIC -shared -Ofast -march=native {cname_gradient} -o {oname_O3_gradient}')
system(f'gcc -fPIC -shared -Ofast -march=native {cname_hessian} -o {oname_O3_hessian}')
system(f'gcc -fPIC -shared -Ofast -march=native {cname_delta} -o {oname_O3_delta}')

print("CasADi version:", ca.__version__)