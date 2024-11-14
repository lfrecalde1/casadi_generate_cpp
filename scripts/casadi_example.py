import casadi as ca
from os import system

# Define your CasADi function
def create_function():
    x = ca.MX.sym('x', 1000, 1)  # 2x1 vector
    Q = ca.MX.sym('Q', 1000, 1000)  # 2x2 matrix, assumed positive definite
    # Define the quadratic cost function x^T * Q * x
    cost = ca.mtimes([x.T, Q, x])
    f = ca.Function('cost', [Q, x], [cost], ['Q', 'x'], ['cost'])  # Define CasADi function
    return f

# Create the function
quadratic_cost_function = create_function()

# Use CodeGenerator to set up code generation with specific options
name = "my_function"

# Add the function to the code generator

# Generate both C and header files with options
opts = dict(main=False, \
            mex=False, \
            with_header=True, \
            casadi_real = "float")
cname = quadratic_cost_function.generate(name, opts)

# Compile the generated C code with optimization flags
oname_O3 = name + '_Ofast.so'

print('Compiling with O3 optimization: ', oname_O3)
system(f'gcc -fPIC -shared -Ofast -march=native {cname} -o {oname_O3}')

print("CasADi version:", ca.__version__)