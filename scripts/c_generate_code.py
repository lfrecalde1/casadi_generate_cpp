from casadi import *
from numpy import *
from os import system
import time
import sys


compileme = True  # Flag wether compiling should take place or not

if len(sys.argv)>1 and sys.argv[1]=='nc':
  compileme = False
else:
  print("Info: Use 'python c_code_generation.py nc' to omit compiling")

# Form an expression for the gradient of the determinant
x = SX.sym('x', 7, 7)
gd = casadi.gradient(det(x), x)

# Random point to evaluate it
x0 = DM.rand(7, 7)

# Form a function and generate C code
name = 'grad_det'
grad_det = Function(name, [x], [gd], ['x'], ['gd'])
cname = grad_det.generate()

oname_no_opt = name + '_no_opt.so'
print('Compiling without optimization: ', oname_no_opt)
t1 = time.time()
if compileme:
  system('gcc -fPIC -shared ' + cname + ' -o ' + oname_no_opt)
t2 = time.time()
print('time = ', (t2-t1)*1e3, ' ms')

oname_O3 = name + '_O3.so'
print('Compiling with O3 optimization: ', oname_O3)
t1 = time.time()
if compileme:
  system('gcc -fPIC -shared -O3 ' + cname + ' -o ' + oname_O3)
t2 = time.time()
print('time = ', (t2-t1)*1e3, ' ms')

oname_Os = name + '_Os.so'
print('Compiling with Os optimization: ', oname_Os)
t1 = time.time()
if compileme:
  system('gcc -fPIC -shared -Os ' + cname + ' -o ' + oname_Os)
t2 = time.time()
print('time = ', (t2-t1)*1e3, ' ms')

# Read function
grad_det_no_opt = external(name, './'+oname_no_opt)
grad_det_O3 = external(name, './'+oname_O3)
grad_det_Os = external(name, './'+oname_O3)
f_test = [grad_det, grad_det_no_opt, grad_det_O3, grad_det_Os]

for f in f_test:
  t1 = time.time()
  nrep = 10000
  for r in range(nrep):
    r = f(x0)
  t2 = time.time()
  print('result = ', r.nonzeros())
  dt = (t2-t1)/nrep
  print('time = ', dt*1e3, ' ms')

  num_op = grad_det.n_nodes()
  print('number of elementary operations: ', num_op)
  print('time per elementary operations: ', dt/num_op*1e9, ' ns')