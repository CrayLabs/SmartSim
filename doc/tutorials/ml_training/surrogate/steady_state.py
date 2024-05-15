#! /usr/bin/env python3
#
def boundary ( nx, ny, x, y, A, rhs ):

#*****************************************************************************80
#
## boundary() sets up the matrix and right hand side at boundary nodes.
#
#  Discussion:
#
#    For this simple problem, the boundary conditions specify that the solution
#    is 0 on the left side, 0 on the right side, and 0 on the top and bottom.
#
#    Nodes are assigned a single index K, which increases as:
#
#    (NY-1)*NX+1  (NY-1)*NX+2  ...  NY * NX
#           ....         ....  ...    .....
#           NX+1         NX+2  ...   2 * NX
#              1            2  ...       NX
#
#    The index K of a node on the lower boundary satisfies:
#      1 <= K <= NX
#    The index K of a node on the upper boundary satisfies:
#      (NY-1)*NX+1 <= K <= NY * NX
#    The index K of a node on the left boundary satisfies:
#      mod ( K, NX ) = 1
#    The index K of a node on the right boundary satisfies:
#      mod ( K, NX ) = 0
#
#    If we number rows from bottom I = 1 to top I = NY
#    and columns from left J = 1 to right J = NX, then the relationship
#    between the single index K and the row and column indices I and J is:
#      K = ( I - 1 ) * NX + J
#    and
#      J = 1 + mod ( K - 1, NX )
#      I = 1 + ( K - J ) / NX
#      
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    12 March 2017
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer NX, NY, the number of grid points in X and Y.
#
#    real X(NX), Y(NY), the coordinates of grid lines.
#
#    real sparse A(N,N), the system matrix, with the entries for the
#    interior nodes filled in.
#
#    real RHS(N), the system right hand side, with the entries for the
#    interior nodes filled in.
#
#  Output:
#
#    real sparse A(N,N), the system matrix, with the entries for all 
#    nodes filled in.
#
#    real RHS(N), the system right hand side, with the entries for 
#    all nodes filled in.
#

#
#  Left boundary.
#
  j = 0
  for i in range ( 0, ny ):
    kc = i * nx + j
    xc = x[j]
    yc = y[i]
    A[kc,kc] = 1.0
    rhs[kc] = 0.0
#
#  Right boundary.
#
  j = nx - 1
  for i in range ( 0, ny ):
    kc = i * nx + j
    xc = x[j]
    yc = y[i]
    A[kc,kc] = 1.0
    rhs[kc] = 0.0
#
#  Lower boundary.
#
  i = 0
  for j in range ( 0, nx ):
    kc = i * nx + j
    xc = x[j]
    yc = y[i]
    A[kc,kc] = 1.0
    rhs[kc] = 0.0
#
#  Upper boundary.
#
  i = ny - 1
  for j in range ( 0, nx ):
    kc = i * nx + j
    xc = x[j]
    yc = y[i]
    A[kc,kc] = 1.0
    rhs[kc] = 0.0

  return A, rhs

def fd2d_heat_steady ( nx, ny, x, y, d, f, source_centers):

#*****************************************************************************80
#
## fd2d_heat_steady() solves the steady 2D heat equation.
#
#  Discussion:
#
#    Nodes are assigned a singled index K, which increases as:
#
#    (NY-1)*NX+1  (NY-1)*NX+2  ...  NY * NX
#           ....         ....  ...    .....
#           NX+1         NX+2  ...   2 * NX
#              1            2  ...       NX
#
#    Therefore, the neighbors of an interior node numbered C are
#
#             C+NY
#              |
#      C-1 --- C --- C+1
#              |
#             C-NY
#
#    Nodes on the lower boundary satisfy:
#      1 <= K <= NX
#    Nodes on the upper boundary satisfy:
#      (NY-1)*NX+1 <= K <= NY * NX
#    Nodes on the left boundary satisfy:
#      mod ( K, NX ) = 1
#    Nodes on the right boundary satisfy:
#      mod ( K, NX ) = 0
#
#    If we number rows from bottom I = 1 to top I = NY
#    and columns from left J = 1 to right J = NX, we have
#      K = ( I - 1 ) * NX + J
#    and
#      J = 1 + mod ( K - 1, NX )
#      I = 1 + ( K - J ) / NX
#      
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    12 March 2017
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer NX, NY, the number of grid points in X and Y.
#
#    real X(NX), Y(NY), the coordinates of grid lines.
#
#    function D(X,Y), evaluates the thermal conductivity.
#
#    function F(X,Y), evaluates the heat source term.
#
#  Output:
#
#    real U(NX,NY), the approximation to the solution at the grid points.
#
  import numpy as np
#
#  Set the total number of unknowns.
#
  n = nx * ny
#
#  Allocate the matrix and right hand side.
#
  A = np.zeros ( [ n, n ] ).astype(np.single)
  rhs = np.zeros ( n ).astype(np.single)
#
#  Define the matrix at interior points.
#
  A, rhs = interior ( nx, ny, x, y, d, f, A, rhs, source_centers )
#
#  Handle boundary conditions.
#
  A, rhs = boundary ( nx, ny, x, y, A, rhs )


  u = np.linalg.solve ( A, rhs )

  u.shape = ( ny, nx )

  return u

def fd2d_heat_steady_test ( ):

#*****************************************************************************80
#
## fd2d_heat_steady_test() tests fd2d_heat_steady().
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    27 August 2013
#
#  Author:
#
#    John Burkardt
#
  print ( '' )
  print ( 'fd2d_heat_steady_test:' )
  print ( '  Python version' )
  print ( '  Test fd2d_heat_steady().' )

  fd2d_heat_steady_test01 ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'fd2d_heat_steady_test:' )
  print ( '  Normal end of execution.' )
  return

def fd2d_heat_steady_test01 (nx, ny ):

#*****************************************************************************80
#
## fd2d_heat_steady_test01() demonstrates the use of fd2d_heat_steady.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    12 March 2017
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt
  from matplotlib import cm
#
#  Specify the spatial grid.
#
  # nx = 64
  # ny = 64
  xvec = np.linspace ( 0.0, 1.0, nx )
  yvec = np.linspace ( 0.0, 1.0, ny )

  source_centers = 0.2+np.random.rand(np.random.randint(1,6),2)*0.6

  Xgrid, Ygrid = np.meshgrid(xvec, yvec)
  u_init = np.zeros_like(Xgrid).astype(bool)
  for center in source_centers:
    u_init |= (Xgrid-center[0])**2 + (Ygrid-center[1])**2 < 0.05**2

  u_init = u_init.astype(np.single)
#
#  Solve the finite difference approximation to the steady 2D heat equation.
#
  umat = fd2d_heat_steady ( nx, ny, xvec, yvec, d, f, source_centers)
#
#  Plotting.
#
  PLOT = False
  if PLOT:
    xmat, ymat = np.meshgrid ( xvec, yvec )

    fig = plt.figure()
    ax = fig.add_subplot ( 111, projection = '3d' )
    ax.plot_surface ( xmat, ymat, umat, cmap = cm.coolwarm,
                        linewidth = 0, antialiased = False )
    ax.set_xlabel ( '<--- Y --->' )
    ax.set_ylabel ( '<--- X --->' )
    ax.set_zlabel ( '<---U(X,Y)--->' )
    ax.set_title ( 'Solution of steady heat equation' )
    plt.draw ( )
    filename = 'fd2d_heat_steady_test01.png'
    fig.savefig ( filename )
    plt.show ( block = False )
    plt.close ( )

    print ( '' )
    print ( '  Plotfile saved as "%s".' % ( filename ) )

  return u_init, umat

def d ( x, y ):

#*****************************************************************************80
#
## d() evaluates the heat conductivity coefficient.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    23 July 2013
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real X, Y, the evaluation point.
#
#  Output:
#
#    real VALUE, the value of the heat conductivity at (X,Y).
#
  value = 1.0

  return value

def f ( x, y , centers):

#*****************************************************************************80
#
## f() evaluates the heat source term. This function was modified to
#  include circles of radiust 0.05, where the temperature is 1
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    23 July 2013
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real X, Y, the evaluation point.
#    real centers, the centers of the heat sources
#
#  Output:
#
#    real VALUE, the value of the heat source term at (X,Y).
#
  source = False
  for center in centers:
    source |= ((x-center[0])**2 + (y-center[1])**2 < 0.05**2)
  return float(source)

def interior ( nx, ny, x, y, d, f, A, rhs, source_centers ):

#*****************************************************************************80
#
## interior() sets up the matrix and right hand side at interior nodes.
#
#  Discussion:
#
#    Nodes are assigned a single index K, which increases as:
#
#    (NY-1)*NX+1  (NY-1)*NX+2  ...  NY * NX
#           ....         ....  ...    .....
#           NX+1         NX+2  ...   2 * NX
#              1            2  ...       NX
#
#    Therefore, the neighbors of an interior node numbered C are
#
#             C+NY
#              |
#      C-1 --- C --- C+1
#              |
#             C-NY
#
#    If we number rows from bottom I = 1 to top I = NY
#    and columns from left J = 1 to right J = NX, then the relationship
#    between the single index K and the row and column indices I and J is:
#      K = ( I - 1 ) * NX + J
#    and
#      J = 1 + mod ( K - 1, NX )
#      I = 1 + ( K - J ) / NX
#      
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    12 March 2017
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer NX, NY, the number of grid points in X and Y.
#
#    real X(NX), Y(NY), the coordinates of grid lines.
#
#    function pointer @D(X,Y), evaluates the thermal conductivity.
#
#    function pointer @F(X,Y), evaluates the heat source term.
#
#    real sparse A(N,N), the system matrix, without any entries set.
#
#    real RHS(N), the system right hand side, without any entries set.
#
#  Output:
#
#    real sparse A(N,N), the system matrix, with the entries for the
#    interior nodes filled in.
#
#    real RHS(N), the system right hand side, with the entries for the
#    interior nodes filled in.
#
  import numpy as np
#
#  For now, assume X and Y are equally spaced.
#
  dx = x[1] - x[0]
  dy = y[1] - y[0]

  for ic in range ( 1, ny - 1 ):
    for jc in range ( 1, nx - 1 ):

      ino = ic + 1
      iso = ic - 1
      je = jc + 1
      jw = jc - 1

      kc = ic * nx + jc
      ke = kc + 1
      kw = kc - 1
      kn = kc + nx
      ks = kc - nx

      dce = d ( 0.5 * ( x[jc] + x[je] ),         y[ic] )
      dcw = d ( 0.5 * ( x[jc] + x[jw] ),         y[ic] )
      dcn = d (         x[jc],           0.5 * ( y[ic] + y[ino] ) )
      dcs = d (         x[jc],           0.5 * ( y[ic] + y[iso] ) )


      rhs[kc] = f ( x[jc], y[ic], source_centers )

      if rhs[kc] != 0.0:
          A[kc,kc] = 1.0
      else:
        A[kc,kc] = ( dce + dcw ) / dx / dx + ( dcn + dcs ) / dy / dy
        A[kc,ke] = - dce         / dx / dx
        A[kc,kw] =       - dcw   / dx / dx
        A[kc,kn] =                           - dcn         / dy / dy
        A[kc,ks] =                                 - dcs   / dy / dy

      

  return A, rhs

def timestamp ( ):

#*****************************************************************************80
#
## timestamp() prints the date as a timestamp.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 April 2013
#
#  Author:
#
#    John Burkardt
#
  import time

  t = time.time ( )
  print ( time.ctime ( t ) )

  return None

if ( __name__ == '__main__' ):
  timestamp ( )
  fd2d_heat_steady_test01 (100, 100 )
  timestamp ( )
 
