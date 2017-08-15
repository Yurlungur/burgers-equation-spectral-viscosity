#!/usr/bin/env python2

"""
orthopoly.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-08-14 16:05:55 (jmiller)>

A module for orthogonal polynomials for pseudospectral methods in Python
"""



# ======================================================================
# imports
# ======================================================================
import numpy as np
from numpy import polynomial
from numpy import linalg
from scipy import integrate
from scipy import optimize
from copy import copy
# ======================================================================



# ======================================================================
# Global constants
# ======================================================================
LOCAL_XMIN = -1. # Maximum and min values of reference cell
LOCAL_XMAX = 1.
LOCAL_WIDTH = float(LOCAL_XMAX-LOCAL_XMIN)
poly = polynomial.chebyshev.Chebyshev  # A class for orthogonal polynomials
pval2d = polynomial.chebyshev.chebval2d
weight_func = polynomial.chebyshev.chebweight
integrator = integrate.quad
# ======================================================================


# ======================================================================
# Utilities
# ======================================================================
def get_norm2_difference(foo,bar,xmin,xmax):
    """
    Returns sqrt(integral((foo-bar)**2)) on the interval [xmin,xmax]
    """
    out = integrator(lambda x: (foo(x)-bar(x))**2,xmin,xmax)[0]
    out /= float(xmax-xmin)
    out = np.sqrt(out)
    return out
# ======================================================================


# ======================================================================
# Nodal and Modal Details
# ======================================================================
def continuous_inner_product(foo,bar):
    """"
    Takes the continuous inner product in the POLY norm between
    the functions foo and bar.
    """
    return integrator(lambda x: foo(x)*bar(x)*weight_func(x),
                      LOCAL_XMIN,LOCAL_XMAX)[0]

def get_quadrature_points(order):
    """
    Returns the quadrature points for Gauss-Lobatto quadrature
    as a function of the order of the polynomial we want to
    represent.
    See: https://en.wikipedia.org/wiki/Gaussian_quadrature
    """
    return np.sort(np.concatenate((np.array([-1,1]),
                                   poly.basis(order).deriv().roots())))

def get_vandermonde_matrices(order,nodes=None):
    """
    Returns the Vandermonde fast-Fourier transform matrices s2c and c2s,
    which convert spectral coefficients to configuration space coefficients
    and vice-versa respectively. Requires the order of the element/method
    as input.
    """
    if nodes is None:
        nodes = get_quadrature_points(order)
    s2c = np.zeros((order+1,order+1),dtype=float)
    for i in range(order+1):
        for j in range(order+1):
            s2c[i,j] = poly.basis(j)(nodes[i])
    c2s = linalg.inv(s2c)
    return s2c,c2s

def get_integration_weights(order,nodes=None):
    """
    Returns the integration weights for Gauss-Lobatto quadrature
    as a function of the order of the polynomial we want to
    represent.
    See: https://en.wikipedia.org/wiki/Gaussian_quadrature
    See: arXive:gr-qc/0609020v1
    """
    if np.all(nodes == False):
        nodes=get_quadrature_points(order)
    if poly == polynomial.chebyshev.Chebyshev:
        weights = np.empty((order+1))
        weights[1:-1] = np.pi/order
        weights[0] = np.pi/(2*order)
        weights[-1] = weights[0]
        return weights
    elif poly == polynomial.legendre.Legendre:
        interior_weights = 2/((order+1)*order*poly.basis(order)(nodes[1:-1])**2)
        boundary_weights = np.array([1-0.5*np.sum(interior_weights)])
        weights = np.concatenate((boundary_weights,
                                  interior_weights,
                                  boundary_weights))
        return weights
    else:
        raise ValueError("Not a known polynomial type.")
        return False

def get_modal_differentiation_matrix(order):
    """
    Returns the differentiation matrix for the first derivative in the
    modal basis.
    """
    out = np.zeros((order+1,order+1))
    for i in range(order+1):
        out[:i,i] = poly.basis(i).deriv().coef
    return out

def get_nodal_differentiation_matrix(order,
                                     s2c=None,c2s=None,
                                     Dmodal=None):
    """
    Returns the differentiation matrix for the first derivative
    in the nodal basis

    It goes without saying that this differentiation matrix is for the
    reference cell.
    """
    if Dmodal is None:
        Dmodal = get_modal_differentiation_matrix(order)
    if s2c is None or c2s is None:
        s2c,c2s = get_vandermonde_matrices(order)
    return np.dot(s2c,np.dot(Dmodal,c2s))
# ======================================================================



# Operators Outside Reference Cell
# ======================================================================
def get_width(xmin=LOCAL_XMIN,xmax=LOCAL_XMAX):
    "Gets the width of the interval [xmin,xmax]"
    return float(xmax-xmin)

def get_scale_factor(xmin=LOCAL_XMIN,xmax=LOCAL_XMAX):
    "Gets the scale factor for the derivative operator"
    return (xmax-float(xmin))/(LOCAL_XMAX-float(LOCAL_XMIN))

def coord_reference_to_global(x_local,
                              xmin=LOCAL_XMIN,
                              xmax=LOCAL_XMAX):
    "maps a point in [LOCAL_XMIN,LOCAL_XMAX] to a point in [xmin,xmax]"
    global_width=get_width(xmin,xmax)
    m = global_width/LOCAL_WIDTH
    b = (LOCAL_XMAX*xmin - LOCAL_XMIN*xmax)/LOCAL_WIDTH
    x_global = m*x_local + b
    return x_global

def coord_global_to_reference(x_global,
                              xmin=LOCAL_XMIN,
                              xmax=LOCAL_XMAX):
    "maps a point in [xmin,xmax] to a point in [LOCAL_XMIN,LOCAL_XMAX]"
    global_width=get_width(xmin,xmax)
    m = LOCAL_WIDTH/global_width
    b = (LOCAL_XMIN*xmax - LOCAL_XMAX*xmin)/global_width
    x_local = m*x_global + b
    return x_local

def get_colocation_points(order,xmin=LOCAL_XMIN,xmax=LOCAL_XMAX,quad_points=None):
    """
    Generates order+1 colocation points on the domain [xmin,xmax]
    """
    if quad_points is None:
        quad_points = get_quadrature_points(order)
    x = coord_reference_to_global(quad_points,xmin,xmax)
    return x

def get_global_differentiation_matrix(order,
                                      xmin=LOCAL_XMIN,
                                      xmax=LOCAL_XMAX,
                                      s2c=None,
                                      c2s=None,
                                      Dmodal=None):
    """
    Returns the differentiation matrix in the nodal basis
    for the global coordinates (outside the reference cell)

    Takes the Jacobian into effect.
    """
    scale_factor = get_scale_factor(xmin,xmax)
    LD = get_nodal_differentiation_matrix(order,s2c,c2s,Dmodal)
    PD = LD/scale_factor
    return PD

def get_rhs_filter_modal(order,s,eta_crit):
    """
    Get's the right-hand-side filter operator in modal form
    """
    #two_s = order+2 if order % 2 == 0 else order+1
    #s = two_s/2
    two_s = 2*s
    p = order
    #Fdiag = np.array([((float(i)/p - eta_crit)/(1-eta_crit))**(two_s)\
    #                  for i in range(p+1)])
    Fdiag = np.array([(float(i)/p)**(2*s) for i in range(p+1)])
    #Fdiag[:(p*eta_crit)] = 0
    F = np.diag(Fdiag)
    return F

def get_rhs_filter_nodal(order,s,eta_crit,
                         xmin,xmax,s2c,c2s):
    """
    Gets the right-hand-side filter operator in nodal form
    """
    p = order
    h = get_width(xmin,xmax)
    epsilon = p/h
    Fmodal = get_rhs_filter_modal(order,s,eta_crit)
    Fnodal = np.dot(s2c,np.dot(Fmodal,c2s))
    Fnodal *= epsilon
    return Fnodal


def get_tadmor_Q(order,s,s2c,c2s):
    epsilon_N = order**(1-2*s)
    theta = 0.5
    m_N = order**theta
    Qmodal = np.zeros(order+1)
    for k in range(Qmodal.shape[0]):
        if k>= m_N:
            Qmodal[k] = 1. - (m_N/k)**((2*s-1)/theta)
    Qmodal = np.diag(Qmodal)
    Qnodal = np.dot(s2c,np.dot(Qmodal,c2s))
    Qnodal *= -1.0*epsilon_N
    return Qnodal
# ======================================================================



# ======================================================================
# Reconstruct Global Solution
# ======================================================================
def get_continuous_object(grid_func,
                          xmin=LOCAL_XMIN,xmax=LOCAL_XMAX,
                          c2s=None):
    """
    Maps the grid function grid_func, which is any field defined
    on the colocation points to a continuous function that can
    be evaluated.

    Parameters
    ----------
    xmin -- the minimum value of the domain
    xmax -- the maximum value of the domain
    c2s  -- The Vandermonde matrix that maps the colocation representation
            to the spectral representation

    Returns
    -------
    An numpy polynomial object which can be called to be evaluated
    """
    order = len(grid_func)-1
    if c2s == None:
        s2c,c2s = get_vandermonde_matrices(order)
    Spec_func = np.dot(c2s,grid_func)
    my_interp = poly(spec_func,domain=[xmin,xmax])
    return my_interp
# ======================================================================



# ======================================================================
# A convenience class that generates everything and can be called
# ======================================================================
class PseudoSpectralDiscretization1D:
    """Given an order, and a domain [xmin,xmax]
    defines internally all structures and methods the user needs
    to calculate spectral derivatives in 1D
    """
    def __init__(self,order,xmin,xmax,eta_crit=0.5,viscous_C=1.):
        "Constructor. Needs the order of the method and the domain [xmin,xmax]."
        self.order = order
        self.xmin = xmin
        self.xmax = xmax
        self.viscous_C = viscous_C
        self.eta_crit = eta_crit
        self.quads = get_quadrature_points(self.order)
        self.weights = get_integration_weights(self.order,self.quads)
        self.s2c,self.c2s = get_vandermonde_matrices(self.order,self.quads)
        self.Dmodal = get_modal_differentiation_matrix(self.order)
        self.Dnodal = get_nodal_differentiation_matrix(self.order,
                                                       self.s2c,self.c2s,
                                                       self.Dmodal)
        self.colocation_points = get_colocation_points(self.order,
                                                       self.xmin,self.xmax,
                                                       self.quads)
        self.PD = get_global_differentiation_matrix(self.order,
                                                    self.xmin,self.xmax,
                                                    self.s2c,self.c2s,
                                                    self.Dmodal)

        self.s = np.log(order)
        self.Fmodal = get_rhs_filter_modal(self.order,self.s,
                                           self.eta_crit)
        self.Frhs = get_rhs_filter_nodal(self.order,
                                         self.s,
                                         self.eta_crit,
                                         self.xmin,
                                         self.xmax,
                                         self.s2c,
                                         self.c2s)
        self.Qnodal = get_tadmor_Q(self.order,self.s,
                                   self.s2c,self.c2s)

    def get_scale_factor(self):
        "Jacobian between local and global"
        return (self.xmax-float(self.xmin))/(LOCAL_XMAX-float(LOCAL_XMIN))

    def get_x(self):
        """
        Returns the colocation points
        """
        return self.colocation_points

    def differentiate(self,grid_func,order=1):
        """
        Given a grid function defined on the colocation points,
        returns its derivative of the appropriate order
        """
        assert type(order) == int
        assert order >= 0
        if order == 0:
            return grid_func
        else:
            return self.differentiate(np.dot(self.PD,grid_func),order-1)

    def rhs_filter(self,grid_func):
        "Given a grid function, apply right-hand-side filter operator"
        return -self.viscous_C*np.dot(self.Frhs,grid_func)

    def spectral_viscosity(self,grid_func):
        du = self.differentiate(grid_func)
        Qdu = np.dot(self.Qnodal,du)
        dQdu = self.differentiate(Qdu)
        return dQdu

    def to_continuum(self,grid_func):
        """
        Given a grid function defined on the colocation points, returns a
        numpy polynomial object that can be evaluated.
        """
        return get_continuous_object(grid_func,self.xmin,self.xmax,self.c2s)

    def _coord_ref_to_global(self,r):
        """Maps a coordinate in the reference cell to a coordinate in
        global coordinates.
        """
        return coord_reference_to_global(r,self.xmin,self.xmax)

    def _coord_global_to_ref(self,x):
        """Maps a coordinate in global coordinates to
        one in the reference cell.
        """
        return coord_global_to_reference(x,self.xmin,self.xmax)
# ======================================================================


# Warning not to run this program on the command line
if __name__ == "__main__":
    raise ImportError("Warning. This is a library. It contains no main function.")
