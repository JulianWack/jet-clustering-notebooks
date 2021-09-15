import numpy as np
import mpmath

mpmath.mp.dps = 100

np_cos = np.frompyfunc(mpmath.cos, 1, 1)
np_sin = np.frompyfunc(mpmath.sin, 1, 1)
np_cosh = np.frompyfunc(mpmath.cosh, 1, 1)
np_sinh = np.frompyfunc(mpmath.sinh, 1, 1)
np_arctan2 = np.frompyfunc(mpmath.atan2, 2, 1)
np_log = np.frompyfunc(mpmath.log, 1, 1)


def CosTheta(x, y, z):
    '''Find angle theta between z axis and cartesian 3 vector'''
    mag = np.sqrt(x**2+y**2+z**2)
    if mag == 0.0:
        costheta = 1
    else:
        costheta = z/mag
        
    return costheta


def get_pt(x, y):
    '''Find pt as polar radial distance from z axis'''
    return np.sqrt(x**2+y**2)


def get_eta(x, y, z):
    '''Find pesudo rapidity from cartesian 3 vector'''
    cosTheta = CosTheta(x, y, z)
    if cosTheta**2 < 1:
        eta = -0.5*np.log((1.0-cosTheta)/(1.0+cosTheta))
    elif z == 0:
        eta = 0
    elif z > 0:
        eta = 10e10
    else:
        eta = -10e10
        
    return eta


def get_phi(x, y):
    '''Find polar angle phi from cartesian 4 vector'''
    if (x==0) and (y==0):
        phi = 0
    else:
        phi = np_arctan2(y,x)
        
    return phi


def get_m(x, y, z, t):
    '''Find invariant mass from cartesian 4 vector'''
    delta_s2 = t**2 - (x**2 + y**2 + z**2)
    if delta_s2 < 0:
        m = -np.sqrt(-delta_s2)
    else:
        m = np.sqrt(delta_s2)
        
    return m


def np_sum4vec(pt, eta, phi, mass):
    '''Takes in 4 1D np.arrays as componnents of len(pt)-many PtEtaPhiM 4 vectors and returns the sum of these 4 vectors
    as a np.array of length 4 with elements in order of: Pt, Eta, Phi, Mass.'''
    
    # make 4 vector based on ROOT TLorentzVector::SetPtEtaPhiM, which allows element wise sumation
    n_PF = len(pt)
    vec4 = np.ndarray((n_PF,4))
    
    vec4[:,0] = pt*np_cos(phi)
    vec4[:,1] = pt*np_sin(phi)
    vec4[:,2] = pt*np_sinh(eta)
    
    non_neg = np.sqrt(pt**2*np_cosh(eta)**2 + mass**2)
    neg = np.sqrt(np.maximum(pt**2*np_cosh(eta)**2 - mass**2, mpmath.mpf('0')))
    vec4[:,3] = np.where(mass>=0, non_neg, neg)

    vec4_summed = np.sum(vec4, axis=0)
    
    # extract pt, eta, phi, mass from cartesian 4 vector based on ROOT TLorentzVector::Pt, Eta, Phi, M
    sum_PtEtaPhiM = np.empty(4)
    sum_PtEtaPhiM[0] = get_pt(vec4_summed[0], vec4_summed[1])
    sum_PtEtaPhiM[1] = get_eta(vec4_summed[0], vec4_summed[1], vec4_summed[2])
    sum_PtEtaPhiM[2] = get_phi(vec4_summed[0], vec4_summed[1])
    sum_PtEtaPhiM[3] = get_m(vec4_summed[0], vec4_summed[1], vec4_summed[2], vec4_summed[3])
    
    
    return sum_PtEtaPhiM
