import numpy as np

def QuadMap(corners, xi, eta):
    # A map from [-1,1]*[-1,1] to a quadrilateral
    # corners=(x1, x2, x3, x4): the coordinates of the four corners of the quadrilateral
    # xi,eta: coordinate in the computational domain
    # xi \in [-1,1], eta \in [-1,1]
    # (-1,-1) -> x1
    # (1,-1) -> x2
    # (1,1) -> x3
    # (-1,1) -> x4
    
    x1, x2, x3, x4 = corners
    
    result = x1 * (1-xi) * (1-eta) + x2 * (1+xi) * (1-eta) + x3 * (1+xi) * (1+eta) + x4 * (1-xi) * (1+eta)
    return result / 4

def QuadMap_Curve24(BC2, BC4, xi, eta):
    # Mapping [-1,1]*[-1,1] to a quadrilateral with edge 2 and 4 curved
    # BC2, BC4: parameterized equations for edge2 and 4.
    # BC2(t), BC4(t): t in [-1,1]
    # xi,eta in [-1,1], coordinates in the computational domain
    
    # Note that BC2 BC4 return numpy arrays of length 2
    # So the result is also a numpy array of length 2
    return (1-xi)/2 * BC4(eta) + (1+xi)/2 * BC2(eta)
   

def QuadMap_Curve(BC1, BC2, BC3, BC4, xi, eta):
    # Mapping [-1,1]*[-1,1] to a quadrilateral
    # All edges may be curved.
    # BC1, BC2, BC3, BC4: parameterized equations for the four edges
    # BC1: x1->x2 as t:-1->1
    # BC2: x2->x3 as t:-1->1
    # BC3: x4->x3 as t:-1->1
    # BC4: x1->x4 as t:-1->1
    # xi,eta in [-1,1], coordinates in the computational domain
    
    # Note that BC1,..., BC4 return numpy arrays of length 2
    # So the result is also a numpy array of length 2
    result  = (1-xi)/2 * BC4(eta) + (1+xi)/2 * BC2(eta)
    result += (1-eta)/2 * BC1(xi) + (1+eta)/2 * BC3(xi)
    result -= (1-xi)/2 * ((1-eta)/2 * BC1(-1) + (1+eta)/2 * BC3(-1))
    result -= (1+xi)/2 * ((1-eta)/2 * BC1(1) + (1+eta)/2 * BC3(1))
    
    return result

