import numpy as np

# Reloading the module
import importlib

import OrthogonalPolynomials
importlib.reload(OrthogonalPolynomials)
from OrthogonalPolynomials import *

class Nodal2D():
    # A class that stores information of the nodes/collocation points
    # and related stuff
    def __init__(self, Nx, Ny):
        # The maximum index of nodes in the x direction
        # index = 0,1,...,N
        self.Nx = Nx
        # The maximum index of nodes in the y direction
        self.Ny = Ny
        
        # class variables created in other functions:
        # self.nodes_x, self.nodes_y:
            # The nodes in the x and y direction
            # Created in cal_nodes
            
        # self.w_bary_x, self.w_bary_y:
            # The Barycentric weights corresponding to self.nodes_x, self.nodes_y
            # Created in cal_BarycentricWeights
        # self.Dx, self.Dy:
            # The differentiating matrix wrt x and y
            # Created in cal_DiffMatrix
        
        
    
    
    def cal_nodes(self, node_type_x, node_type_y):
        # Generating the nodes along x and y directions
        # node_type_x, node_type_y: types of the nodes,
        # including: Legendre, Chebyshev
        
        if node_type_x=="Legendre":
            self.nodes_x = LegendreGaussLobattoNodes(self.Nx)
        elif node_type_x=="Chebyshev":
            self.nodes_x = ChebyshevGaussLobattoNodes_Reversed(self.Nx)
        else:
            print("Unknown node type")
            
        if node_type_y=="Legendre":
            self.nodes_y = LegendreGaussLobattoNodes(self.Ny)
        elif node_type_y=="Chebyshev":
            self.nodes_y = ChebyshevGaussLobattoNodes_Reversed(self.Ny)
        else:
            print("Unknown node type")
            
    def cal_BarycentricWeights(self):
        # Computing the Barycentric weights for the x and y nodes
        self.w_bary_x = BarycentricWeights(self.nodes_x)
        self.w_bary_y = BarycentricWeights(self.nodes_y)
            
            
    def cal_DiffMatrix(self):
        # Computing the differentiation matrix for the x and y directions
        self.Dx = PolynomialDiffMatrix(self.nodes_x, self.w_bary_x)
        self.Dy = PolynomialDiffMatrix(self.nodes_y, self.w_bary_y)
            
    
        
        
        
        
        
        
        