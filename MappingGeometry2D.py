import numpy as np

# Reloading the module
import importlib

import Nodal2D
importlib.reload(Nodal2D)
from Nodal2D import *

class MappingGeometry2D():
    def __init__(self, nodal2D, curves):
        # nodes: Nodal2D class
        self.nodal2D = nodal2D
        
        # curves: [curve1, curve2, curve3, curve4]
        # 4 boundary curves, of class Curve
        self.curve1, self.curve2, self.curve3, self.curve4 = curves
        
        # Corners:
        # Curve 1: corner1 -> corner2
        # Curve 2: corner2 -> corner3
        # Curve 3: corner4 -> corner3 
        # Curve 4: corner1 -> corner4
        self.corner1 = np.array([self.curve1.x_nodes[0], self.curve1.y_nodes[0]])
        self.corner2 = np.array([self.curve1.x_nodes[-1], self.curve1.y_nodes[-1]]) 
        self.corner3 = np.array([self.curve3.x_nodes[-1], self.curve3.y_nodes[-1]]) 
        self.corner4 = np.array([self.curve3.x_nodes[0], self.curve3.y_nodes[0]])
        
        # Computational nodes
        Nx = self.nodal2D.Nx
        Ny = self.nodal2D.Ny
        self.nodes_comp_x, self.nodes_comp_y = np.meshgrid(self.nodal2D.nodes_x,
                                                           self.nodal2D.nodes_y,
                                                           indexing='ij')
        
        
        #self.nodes_comp_x = np.zeros((Nx+1, Ny+1))
        #self.nodes_comp_y = np.zeros((Nx+1, Ny+1))
        #for i in range(Nx+1):
        #    for j in range(Ny+1):
        #        self.nodes_comp_x[i,j] = self.nodal2D.nodes_x[i]
        #        self.nodes_comp_y[i,j] = self.nodal2D.nodes_y[j]
        
        
        
        # Physical nodes (only initialization)
        self.nodes_phy_x = np.zeros((Nx+1, Ny+1))
        self.nodes_phy_y = np.zeros((Nx+1, Ny+1))
        
        # Metrics (derivatives) (only initialization)
        self.X_xi = np.zeros((Nx+1, Ny+1))
        self.X_eta = np.zeros((Nx+1, Ny+1))
        self.Y_xi = np.zeros((Nx+1, Ny+1))
        self.Y_eta = np.zeros((Nx+1, Ny+1))
        
        # Jacobian (only initialization)
        self.J = np.zeros((Nx+1, Ny+1))
        
        # Class variables created in other functions
        # self.scal_lower, self.norm_vect_lower
        # self.scal_upper, self.norm_vect_upper
        # self.scal_left, self.norm_vect_left
        # self.scal_right, self.norm_vect_right
            # The scaling factors and normal vectors on the four boundaries
            # Created in cal_normal_vector_nodes
        
        
    #########################################################
    # Functions for quadrilateral maps
    #########################################################
        
    def cal_QuadMap(self, xi, eta):
        # Finding the physical coordinnates for the map to a quadrilateral
        # at a single point (xi, eta)
        # The mapping functions are X(xi, eta) and Y(xi, eta)
        result = 1/4 * (self.corner1 * (1-xi) * (1-eta)
                      + self.corner2 * (1+xi) * (1-eta)
                      + self.corner3 * (1+xi) * (1+eta)
                      + self.corner4 * (1-xi) * (1+eta)
                       )
        return result
    
    def cal_QuadMap_nodes(self):
        # Computing the physical nodes for the map to quadrature
        x1, y1 = self.corner1
        x2, y2 = self.corner2
        x3, y3 = self.corner3
        x4, y4 = self.corner4
        
        # Notation in the textbook
        xi = self.nodes_comp_x
        eta = self.nodes_comp_y
        
        self.nodes_phy_x = 1/4 * ( x1 * (1-xi) * (1-eta)
                                 + x2 * (1+xi) * (1-eta)
                                 + x3 * (1+xi) * (1+eta)
                                 + x4 * (1-xi) * (1+eta) )
                                  
        
        
        self.nodes_phy_y = 1/4 * ( y1 * (1-xi) * (1-eta)
                                 + y2 * (1+xi) * (1-eta)
                                 + y3 * (1+xi) * (1+eta)
                                 + y4 * (1-xi) * (1+eta) )
                                  
        
        #for i in range(self.nodal2D.Nx + 1): #i=0,1,...,Nx
        #    for j in range(self.nodal2D.Ny + 1): #j=0,1,...,Ny
        #        xi = self.nodes_comp_x[i,j]
        #        eta = self.nodes_comp_y[i,j] 
        #        self.nodes_phy_x[i,j], self.nodes_phy_y[i,j] = self.cal_QuadMap(xi, eta)
        
    def cal_QuadMapDerivatives(self, xi, eta):
        # Finding the partial derivatives for the map to a quadrilateral
        # at a single point (xi, eta)
        # The mapping functions are X(xi, eta) and Y(xi, eta)
        
        X_xi, Y_xi = 1/4 * (1-eta) * (self.corner2 - self.corner1) \
                   + 1/4 * (1+eta) * (self.corner3 - self.corner4)
        X_eta, Y_eta = 1/4 * (1-xi) * (self.corner4 - self.corner1) \
                     + 1/4 * (1+xi) * (self.corner3 - self.corner2)
        
        return X_xi, Y_xi, X_eta, Y_eta
    
    def cal_QuadMapDerivatives_nodes(self):
        # Finding the partial derivatives for the map to a quadrilateral
        # at all nodes
        
        x1, y1 = self.corner1
        x2, y2 = self.corner2
        x3, y3 = self.corner3
        x4, y4 = self.corner4
        
        xi = self.nodes_comp_x
        eta = self.nodes_comp_y
        
        self.X_xi = 0.25 * (1-eta) * (x2 - x1) + 0.25 * (1+eta) * (x3 - x4)
        self.Y_xi = 0.25 * (1-eta) * (y2 - y1) + 0.25 * (1+eta) * (y3 - y4)
                   
        self.X_eta = 0.25 * (1-xi) * (x4 - x1) + 0.25 * (1+xi) * (x3 - x2)
        self.Y_eta = 0.25 * (1-xi) * (y4 - y1) + 0.25 * (1+xi) * (y3 - y2)    
        
        
        
        # Computing the derivatives for the quad map
        #for i in range(self.nodal2D.Nx + 1): #i=0,1,...,Nx
        #    for j in range(self.nodal2D.Ny + 1): #j=0,1,...,Ny
        #        xi = self.nodes_comp_x[i,j]
        #        eta = self.nodes_comp_y[i,j] 
        #        self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j] = \
        #                            self.cal_QuadMapDerivatives(xi, eta)
    
    
    #########################################################
    # Functions for general maps
    #########################################################
    
    
    def cal_Map(self, xi, eta):
        # Finding the physical coordinnates for the map 
        # at a single point (xi, eta)
        # The mapping functions are X(xi, eta) and Y(xi, eta)
        
        # Evaluate the curves at xi or eta
        curve1_xi = self.curve1.cal_coordinates(xi)
        curve3_xi = self.curve3.cal_coordinates(xi)
        curve2_eta = self.curve2.cal_coordinates(eta)
        curve4_eta = self.curve4.cal_coordinates(eta)
        
        result = 1/2 * ((1-eta) * curve1_xi
                      + (1+eta) * curve3_xi
                      + (1+xi) * curve2_eta
                      + (1-xi) * curve4_eta) \
               - 1/4 * (1-xi) * ((1-eta) * self.corner1 + (1+eta) * self.corner4 ) \
               - 1/4 * (1+xi) * ((1-eta) * self.corner2 + (1+eta) * self.corner3 )
        return result
    
        
    
    
    def cal_MapDerivatives(self, xi, eta):
        # # Finding the partial derivatives for a general map 
        # at a single point (xi, eta)
        # The mapping functions are X(xi, eta) and Y(xi, eta)
        
        # The partial derivatives of curve1 and 3 wrt xi
        p_curve1_p_xi = self.curve1.cal_derivatives(xi)
        p_curve3_p_xi = self.curve3.cal_derivatives(xi)
        # Evaluate curve2 and 4 at eta
        curve2_eta = self.curve2.cal_coordinates(eta)
        curve4_eta = self.curve4.cal_coordinates(eta)
        # Metrics wrt xi
        X_xi, Y_xi = 1/2 * (curve2_eta - curve4_eta 
                            + (1-eta)*p_curve1_p_xi 
                            + (1+eta)*p_curve3_p_xi) \
                    -1/4 * ((1-eta)*(self.corner2 - self.corner1) 
                          + (1+eta)*(self.corner3 - self.corner4))
        
        # The partial derivatives of curve2 and 4 wrt eta
        p_curve2_p_eta = self.curve2.cal_derivatives(eta)
        p_curve4_p_eta = self.curve4.cal_derivatives(eta)
        # Evaluate curve1 and 1 at xi
        curve1_xi = self.curve1.cal_coordinates(xi)
        curve3_xi = self.curve3.cal_coordinates(xi)
        # Metrics wrt eta
        X_eta, Y_eta = 1/2 * ((1-xi)*p_curve4_p_eta + (1+xi)*p_curve2_p_eta
                             - curve1_xi + curve3_xi) \
                     - 1/4 * ((1-xi)*(self.corner4-self.corner1)
                             +(1+xi)*(self.corner3-self.corner2))
        
        return X_xi, Y_xi, X_eta, Y_eta
    
    
        
    def cal_Map_nodes(self):
        # Computing the physical nodes for general map
        for i in range(self.nodal2D.Nx + 1): #i=0,1,...,Nx
            for j in range(self.nodal2D.Ny + 1): #j=0,1,...,Ny
                xi = self.nodes_comp_x[i,j]
                eta = self.nodes_comp_y[i,j] 
                self.nodes_phy_x[i,j], self.nodes_phy_y[i,j] = self.cal_Map(xi, eta)
        
        
        
    
        
         
        
        
    
    def cal_MapDerivatives_nodes(self):
        # Computing the derivatives for general map
        for i in range(self.nodal2D.Nx + 1): #i=0,1,...,Nx
            for j in range(self.nodal2D.Ny + 1): #j=0,1,...,Ny
                xi = self.nodes_comp_x[i,j]
                eta = self.nodes_comp_y[i,j] 
                self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j] = \
                                    self.cal_MapDerivatives(xi, eta)
                
                
    def cal_Jacobian(self):
        # Computing the Jabobian at nodes
        self.J = self.X_xi * self.Y_eta - self.X_eta * self.Y_xi
        
        #for i in range(self.nodal2D.Nx + 1): #i=0,1,...,Nx
        #    for j in range(self.nodal2D.Ny + 1): #j=0,1,...,Ny
        #        xi = self.nodes_comp_x[i,j]
        #        eta = self.nodes_comp_y[i,j] 
        #        self.J[i,j] = self.X_xi[i,j] * self.Y_eta[i,j] \
        #                    - self.X_eta[i,j] * self.Y_xi[i,j]
                
                
    def cal_normal_vector_nodes(self):
        # Computing the normal vectors on the boundary
        
        # Cartesian basis
        ex = np.array([1,0])
        ey = np.array([0,1])
        
        # 'Lower boundary'
        j = 0
        self.norm_vect_lower = np.zeros((self.nodal2D.Nx + 1, 2))
        self.scal_lower = np.zeros(self.nodal2D.Nx + 1)
        for i in range(self.nodal2D.Nx + 1): #i=0,1,...,Nx
            # Use simplified variable names
            #x = self.nodes_phy_x[i,j]
            #y = self.nodes_phy_y[i,j]
            X_xi, Y_xi, X_eta, Y_eta = self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j]
            sign_J = np.sign(self.J[i,j])
            # The scaling factor
            self.scal_lower[i] = np.sqrt(X_xi**2 + Y_xi**2)
            # The normal vector is outward, so a negative sign is present.
            self.norm_vect_lower[i] = -sign_J / self.scal_lower[i] * (X_xi * ey - Y_xi * ex)
            
        # 'Upper boundary'
        j = self.nodal2D.Ny
        self.norm_vect_upper = np.zeros((self.nodal2D.Nx + 1, 2))
        self.scal_upper = np.zeros(self.nodal2D.Nx + 1)
        for i in range(self.nodal2D.Nx + 1): #i=0,1,...,Nx
            # Use simplified variable names
            #x = self.nodes_phy_x[i,j]
            #y = self.nodes_phy_y[i,j]
            X_xi, Y_xi, X_eta, Y_eta = self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j]
            sign_J = np.sign(self.J[i,j])
            # The scaling factor
            self.scal_upper[i] = np.sqrt(X_xi**2 + Y_xi**2)
            # The normal vector 
            self.norm_vect_upper[i] = sign_J / self.scal_upper[i] * (X_xi * ey - Y_xi * ex)
            
            
        # 'Left boundary'
        i = 0
        self.norm_vect_left = np.zeros((self.nodal2D.Ny + 1, 2))
        self.scal_left = np.zeros(self.nodal2D.Ny + 1)
        for j in range(self.nodal2D.Ny + 1): #j=0,1,...,Ny
            # Use simplified variable names
            #x = self.nodes_phy_x[i,j]
            #y = self.nodes_phy_y[i,j]
            X_xi, Y_xi, X_eta, Y_eta = self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j]
            sign_J = np.sign(self.J[i,j])
            # The scaling factor
            self.scal_left[j] = np.sqrt(X_eta**2 + Y_eta**2)
            # The normal vector is outward, so a negative sign is present.
            self.norm_vect_left[j] = -sign_J / self.scal_left[j] * (Y_eta * ex - X_eta * ey)
            
        # 'Right boundary'
        i = self.nodal2D.Nx
        self.norm_vect_right = np.zeros((self.nodal2D.Ny + 1, 2))
        self.scal_right = np.zeros(self.nodal2D.Ny + 1)
        for j in range(self.nodal2D.Ny + 1): #j=0,1,...,Ny
            # Use simplified variable names
            #x = self.nodes_phy_x[i,j]
            #y = self.nodes_phy_y[i,j]
            X_xi, Y_xi, X_eta, Y_eta = self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j]
            sign_J = np.sign(self.J[i,j])
            # The scaling factor
            self.scal_right[j] = np.sqrt(X_eta**2 + Y_eta**2)
            # The normal vector
            self.norm_vect_right[j] = sign_J / self.scal_right[j] * (Y_eta * ex - X_eta * ey)
            
            
        
        
        