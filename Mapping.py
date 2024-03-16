import numpy as np

class Mapping2D():
    # 2D mapping class
    
    def __init__(self, Curves, Nodes2D):
        self.Gamma1, self.Gamma2, self.Gamma3, self.Gamma4 = Curves
        
        