import numpy as np 

class medium():

    def __init__(self):

        
        # refractive indexes

        self.NumberPhotons = 10000
        '''
        # [depth, refractive_index(n), u_a, u_s, g]
        layer_null = [-999.9, 1, 1, 1, 0]
        layer0 = [float(0), 1, 1, 1, 0]
        layer1 = [0.5, 1.37, 0.1, 100, 0.9]
        layer2 = [1, 1.37, 0.1, 100, 0.9]
        layer3 = [999.9, 1, 1, 1, 0]

        self.layers = {-1:layer_null,
                       0:layer0,
                       1:layer1,
                       2:layer2,
                       3:layer3}
        
        self.layers_important = {0:layer1,
                                 1:layer2}
        self.depth = 1
        '''
        # [depth, refractive_index(n), u_a, u_s, g]
        layer_null = [-999.9, 1, 1, 1, 0]
        layer0 = [float(0), 1, 1, 1, 0]         # Air
        layer1 = [0.001, 1.45, 0.7405, 176.125, 0.8]          # Stratum Corneum
        layer2 = [0.009, 1.4, 1.3, 176.125, 0.8]              # Epidermis 
        layer3 = [0.019, 1.4, 1.05, 106.25, 0.8]               # Papliary Dermis
        layer4 = [0.027, 1.39, 1.427, 145.625, 0.818]         # Upper Blood Plexus
        layer5 = [0.177, 1.4, 1.05, 1106.25, 0.8]              # Reticular Dermis
        layer6 = [0.184, 1.34, 4.443, 460.625, 0.962]         # Deep Blood Plexus
        layer7 = [0.200, 1.4, 1.05, 106.25, 0.8]              # Lower Dermis
        layer8 = [0.500, 1.46, 0.975, 97.125, 0.8]             # Subcutaneous Fat
        layer9 = [999.9, 1.37, 1, 1, 0]                      # Muscle
        
        self.layers = self.layers = {-1:layer_null,
                       0:layer0,
                       1:layer1,
                       2:layer2,
                       3:layer3,
                       4:layer4,
                       5:layer5,
                       6:layer6,
                       7:layer7,
                       8:layer8,
                       9:layer9}
     
        
        self.layers_important = {0:layer1,
                       1:layer2,
                       2:layer3,
                       3:layer4,
                       4:layer5,
                       5:layer6,
                       6:layer7,
                       7:layer8}
        

        self.depth = self.layers_important[7][0]

       
    def inclusion(self, size):

        '''
        Defines a square inclusion based on the layers defined in __init__
        returns the 6 faces of the cube as '''

        planes = [
        {'normal': np.array([1, 0, 0]), 'point': np.array([-size/2, 0, 0]), 'face':'left'},  # left face
        {'normal': np.array([-1, 0, 0]), 'point': np.array([size/2, 0, 0]), 'face':'right'},  # right face
        {'normal': np.array([0, 1, 0]), 'point': np.array([0, -size/2, 0]), 'face':'back'},  # back face
        {'normal': np.array([0, -1, 0]), 'point': np.array([0, size/2, 0]), 'face':'front'},  # front face
        {'normal': np.array([0, 0, 1]), 'point': np.array([0, 0, -size/2]), 'face':'top'},  # front face
        {'normal': np.array([0, 0, -1]), 'point': np.array([0, 0, size/2]), 'face':'bottom'}   # back face
        ]

        return planes



        

    def find_collision_distance(self, planes, position, velocity, cube_side_length):
        # Normalize velocity vector to get direction
        direction = velocity / np.linalg.norm(velocity)
        
        
        # Find distances to each plane
        distances = []
        faces = []
        for plane in planes:
            numerator = np.dot(plane['normal'], (plane['point'] - position))
            denominator = np.dot(plane['normal'], direction)
            if denominator != 0:
                distance = numerator / denominator
                if distance > 0:
                    distances.append(distance)
                    faces.append(plane['face'])
        
        # Return the minimum distance
        if distances:
            index = np.argmin(distances)
            face = faces[index]
            return min(distances), face
            
        else:
            return None, None

    


    '''
    collision_distance, face = find_collision_distance(position, velocity, cube_side_length)

    if collision_distance is not None:
        print(f"The photon will collide with the {face} face of the cube at a distance of {collision_distance} units.")
    else:
        print("The photon will not collide with any face of the cube.")
    '''   
        

    
    
       
            

    









