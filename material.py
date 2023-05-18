import numpy as np 

class medium():

    def __init__(self):

        
        # refractive indexes

        self.NumberPhotons = 10000


        
        '''
        # [depth, refractive_index(n), u_a, u_s, g]
        layer_null = [-99999.9, 1, 1, 1, 0]
        layer0 = [float(0), 1, 1, 1, 0]
        layer1 = [0.01, 1, 10, 90, 0.75]
        layer2 = [0.02, 1, 10, 90, 0.75]
        layer3 = [99999.9, 1, 1, 1, 0]

        self.layers = {-1:layer_null,
                       0:layer0,
                       1:layer1,
                       2:layer2,
                       3:layer3}
        
        self.layers_important = {0:layer1,
                                 1:layer2,}
        
        self.depth = 0.02
        
        
        
        # [depth, refractive_index(n), u_a, u_s, g]
        self.inclusion_center = np.array([100,0,0.012])
        self.inclusion_size = 0.002
        self.inclusion_properties = [self.inclusion_center[-1],1,10,90,0.75]
        
        #self.inclusion_layer = 0
        #self.depth = 20
        '''
    
        '''
        # [depth, refractive_index(n), u_a, u_s, g]
        layer_null = [-99999.9, 1, 1, 1, 0]
        layer0 = [float(0), 1, 1, 1, 0]
        layer1 = [1, 1, 0.1, 90, 0.9]
        layer2 = [5, 1, 0.1, 90, 0.9]
        layer3 = [10, 1, 0.1, 90, 0.9]
        layer4 = [99999.9, 1, 1, 1, 0]

        self.layers = {-1:layer_null,
                       0:layer0,
                       1:layer1,
                       2:layer2,
                       3:layer3,
                       4:layer4}
        
        self.layers_important = {0:layer1,
                                 1:layer2,
                                 2:layer3}
        
        self.depth = 10
        
        
        
        # [depth, refractive_index(n), u_a, u_s, g]
        self.inclusion_center = np.array([100,0,5.5])
        self.inclusion_size = 0.5
        self.inclusion_properties = [self.inclusion_center[-1],1,10,90,0.75]
        
        #self.inclusion_layer = 0
        #self.depth = 20
        '''
        #______________________________________________________
        #------------------------------------------------------
        #______________________________________________________

        
        # [depth, refractive_index(n), u_a, u_s, g]
        layer_null = [-999.9, 1, 1, 1, 0]
        layer0 = [float(0), 1, 1, 1, 0]
        layer1 = [0.7, 1.4, 0.2525, 254, 0.9] # surrounding layer 1 0.7
        layer2 = [1.7, 1.4, 0.2525, 254, 0.9] # inclusion layer     1.7
        layer3 = [6, 1.4, 0.2525, 254, 0.9]   # surrounding layer 3 semi infinite
        layer4 = [999.9, 1, 1, 1, 0]

        self.layers = {-1:layer_null,
                       0:layer0,
                       1:layer1,
                       2:layer2,
                       3:layer3,
                       4:layer4}
        
        self.layers_important = {0:layer1,
                                 1:layer2,
                                 2:layer3}
        
        # [depth, refractive_index(n), u_a, u_s, g]
        self.inclusion_center = np.array([0,0,1.2])
        self.inclusion_size = 1
        self.inclusion_properties = [self.inclusion_center[-1],1.3, 1.7049,180,0.9] # inclusion params
        
        #self.inclusion_layer = 0
        self.depth = 6
        


        
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
        layer8 = [0.500, 1.46, 0.975, 97.125, 0.8]
        layer9 = [0.6, 1.37, 1, 1, 0]             # Subcutaneous Fat
        layer10 = [999.9, 1.37, 1, 1, 0]                      # Muscle
        
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
                       9:layer9,
                       10:layer10}
     
        
        self.layers_important = {0:layer1,
                       1:layer2,
                       2:layer3,
                       3:layer4,
                       4:layer5,
                       5:layer6,
                       6:layer7,
                       7:layer8,
                       8:layer9}
        

        self.depth = self.layers_important[8][0]

        self.inclusion_center = np.array([100,0,0.5])
        self.inclusion_size = 0.5
        self.inclusion_properties = [self.inclusion_center[-1],1.39, 1.427, 145.625, 0.818]
        '''
        
    def inclusion(self, size, center_point):

        '''
        Defines a square inclusion based on the layers defined in __init__
        returns the 6 faces of the cube as '''

          # Normalize velocity vector to get direction
        '''
        direction = velocity / np.linalg.norm(velocity)
        print ('direction: ', direction)
        '''

        self.center_point = center_point 
        self.size = size 
        
        # Define the six planes of the cube
        planes = [
            {'normal': np.array([1, 0, 0]), 'point': center_point + [-size/2, 0, 0], 'face':'left'},  # left face
            {'normal': np.array([-1, 0, 0]), 'point':center_point + [size/2, 0, 0], 'face':'right'},  # right face
            {'normal': np.array([0, 1, 0]), 'point':center_point + [0, -size/2, 0], 'face':'back'},  # back face
            {'normal': np.array([0, -1, 0]), 'point':center_point + [0, size/2, 0], 'face':'front'},  # front face
            {'normal': np.array([0, 0, 1]), 'point':center_point + [0, 0, -size/2], 'face':'top'},  # front face
            {'normal': np.array([0, 0, -1]), 'point':center_point + [0, 0, size/2], 'face':'bottom'}   # back face
        ]

        
            
        inclusion_layer = False
        for i in range(len(self.layers_important)):
            if self.layers_important[i][0] > center_point[-1] and not inclusion_layer:
                inclusion_layer = i

        return planes, inclusion_layer



        

    def find_collision_distance(self, planes,center_point, size, position, velocity):
        # Normalize velocity vector to get direction
        direction = velocity / np.linalg.norm(velocity)
        
        
        # Find distances to each plane
        distances = []
        faces = []
        for plane in planes:
            numerator = np.dot(plane['normal'], (plane['point'] - position))
            #print ('numerator {}'.format(numerator))
            denominator = np.dot(plane['normal'], direction)
            if denominator != 0:
                distance = numerator / denominator
                if distance > 0:
                    intersection_pos = position + (direction * distance)
                    in_bounds = True
                    for i in range(len(intersection_pos)):
                
                        neg_side = center_point[i] - size/2
                        pos_side = center_point[i] + size/2

                        if abs(intersection_pos[i]) < neg_side or abs(intersection_pos[i]) > pos_side:
                            in_bounds = False       
                    # print (in_bounds)
                    if in_bounds:                         
                        distances.append(distance)
                        faces.append(plane['face'])
        # Return the minimum distance
        if distances:
            index = np.argmin(distances)
            face = faces[index]
            return min(distances), face
            
        else:
            return 99999999, None

    


    '''
    collision_distance, face = find_collision_distance(position, velocity, cube_side_length)

    if collision_distance is not None:
        print(f"The photon will collide with the {face} face of the cube at a distance of {collision_distance} units.")
    else:
        print("The photon will not collide with any face of the cube.")
    '''   
        

    
    
       
            

    









