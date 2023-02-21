import numpy as np 

class material():

    def __init__(self, l1depth=1, l1n=1, l2depth=1, l2n=2):

        self.size = l1depth + l2depth
        # refractive indexes
        self.depth1  = 1000*l1depth
        self.depth2  = 1000*l2depth
        self.n0 = l1n
        self.n1 = l2n

        self.z_array = np.ones(1000*self.size)

        self.z_array[:self.depth1] = self.n0
        self.z_array[self.depth1:] = self.n1

    def printArray(self):
        print(self.z_array)


test = material()
print(test.z_array)






