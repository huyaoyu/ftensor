
# Author:
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
#
# Data:
# 20220626

import networkx as nx
import numpy as np
import torch

from .ftensor import FTensor, f_eye

FLOAT_TYPE=torch.float

# The node class.
class RefFrame(object):
    def __init__(self, name, comment=''):
        super().__init__()
        
        self.name    = name    # The name of the reference frame.
        self.comment = comment # More explanation of the naming.
        
# # The edge class:
# class RelPose(object):
#     def __init__(self, f0:str, f1:str, device=None):
#         '''
#         f0 and f1 are the names of the reference frames.
#         A relative pose is defined as the position and orientation of f1
#         w.r.t. f0 measured in f0.
#         '''
#         super().__init__()
        
#         self.pose = FTensor( 
#                 torch.eye(4, dtype=FLOAT_TYPE, device=device),
#                 f0=f0, 
#                 f1=f1 )
        
#     def to(self, device=None):
#         temp = RelPose(self.pose.f0, self.pose.f1)
#         temp.pose = self.pose.to(device=device)
#         return temp
    
#     @property
#     def device(self):
#         return self.pose.device
    
#     @property
#     def translation(self):
#         return self.pose.translation
    
#     @translation.setter
#     def translation(self, vec):
#         self.pose[0, 3] = vec[0]
#         self.pose[1, 3] = vec[1]
#         self.pose[2, 3] = vec[2]
    
#     @property
#     def rotation(self):
#         return self.pose[:3, :3]
    
#     @rotation.setter
#     def rotation(self, mat):
#         '''
#         mat could be a tensor, a numpy array, or a 2D list.
#         '''

#         if isinstance(mat, np.ndarray):
#             mat = torch.from_numpy(mat).to(device=self.device)
#         elif isinstance(mat, (list)):
#             mat = torch.Tensor(mat).to(dtype=FLOAT_TYPE, device=self.device)
#         elif isinstance(mat, torch.Tensor):
#             if isinstance(mat, FTensor):
#                 assert mat.f0 == self.pose.f0 and mat.f1 == self.pose.f1, \
#                     f'mat and self.pose must have the same frame names. \n mat: {mat.f0}, {mat.f1}, self.pose: {self.pose.f0}, {self.pose.f1}. '
#             mat = mat.to(device=self.device)
#         else:
#             raise Exception(f'Unsupported type of rotation. type(mat) = {type(mat)}')
            
#         self.pose[:3, :3] = mat
        
#     def inverse(self):
#         '''
#         Return the inverse of the transform/pose.
#         '''
#         return self.pose.inverse()

def add_frame(g, frame):
    g.add_node(frame.name, data=frame)

# A function for inserting an edge to the graph.
def add_pose_edge(g, pose):
    '''
    g is the graph.
    pose is defined as T_parent_child, following the convention defined in the Google Slides.
    '''
    
    g.add_edge(pose.f0, pose.f1, pose=pose)
    g.add_edge(pose.f1, pose.f0, pose=pose.inverse())

# A function for compute the accumulated transformation (tf) along a chain of reference frames.
def get_tf_along_chain(g, chain):
    '''
    g (DiGraph)
    chain (list of node keys)
    '''
    
    # We need at least two nodes in the chain.
    assert len(chain) > 1, f'len(path) = {len(chain)}, expect it to be larger than 1'
    
    for i in range( len(chain) - 1 ):
        p = chain[i]   # parent
        c = chain[i+1] # child
        
        # Get the edge between p and c.
        e = g[p][c]
        
        # Extract the pose attribute of the edge.
        pose_e = e['pose']
        
        # Initialize the tf in the first iteration.
        if i == 0:
            tf = torch.eye(4, dtype=FLOAT_TYPE, device=pose_e.device)
        
        # Chain the TFs.
        tf = tf @ pose_e
        
    return tf
