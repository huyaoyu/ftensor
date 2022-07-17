import matplotlib.pyplot as plt
import networkx as nx
import torch

from .ftensor import f_eye
from .frame_graph import RefFrame, add_frame, add_pose_edge, get_tf_along_chain

if __name__ == '__main__':
    # Create a directed graph.
    G = nx.DiGraph()

    # ========== Add nodes. ==========
    awf = RefFrame('awf', 'AirSim World NED Frame. ')
    rbf = RefFrame('rbf', 'Rig Body Frame. ')
    rpf = RefFrame('rpf', 'Rig Panorama Frame. ')
    cbf = RefFrame('cbf', 'Camera Body Frame. ')
    cpf = RefFrame('cpf', 'Camera Panorama Frame. ')
    cif = RefFrame('cif', 'Camera Image Frame. ')

    add_frame(G, awf)
    add_frame(G, rbf)
    add_frame(G, rpf)
    add_frame(G, cbf)
    add_frame(G, cpf)
    add_frame(G, cif)

    # ========== Add edges. ==========

    # RBF.
    T_awf_rbf = f_eye(4, 'awf', 'rbf')
    T_awf_rbf.translation = torch.as_tensor([ 0, 1, 0 ])
    T_awf_rbf = T_awf_rbf.to('cuda')
    add_pose_edge(G, T_awf_rbf )

    # RPF.
    T_rbf_rpf = f_eye(4, 'rbf', 'rpf')
    T_rbf_rpf.rotation = torch.as_tensor([
        [ 0,  0, -1],
        [-1,  0,  0],
        [ 0,  1,  0]
    ])
    T_rbf_rpf = T_rbf_rpf.to('cuda')
    add_pose_edge(G, T_rbf_rpf )

    # A CBF.
    T_rbf_cbf = f_eye(4, 'rbf', 'cbf')
    T_rbf_cbf.translation = torch.as_tensor([ 0, 1, 0 ])
    T_rbf_cbf = T_rbf_cbf.to('cuda')
    add_pose_edge(G, T_rbf_cbf )

    # A CPF.
    T_cbf_cpf = f_eye(4, 'cbf', 'cpf')
    T_cbf_cpf.rotation = torch.as_tensor([
        [ 0,  0, -1],
        [-1,  0,  0],
        [ 0,  1,  0]
    ])
    T_cbf_cpf = T_cbf_cpf.to('cuda')
    add_pose_edge(G, T_cbf_cpf )

    # A CIF.
    T_cbf_cif = f_eye(4, 'cbf', 'cif')
    T_cbf_cif.rotation = torch.as_tensor([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    T_cbf_cif = T_cbf_cif.to('cuda')
    add_pose_edge(G, T_cbf_cif )

    # ========== End of adding edges. ==========

    # Try to find a shortest path between two nodes.
    path = nx.shortest_path(G, source='cif', target='awf')
    print(path)

    # Get the TF chain along path.
    tf = get_tf_along_chain(G, path)
    print(tf)

    # Should be
    # [[ 0.,  1.,  0., -2.], 
    #  [ 0.,  0.,  1.,  0.], 
    #  [ 1.,  0.,  0.,  0.], 
    #  [ 0.,  0.,  0.,  1.]]

    # Draw the graph for visualization.
    vis_pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, vis_pos, cmap=plt.get_cmap('jet'))
    nx.draw_networkx_labels(G, vis_pos)
    nx.draw_networkx_edges(G, vis_pos, edge_color='r', arrows=True)
    plt.show()
    