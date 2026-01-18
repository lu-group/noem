# For the visualization of the FE model

import matplotlib.pyplot as plt

def visualize(femodel):
    # The node and member visualization should be plotted in one figure
    visualize_node(femodel)
    visualize_member(femodel)
    plt.show()
    return

def visualize_node(femodel):
    # Plot the node
    node = femodel.node
    plt.plot(node.x0, node.y0, 'o')
    return

def visualize_member(femodel):
    # Plot the member
    member = femodel.member
    node = femodel.node
    for i in range(len(member.ID)):
        start = member.NI[i]
        end = member.NJ[i]
        plt.plot([node.x0[start], node.x0[end]], [node.y0[start], node.y0[end]], 'k-')
    return