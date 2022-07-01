'''
Interleaving FPT.py

This code is built using the algorithm presented in Touli and Wang's 2019 FPT-Algorithms paper.

Author: Zach Schlamowitz
Date: Jun 27 2022'''

import binarytrees as bt

def augment_trees(tree1, tree2, delta):
    '''
    This function augments two trees, adding all nodes in the super levels of each tree.
    '''
    critical_heights_T1 = get_criticalheights(tree1)
    critical_heights_T2 = get_criticalheights(tree2)
    print(critical_heights_T1 == critical_heights_T2)
    for height in critical_heights_T1:
        print("height 1:", height)
        add_level(tree2.root, height + delta)
    
    for height in critical_heights_T2:
        print("height 2:", height)
        add_level(tree1.root, height - delta)

    return tree1, tree2



def get_nodeheights(node, nodeheights):
    '''
    This function creates a dictionary mapping the heights of nodes in a tree to the nodes at that height
    '''
    # Recursive Case:
    if node.left is not None:
        nodeheights = get_nodeheights(node.left, nodeheights)
    
    if node.key not in nodeheights.keys():
        nodeheights[node.key] = list()  # array will preserve left-right order
    nodeheights[node.key].append(node)  # adding existing tree node to the set of nodes at that height

    if node.right is not None:
        nodeheights = get_nodeheights(node.right, nodeheights)

    return nodeheights



def get_criticalheights(tree):
    '''
    This function identifies the critical heights of a tree, adds all levels (nodes) at those heights (using add_level()), 
    and creates a dictionary mapping the heights of nodes in a tree to the nodes at that height (using get_nodeheights())
    '''
    nodeheights = get_nodeheights(tree.root, nodeheights={})
    # print("nodeheights in get_crit:", nodeheights)

    for height in nodeheights:
        add_level(tree.root, height)
    
    criticalheights = get_nodeheights(tree.root, nodeheights={})
    # print("critheights in get_crit:", criticalheights)

    return criticalheights



def add_level(root, height):
    '''
    This function adds nodes to a tree at all intesection points along a horizontal line.
    The height of this horizontal line is specificed by <height>. 
    '''
    if root is None or root.key <= height:
        return
    
    if root.left:
        add_level(root.left, height)

        if root.left.key < height:  # if desired height is between the two nodes in question
            new_node = bt.TreeNode(key=height)
            new_node.left = root.left
            root.left = new_node
            new_node.origin = 'augment'
        
    if root.right:
        add_level(root.right, height)

        if root.right.key < height:  # if desired height is between the two nodes in question
            new_node = bt.TreeNode(key=height)
            new_node.right = root.right
            root.right = new_node
            new_node.origin = 'augment'
        
    return









def DPgoodmap(tree1, tree2, delta):
    # NEED: trees with the following strutures: valid pairs (S,w) 
    #   <-- super-levels 
    #   <-- critical heights 
    #   <-- all nodes in tree and their heights 
    #   also, depth() method in treenode class
    #   way to access children sets Ch(S) and Ch(w)

    ## BASE CASE:
    

    ## RECURSIVE CASE:

    return




