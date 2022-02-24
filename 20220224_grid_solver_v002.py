# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 08:22:22 2022

@author: 54911
"""

import networkx as nx
from matplotlib import pyplot as plt

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

circuit = Circuit("Millman's theorem")

Nx = 300
Ny = 40 
G = nx.grid_2d_graph(Ny,Nx)

fig, ax = plt.subplots(1,1)
pos = {(x,y):(y,-x) for x,y in G.nodes()}
# nx.draw(G, pos=pos, node_color='lightgreen', 
        # with_labels=True,  node_size=600)

ax.set_title('ee')
ax.axis('on')
ax.lines

ax.patches
ax.artists
# ax.set_title('grid')
## enumeration of nodes from 1 till Nnodes
def get_key(val, dictu):
    for key, value in dictu.items():
         if val == value:
             return key
         
dict_nodes = {}
for n, node in enumerate(G.nodes()):
    print (n,node)
    dict_nodes[n+1] = node

assert 2 == get_key((0,1),dict_nodes)

print ('the numbers of nodes is:', len(dict_nodes))

NV = Nx*Ny+100

circuit.V('input',NV,circuit.gnd, 10@u_V)

i=0
for x,y in G.nodes():
    if x == 0:
        circuit.R('el'+str(i),NV, 
                  get_key((x,y),dict_nodes),0.001@u_kOhm)
        i += 1
    # print (x,y)
    
for edge in G.edges():
    origin, end = edge
    print (origin, get_key(origin,dict_nodes), '-->', 
           end, get_key(end,dict_nodes))
    circuit.R('el'+str(i),get_key(origin,dict_nodes), 
              get_key(end,dict_nodes),2@u_kOhm)
    i += 1
    
for x,y in G.nodes():
    if x == Ny-1:
        circuit.R('el'+str(i),get_key((x,y),dict_nodes), 
                  circuit.gnd,0.001@u_kOhm)
        i += 1
        
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.operating_point()

for node in analysis.nodes.values(): 
    print('Node {}: {:4.1f} V'.format(str(node), float(node))) 

for node in analysis.branches.values(): 
    print('Node {}: {:5.2f} A'.format(str(node), float(node)))     

# nx.draw(
#     G,  pos, ax = ax,
#     node_color="tab:green",
#     edgecolors="tab:gray",  # Node surface color
#     edge_color="tab:gray",  # Color of graph edges
#     node_size=400,
#     with_labels=True,
#     width=6,
# )
# #%%
voltages = [i for i in range(0,Nx*Ny)]
nodes = [i for i in range(0,Nx*Ny)]
for node in analysis.nodes.values(): 
    nnode = int(str(node)) 
    # print (nnode, float(node))
    if nnode in list(range(1,Nx*Ny+1)):
        print (nnode, float(node))
        voltages[nnode-1] = plt.cm.jet(float(node)/10.)
        nodes[nnode-1] = float(node)

currents = []

for edge in G.edges():
    origin, end = edge
    idx_init = get_key(origin,dict_nodes)
    idx_end = get_key(end,dict_nodes)
    current = nodes[idx_init-1]- nodes[idx_end-1]
    print (origin, idx_init, '-->', 
           end, idx_end, ':', current)
    currents.append(plt.cm.jet(float(current)/2.5))
    
nx.draw(
    G,  pos, ax = ax,
    node_color=voltages,
    # edgecolors="tab:gray",  # Node surface color
    edge_color=currents,  # Color of graph edges
    node_size=400,
    with_labels=False,
    width=6,
)

#%%

