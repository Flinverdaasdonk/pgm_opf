# Overview

The goal of this project is to implement various formulations of the 'optimal power flow' problem. This is done based on sources, listed below. The list of required packages to run this project is listed below

## Optimal power flow
Optimal power flow is an approach to determine setpoints in an electricity grid (e.g. generator production) such that an objective (e.g. minimize power losses) is achieved while constrained by grid limits (e.g. stay below a given current) and physics (e.g. power in = power out).

Optimal power flow is based on 'regular' power flow, which is a way to calculate the flow of power through a grid given some setpoints. Power Flow can be formulated in two ways: The bus-injection model (BIM) and the branch-flow model (BFM). Even though they describe the system using different equations, they are equivalent (According to [1].II.C). This means that an optimal point found using the BIM formulation corresponds to the same optimal point that would be found using the BFM formulation. Both are used depending on when the formulations are easiest

## BIM implementation
The BIM is in `OPF_SDP_BIM.ipynb`. The `SDP` means 'semi-definite programming' which is the most general possible formulation of OPF. It is based on the work in [1].This formulation:
- can be used for radial and mesh networks.
- The decision variables are defined by the node voltages (which can be converted to node powers)
- The constraints are
  - nodal power limits
  - nodal voltage limits
  - line current limits

## BFM (DistFlow) implementation
The BFM implementation is a special variation called 'DistFlow' (see [2]). This formulation can only be used for radial (=tree) networks. This formulation
- Can only be used for radial networks
- Is supposedly faster than the BIM implementation
- The decision variables are
  - Line powers
  - Line currents
  - Nodal free power
  - Nodal voltages

## Description of data format

The data used is an extension of the format used in Power-Grid-Model. It consists of a network that is comprised of
- Nodes that correspond to sources/loads/junctions in the network.
- Lines that correspond to edges in the graph/connecting cables in the network
- Sources that uniquely map to nodes
- Loads that uniquely map to nodes

All of these are in a dictionary. The dictionaries are constructed using a consistent pattern. To illustrate this pattern, consider
```
    line["id"] = [3, 5]
    line["from_node"] = [1, 2]
    line["to_node"] = [2, 6]
```

Here each `key` ('id', 'from_node', etc.) describes a feature of that component, and maps to a list. `id` is the line id. Each component (node/line/source/load) has a unique `id` in the network. 
The line at index `1` maps to `id=5` and `from_node=2` (which means that the `id` of the node it is coming from is `2`), and `to_node=6` (which means that the `id` of the node it is going to is `6`). 

The `id`s are used to uniquely identify components. The indices of the list are used to keep track of which attributes correspond to which id. 

This could be seen as a matrix with features (=keys) as rows, and index as columns, or as a pandas DataFrame. However, since power-grid-model uses this format we'll use it too.

See `network_utilities.make_simple_radial_net()` to see which features are used in the OPF calculation.

## Sources
- [1] Convex Relaxations of Optimal Power Flow, Part I. By Steven Low, https://arxiv.org/abs/1405.0766
- [2] Towards Distributed Energy Services: Decentralizing Optimal Power Flow with Machine Learning, By Roel Dobbe, https://arxiv.org/pdf/1806.06790.pdf
- [3] Branch Flow Model: Relaxations and Convexification (Part I), By Masoud Farivar and Steven Low https://arxiv.org/pdf/1204.4865.pdf


## Requirements
This project uses the following packages:

power-grid-model
pandapower
cvxpy
optional: mosek. A much faster solver than the standard cvxpy solver (about 20x faster) and freely available for academic use (just download it with academic email on their website). If you choose to use this, make sure you install mosek using the same channel (e.g. conda-forge) as cvxpy, otherwise cvxpy can't find it.
