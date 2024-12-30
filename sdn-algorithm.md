Below is a summary of SDN (scattering delay networks) method's algorithmic steps in LaTeX format. This is a simulator for calculating room impulse responses (RIR). IN SDN; first order reflections has exactly the same thoeretical path length which is same as the image source method (ISM) path lengths. HOwever, higher order reflections uses an approximation for this path length, to have a low-cost algorithm which does not calculate any actual path lengths above 1st order but only uses the distances between the specified nodes (scattering nodes). There is 1 scattering node at each wall, at the locations in which the first order reflection happens, according to the specified source and receiver (microphone) position.

In SDN,higher-order reflections will be automatically approximated by propagating signals iteratively through the delay network. No explicit distance calculations will be made for higher order reflections. Instead, the distance between nodes will be used to approximate the high number of bounces for higher order reflections.


The goal: of this project is to find why SDN RIRs has a sudden drop after first order reflections, and to find a way to fix this.

Code structure:
Geometric classes are already defined. walls has labels, their plane coefficients. Image sources are found for a given wall and the source location. For propagating the impulse signal from the source to the microphoen and to the SDN network, deque from python collections will be used since it is a flexible sized tube like structure with pop-add-remove-append-popleft properties. 


First experiments that Im carrying out:

Both for SDN and for ISM, and for every reflection path for every order, logging the path lengths to a dictionary, based on its order so that we will compare them and understand whether the distance approximation is logical or too broad. You may suggest a better way to do this in terms of data structure, code structure, etc. 

Simplified steps:

1. Initialize Room Geometry and Parameters, walls, nodes, delay lines
   - Use wall absorption coefficients \( \alpha \) to determine wall reflection coefficients sqrt(1-alpha)
   - For each wall, create a scattering node at the location of the first-order reflection, by finding the intersection point of the line connecting the image source  and the receive, with the corresponding wall.
   - Compute delay \( D_{S,k} \) from source \( \mathbf{x}_S \) to each wall node \( \mathbf{x}_k \):
     \[
     D_{S,k} = \left\lfloor \frac{F_s \|\mathbf{x}_S - \mathbf{x}_k\|}{c} \right\rfloor
     \]
   - Compute attenuation for spherical spreading, between source to nodes
     \[
     g_{S,k} = \frac{1}{\|\mathbf{x}_S - \mathbf{x}_k\|}
     \]

3. Scattering Operation at the nodes:
   - Define the scattering matrix \( S \):
     \[
     S = \beta A, \quad \beta = \sqrt{1 - \alpha}, \quad A \text{ is a lossless matrix}
     \]
   - Implement scattering:
     \[
     \mathbf{p}^- = S \mathbf{p}^+
     \]
     where \( \mathbf{p}^+ \) and \( \mathbf{p}^- \) are incoming and outgoing wave vectors.

4. Signal Propagation:
   - Connect nodes via bidirectional delay lines:
     \[
     D_{k,m} = \left\lfloor \frac{F_s \|\mathbf{x}_k - \mathbf{x}_m\|}{c} \right\rfloor
     \]
   - no attenuation for propagation between nodes.

5.  Node-to-Microphone Connections:
   
     attenuation:  \[
     g_{k,M} = \frac{1}{\1 + frac{|\mathbf{x}_k - \mathbf{x}_M\|}{|\mathbf{x}_S - \mathbf{x}_k\|}}
     \]




