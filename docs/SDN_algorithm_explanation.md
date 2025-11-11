# Scattering Delay Network (SDN) Algorithm Explanation

## Overview
The Scattering Delay Network (SDN) is an artificial reverberator that models room acoustics using scattering nodes positioned on the walls of an enclosure. These nodes are interconnected via delay lines and perform scattering operations to simulate sound propagation and reverberation.

## Key Concepts

### Pressure Variables (p)
- Represent the actual sound pressure at a point in space.
- In SDN, the pressure at a node `p_k(n)` is calculated as:
  
  \[
  p_k(n) = p_{Sk}(n) + \frac{2}{N-1} \sum_{i=1}^{N-1} p_{ki}^{+}(n)
  \]
  
  - `p_{Sk}(n)`: Pressure contribution from the source.
  - `p_{ki}^{+}(n)`: Incoming wave variables from neighboring nodes.
  
### Wave Variables (p^+ and p^-)
- Represent traveling waves moving in specific directions.
- Used in scattering operations between nodes.
- Related to pressure but not equal to it.

### Source Input Distribution
- Source input is distributed to incoming wave variables:
  
  \[
  \tilde{p}_{ki}^{+}(n) = p_{ki}^{+}(n) + 0.5 \, p_{Sk}(n)
  \]
  
  - Corresponds to Equation (7) in the paper.

### Scattering Matrix
- Operates on wave variables, not pressures.
- Input: Incoming waves (`p_{ki}^{+}`)
- Output: Outgoing waves (`p_{ki}^{-}`)

## Algorithm Steps
1. **Distribute Source Input to Nodes**
   - Calculate source pressure contribution with attenuation.
   - Append to source-to-node delay lines.

2. **Process Each Node**
   - Collect incoming wave variables from other nodes.
   - Apply scattering matrix to get outgoing waves.
   - Calculate node pressure using incoming waves.
   - Send outgoing waves to microphone.

3. **Update Node-to-Node Connections**
   - Use outgoing waves to update delay lines between nodes.
   - Apply wall attenuation to outgoing waves.

4. **Calculate Microphone Output**
   - Sum outgoing waves scaled by `2/(N-1)`.
   - Apply microphone gains.

## Pseudocode
```pseudo
Initialize room and SDN parameters

For each sample:
  For each node:
    Distribute source input to node
    Collect incoming waves from other nodes
    Apply scattering matrix to incoming waves
    Calculate node pressure
    Send outgoing waves to microphone
    Update node-to-node delay lines

Calculate room impulse response
```

This explanation provides a detailed understanding of the SDN algorithm, highlighting the distinction between pressure and wave variables, and ensuring energy conservation through proper scattering operations. The pseudocode outlines the main steps involved in processing sound through the SDN network. 