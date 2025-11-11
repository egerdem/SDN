The scattering matrix S, calculated as:
S = \frac{2}{N-1}1_{(N-1) \times (N-1)} - I,,
is applied to the incoming pressure wave variables to compute the outgoing wave variables at each node. This is a fundamental part of how energy is redistributed among connected nodes, ensuring the energy-preserving (unitary) nature of the SDN model.

	•	Incoming Waves: Each SDN node receives incoming wave variables (p+) from its connected neighbors.
	•	Scattering Operation: The scattering matrix S determines how these incoming waves are mixed and reflected back to the neighboring nodes. 
	•	Propagation: After the scattering operation, the outgoing wave variables (p−) are propagated through the bidirectional delay lines to the connected nodes.
In summary, the - coefficients of the matrix is intrinsic to the energy redistribution process within the SDN model. It ensures that the scattering process adheres to the energy-conserving principles required for realistic room acoustic simulation.

Key Definitions
	•	Incoming Delay Lines: These are the wave signals coming into the current node from other connected nodes.
	•	Example: incoming_delay_lines = [Node1 to Node2, Node3 to Node2, Node4 to Node2, Node5 to Node2, Node6 to Node2]
	•	Length: Matches the number of connections to the current node.
	•	Outgoing Delay Lines: These are the wave signals propagating out from the current node to other connected nodes.
	•	Example: outgoing_delay_lines = [Node2 to Node1, Node2 to Node3, Node2 to Node4, Node2 to Node5, Node2 to Node6]
	•	Length: Matches the number of connections from the current node.
	•	Scattering Matrix: Governs how incoming waves are distributed to outgoing waves.
	•	S[i][j]: The coefficient used to compute the contribution of the jth incoming wave to the ith outgoing wave.
	•	Rows: Correspond to outgoing delay lines (where the wave is sent).
	•	Columns: Correspond to incoming delay lines (where the wave came from).

Understanding the Indices
	•	Incoming Index j:
	•	Refers to the index in incoming_delay_lines (e.g., j=0 corresponds to "Node1 to Node2").
	•	Each column in the scattering matrix corresponds to a specific incoming delay line.
	•	Outgoing Index i:
	•	Refers to the index in outgoing_delay_lines (e.g., i=0 corresponds to "Node2 to Node1").
	•	Each row in the scattering matrix corresponds to a specific outgoing delay line.

How S[i][j] Works
	•	S[i][j] The scattering matrix element determines how much of the wave coming from the jth incoming line is sent to the ith outgoing line.
	•	For example:
	•	S[0][0]: Contribution of the wave from Node1 to Node2 to the outgoing wave Node2 to Node1.
	•	S[1][0] Contribution of the wave from Node1 to Node2 to the outgoing wave Node2 to Node3.

