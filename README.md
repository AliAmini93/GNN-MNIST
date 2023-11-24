# Fundamental Graph Neural Network for Edge Prediction in MNIST Dataset
Implementing a Graph Neural Network (GNN) for Edge Prediction on MNIST Images

Each 28x28 digit image from the MNIST dataset is represented in this implementation as a graph. Each pixel in the image grid is considered a node, with the node's feature being its pixel intensity, normalized within the range [0, 1].

Contrast with a related notebook, (https://github.com/AliAmini93/GNN-MNIST/blob/main/gnn-mnist.ipynb), where pixel connectivity in the adjacency matrix was determined by neighborhood pixels using a Gaussian filter based on Euclidean distance, this notebook introduces a novel approach. Here, we predict the edges of the graph through a dedicated subnetwork within the GNN model:

```python
self.edge_predictor = nn.Sequential(nn.Linear(features_coord, 32),  # coord to hidden
                                nn.ReLU(),
                                nn.Linear(32, 1),  # hidden to edge
                                nn.Tanh())

```

The resulting adjacency matrix, denoted as $A$, is crucial for computing the output of a network layer:

$$X^{(l+1)}=A X^{(l)} W^{(l)}.$$

In this formula, $A$ is the $N \times N$ adjacency matrix, and $X$ is the $N \times C$ feature matrix representing a 2D coordinate array. Here, $N$  equals the total number of pixels in an MNIST image, $28 \times 28 = 784$. $W$ is the weight matrix with dimensions $N \times P$, where $P$ corresponds to the number of classes, especially relevant in a single hidden layer architecture context.
