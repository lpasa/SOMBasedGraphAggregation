# SOM-based aggregation for graph convolutional neural networks

Graph property prediction is becoming more and more popular due to the increasing availability of scientific and social data naturally represented in a graph form. Because of that, many researchers are focusing on the development of improved graph neural network models. One of the main components of a graph neural network is the aggregation operator, needed to generate a graph-level representation from a set of node-level embeddings. The aggregation operator is critical since it should, in principle, provide a representation of the graph that is isomorphism invariant, i.e. the graph representation should be a function of graph nodes treated as a set. DeepSets (in: Advances in neural information processing systems, pp 3391–3401, 2017) provides a framework to construct a set-aggregation operator with universal approximation properties. In this paper, we propose a DeepSets aggregation operator, based on Self-Organizing Maps (SOM), to transform a set of node-level representations into a single graph-level one. The adoption of SOMs allows to compute node representations that embed the information about their mutual similarity. Experimental results on several real-world datasets show that our proposed approach achieves improved predictive performance compared to the commonly adopted sum aggregation and many state-of-the-art graph neural network architectures in the literature.

Paper: https://link.springer.com/article/10.1007/s00521-020-05484-4#Sec10

If you find this code useful, please cite the following:

>@article{pasa2020som,  
  title={SOM-based aggregation for graph convolutional neural networks},  
  author={Pasa, Luca and Navarin, Nicol{\`o} and Sperduti, Alessandro},  
  journal={Neural Computing and Applications},  
  pages={1--20},  
  year={2020},  
  publisher={Springer}  
}
