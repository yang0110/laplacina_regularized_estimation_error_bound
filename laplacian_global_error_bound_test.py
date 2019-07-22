import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy.linalg
import os 
os.chdir('C:/Kaige_Research/Code/GSP/code/')
from utils import *

user_num=20
item_num=100
iteration=item_num
dimension=5
noise_level=0.1
random_user_f=np.random.normal(size=(user_num, dimension))
adj=rbf_kernel(random_user_f)

#plot graph
# np.fill_diagonal(adj,0)
# true_adj=np.round(adj,decimals=2)
# graph, edge_num=create_networkx_graph(user_num, true_adj)
# labels = nx.get_edge_attributes(graph,'weight')
# edge_weight=true_adj[np.triu_indices(user_num,1)]
# edge_color=edge_weight[edge_weight>0]
# pos = nx.spring_layout(graph)
# plt.figure(figsize=(5,5))
# nodes=nx.draw_networkx_nodes(graph, pos, node_size=20, node_color='y')
# edges=nx.draw_networkx_edges(graph, pos, width=0.05, alpha=1, edge_color='k')
# edge_labels=nx.draw_networkx_edge_labels(graph,pos, edge_labels=labels, font_size=5)
# plt.axis('off')
# plt.show()

lap=csgraph.laplacian(adj, normed=False)
normed_lap=csgraph.laplacian(adj, normed=True)
lambda_=1
user_f=dictionary_matrix_generator(user_num, random_user_f, lap, lambda_)
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise

cov=0.1*np.kron(normed_lap, np.identity(dimension))
bias=np.zeros(user_num*dimension)

graph_error_list=[]
error_bound_list=[]
for i in range(item_num):
	print('item index=', i)
	x=item_f[i]
	x_long_matrix=np.zeros((user_num, user_num*dimension))
	for u in range(user_num):
		x_long_matrix[u, u*dimension:(u+1)*dimension]=x 

	cov+=np.dot(x_long_matrix.T, x_long_matrix)
	bias+=np.dot(x_long_matrix.T, noisy_signal[:,i])
	graph_est=np.dot(np.linalg.pinv(cov), bias).reshape((user_num, dimension))
	graph_error=np.linalg.norm(graph_est-user_f)
	graph_error_list.extend([graph_error])

	error_matrix=graph_est-user_f
	rank=np.linalg.matrix_rank(error_matrix)
	rank=user_num
	alpha=8*noise_level*np.sqrt(2)*np.sqrt(i+dimension)/(i+user_num)
	fitness=2*np.linalg.norm(np.dot(lap,user_f))
	Sigma=np.cov(item_f.T)
	sigma_evals=np.linalg.eigvals(Sigma)
	min_e=np.min(sigma_evals)
	kappa=min_e/18
	evals, evec=np.linalg.eig(lap)
	lambda_2=np.sort(evals)[1]
	error_bound=alpha*(rank+fitness)/(kappa+lambda_2)
	error_bound_list.extend([error_bound])

plt.figure()
plt.plot(graph_error_list, label='emp error')
plt.plot(error_bound_list, label='error bound')
plt.legend(loc=0)
plt.show()

