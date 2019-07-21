import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy.linalg
import os 
os.chdir('C:/Kaige_Research/Code/GSP/code/')
from utils import *

user_num=50
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
lambda_=1

user_f=dictionary_matrix_generator(user_num, random_user_f, lap, lambda_)
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise

## smoothness level 
## L\theta*
## lambda_

#smoothness=np.trace(np.dot(np.dot(user_f.T, lap), user_f))
#fitness=np.linalg.norm(np.dot(lap, user_f))

lam_list=list(np.arange(5))
smoothness_list=np.zeros(len(lam_list))
fitness_list=np.zeros(len(lam_list))
for index, lam in enumerate(np.arange(5)):
	print('lambda_=',lam)
	user_f=dictionary_matrix_generator(user_num, random_user_f, lap, lam)
	smoothness_list[index]=np.trace(np.dot(np.dot(user_f.T, lap), user_f))
	fitness_list[index]=np.linalg.norm(np.dot(lap, user_f))

plt.figure()
plt.plot(smoothness_list, label='smoothness')
plt.plot(fitness_list, label='fitness')
plt.legend(loc=0, fontsize=12)
plt.ylabel('smoothness')
plt.xlabel('lambda')
plt.show()

Sigma=np.cov(item_f.T)
sigma_evals=np.linalg.eigvals(Sigma)
min_e=np.min(sigma_evals)
max_e=np.max(sigma_evals)
kappa=min_e/18

evals, evec=np.linalg.eig(lap)
lambda_2=np.sort(evals)[1]

D=2
alpha=8*noise_level*np.sqrt(D)*np.sqrt(item_num+dimension)/(item_num*user_num)
rank=user_num
error_bound=alpha*(rank+2*np.linalg.norm(np.dot(lap, user_f)))/(kappa+lambda_2)

error_bound_matrix=np.zeros((3, item_num))
for index, lambda_ in enumerate([1,2,3]):
	error_bound_list=[]
	user_f=dictionary_matrix_generator(user_num, random_user_f, lap, lambda_)
	for m in range(item_num):
		alpha=8*noise_level*np.sqrt(D)*np.sqrt(m+dimension)/(m*user_num)
		rank=user_num
		error_bound=alpha*(rank+2*np.linalg.norm(np.dot(lap, user_f)))/(kappa+lambda_2)
		error_bound_list.extend([error_bound])
	error_bound_matrix[index]=error_bound_list

plt.figure()
plt.plot(error_bound_matrix.T)
plt.show()

user_v={}
user_bias={}
user_noise={}
user_local_smooth={}
user_ls={}
user_ls_matrix=np.zeros((user_num, dimension))
for i in range(user_num):
	user_v[u]=np.zeros((dimension, dimension))
	user_bias[u]=np.zeros(dimension)
	user_noise[u]=np.zeros(dimension)
	user_local_smooth[u]=np.zeros(dimension)
	user_ls[u]=np.zeros(dimension)
user_error=np.zeros(user_num)

graph_error_matrix=np.zeros((3, item_num))
ridge_error_matrix=np.zeros((3, item_num))
error_bound_matrix_2=np.zeros((3, item_num))

for index, lambda_ in enumerate([1,2,3]):
	user_f=dictionary_matrix_generator(user_num, random_user_f, lap, lambda_)
	item_f=np.random.normal(size=(item_num, dimension))
	item_f=Normalizer().fit_transform(item_f)
	clear_signal=np.dot(user_f, item_f.T)
	noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
	noisy_signal=clear_signal+noise
	## emperical error graph-ridge 
	graph_error_list=[]
	ridge_error_list=[]
	error_bound_list_2=[]
	Lap=np.kron(lap, np.identity(dimension))
	cov=0.1*Lap+0.01*np.identity(user_num*dimension)
	v=0.1*np.identity(user_num*dimension)
	bias=np.zeros(user_num*dimension)
	for i in range(item_num):
		print('item num=',i)
		x=item_f[i]
		x_long_matrix=np.zeros((user_num, user_num*dimension))
		for u in range(user_num):
			x_long_matrix[u, u*dimension:(u+1)*dimension]=x
			user_noise[u]=np.linalg.norm(np.dot(item_f[:i].T, noise[u,:i]))
			user_v[u]=np.dot(item_f[:i].T, item_f[:i])
			user_bias[u]=np.dot(item_f[:i].T, noisy_signal[u,:i])
			user_ls[u]=np.dot(np.linalg.pinv(user_v[u]), user_bias[u])
			user_ls_matrix[u]=user_ls[u]
			user_local_smooth[u]=user_f[u]-np.dot(user_ls_matrix.T,-lap[u],)+user_ls_matrix[u]
			user_error[u]=np.sqrt(np.trace(np.linalg.pinv(user_v[u])**2))*(0.1*np.linalg.norm(user_local_smooth[u])+np.linalg.norm(user_noise[u]))
			sum_error=np.sum(user_error)
		error_bound_list_2.extend([sum_error])

		cov+=np.dot(x_long_matrix.T, x_long_matrix)
		v+=np.dot(x_long_matrix.T, x_long_matrix)
		y_vec=noisy_signal[:, i].T.flatten()
		bias+=np.dot(x_long_matrix.T, y_vec)
		graph_est=np.dot(np.linalg.pinv(cov), bias).reshape((user_num, dimension))
		ridge_est=np.dot(np.linalg.pinv(v), bias).reshape((user_num, dimension))
		graph_error=np.linalg.norm(graph_est-user_f)
		ridge_error=np.linalg.norm(ridge_est-user_f)
		graph_rank=np.linalg.matrix_rank(graph_est-user_f)
		ridge_rank=np.linalg.matrix_rank(ridge_est-user_f)
		graph_error_list.extend([graph_error])
		ridge_error_list.extend([ridge_error])
	graph_error_matrix[index]=graph_error_list
	ridge_error_matrix[index]=ridge_error_list
	error_bound_matrix_2[index]=error_bound_list_2

plt.figure()
plt.plot(graph_error_matrix.T, label='graph')
plt.plot(ridge_error_matrix.T,'.-', label='ridge')
plt.plot(error_bound_matrix.T, '*', label='error bound')
plt.plot(error_bound_matrix_2.T[30:], '-', label='error bound 2')
plt.legend(loc=0, fontsize=12)
plt.show()

## graph error bound individual 
