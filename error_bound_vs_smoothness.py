import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy.linalg
import os 
os.chdir('C:/Kaige_Research/Code/GSP/code/')
from utils import *

path='C:/Kaige_Research/Code/GSP/results/'

user_num=20
item_num=200
iteration=item_num
dimension=5
noise_level=0.1
random_user_f=np.random.normal(size=(user_num, dimension))
random_user_f=Normalizer().fit_transform(random_user_f)
adj=rbf_kernel(random_user_f)
thres=0
adj[adj<thres]=0
##plot graph
np.fill_diagonal(adj,0)
true_adj=np.round(adj,decimals=2)
graph, edge_num=create_networkx_graph(user_num, true_adj)
labels = nx.get_edge_attributes(graph,'weight')
edge_weight=true_adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
pos = nx.spring_layout(graph)
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_size=20, node_color='y')
edges=nx.draw_networkx_edges(graph, pos, width=0.05, alpha=1, edge_color='k')
edge_labels=nx.draw_networkx_edge_labels(graph,pos, edge_labels=labels, font_size=5)
plt.axis('off')
plt.savefig(path+'full_connected_random_graph'+'.png', dpi=200)
plt.show()

lap=csgraph.laplacian(adj, normed=False)
normed_lap=csgraph.laplacian(adj, normed=True)
lambda_=1
random_user_f_2=np.random.normal(size=(user_num, dimension))
random_user_f_2=Normalizer().fit_transform(random_user_f_2)
user_f=dictionary_matrix_generator(user_num, random_user_f_2, lap, lambda_)
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise

## test the fitness and smoothness 
fitness=np.linalg.norm(np.dot(normed_lap, user_f))
smoothness=np.trace(np.dot(np.dot(user_f.T, normed_lap), user_f))

fitness_list=[]
smoothness_list=[]
for lam in np.arange(0, 5,0.1):
	user_f=dictionary_matrix_generator(user_num, random_user_f_2, lap, lam)
	fitness=np.linalg.norm(np.dot(normed_lap, user_f))
	smoothness=np.trace(np.dot(np.dot(user_f.T, normed_lap), user_f))
	fitness_list.extend([fitness])
	smoothness_list.extend([smoothness])

plt.figure(figsize=(5,5))
plt.plot(np.arange(0,5,0.1), fitness_list, label='Fitness')
plt.plot(np.arange(0,5,0.1), smoothness_list, label='Smoothness')
plt.legend(loc=0, fontsize=12)
plt.ylabel('Fitness and Smoothness', fontsize=12)
plt.xlabel('Alpha', fontsize=12)
plt.tight_layout()
plt.savefig(path+'fitness_and_smoothness'+'.png', dpi=100)
plt.show()

lambda_list=[0,1,5]
emp_error_matrix=np.zeros((len(lambda_list), item_num))
global_error_matrix=np.zeros((len(lambda_list), item_num))
ind_error_matrix=np.zeros((len(lambda_list), item_num))

random_user_f_2=np.random.normal(size=(user_num, dimension))
random_user_f_2=Normalizer().fit_transform(random_user_f_2)
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)

for index, lam in enumerate(lambda_list):
	user_f=dictionary_matrix_generator(user_num, random_user_f_2, lap, lam)
	clear_signal=np.dot(user_f, item_f.T)
	noisy_signal=clear_signal+noise
	graph_user_f=np.zeros((user_num, dimension))
	ridge_user_f=np.zeros((user_num, dimension))
	ls_user_f=np.zeros((user_num, dimension))
	cov=0.5*np.kron(normed_lap, np.identity(dimension))
	bias=np.zeros(user_num*dimension)
	user_v={}
	user_xx={}
	user_bias={}
	user_noise={}
	for u in range(user_num):
		user_v[u]=0.1*np.identity(dimension)
		user_xx[u]=np.zeros((dimension, dimension))
		user_bias[u]=np.zeros(dimension)
		user_noise[u]=np.zeros(dimension)
	for i in range(item_num):
		print('lambda=', lam)
		print('item index=', i)
		x=item_f[i]
		v=np.outer(x,x)
		x_long_matrix=np.zeros((user_num, user_num*dimension))
		ind_error_bound=[]
		for u in range(user_num):
			user_v[u]+=v 
			user_xx[u]+=v 
			user_bias[u]+=np.dot(x, noisy_signal[u,i])
			user_noise[u]+=np.dot(x, noise[u,i])
			ridge_user_f[u]=np.dot(np.linalg.pinv(user_v[u]), user_bias[u])
			ls_user_f[u]=np.dot(np.linalg.pinv(user_xx[u]), user_bias[u])
			average=np.dot(user_f.T, -normed_lap[u])+user_f[u]
			graph_user_f[u]=ridge_user_f[u]+0.5*np.dot(np.linalg.pinv(user_xx[u]), average)
			xx_inv=np.linalg.pinv(user_xx[u])
			ind_error_bound.extend([np.linalg.norm(xx_inv)*(0.5*np.linalg.norm(user_f[u]-average)+np.linalg.norm(user_noise[u]))])
			x_long_matrix[u, u*dimension:(u+1)*dimension]=x 
		sum_ind_error=np.sum(ind_error_bound)
		ind_error_matrix[index, i]=sum_ind_error
		cov+=np.dot(x_long_matrix.T, x_long_matrix)
		bias+=np.dot(x_long_matrix.T, noisy_signal[:,i])
		graph_est=np.dot(np.linalg.pinv(cov), bias).reshape((user_num, dimension))
		graph_error=np.linalg.norm(graph_est-user_f)
		emp_error_matrix[index, i]=graph_error
		error_matrix=graph_est-user_f
		rank=np.linalg.matrix_rank(error_matrix)
		alpha=8*noise_level*np.sqrt(3)*np.sqrt(i+dimension)/(i+user_num)
		fitness=2*np.linalg.norm(np.dot(normed_lap,user_f))
		Sigma=cov.copy()
		sigma_evals=np.linalg.eigvals(Sigma)
		min_e=np.min(sigma_evals)
		kappa=min_e/18
		evals, evec=np.linalg.eig(normed_lap)
		lambda_2=np.sort(evals)[1]
		global_error_matrix[index,i]=alpha*(rank+fitness)/(kappa+lambda_2)


labels=lambda_list
for i in range(len(labels)):
	labels[i]='alpha='+str(labels[i])

plt.figure(figsize=(5,5))
lines=plt.plot(emp_error_matrix.T[10:])
plt.legend(lines, labels, loc=0, fontsize=12)
plt.xlabel('sample size', fontsize=12)
plt.ylabel('Error (MSE)', fontsize=12)
plt.tight_layout()
plt.savefig(path+'emperical_error'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
lines=plt.plot(global_error_matrix.T[10:])
plt.legend(lines, labels, loc=0,fontsize=12)
plt.xlabel('sample size', fontsize=12)
plt.ylabel('Error (MSE)', fontsize=12)
plt.tight_layout()
plt.savefig(path+'global_error_bound'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
lines=plt.plot(ind_error_matrix.T[10:])
plt.legend(lines, labels, loc=0,fontsize=12)
plt.xlabel('sample size', fontsize=12)
plt.ylabel('Error (MSE)', fontsize=12)
plt.tight_layout()
plt.savefig(path+'ind_error_bound'+'.png', dpi=100)
plt.show()



for i in range(len(lambda_list)):
	plt.figure(figsize=(5,5))
	plt.plot(emp_error_matrix[i][10:], label='Emperical Error')
	plt.plot(global_error_matrix[i][10:], label='Global Error Bound')
	plt.plot(ind_error_matrix[i][10:], label='Ind Error Bound (SUM)')
	plt.legend(loc=0, fontsize=12)
	plt.xlabel('sample size', fontsize=12)
	plt.ylabel('Error(MSE)', fontsize=12)
	plt.savefig(path+'error_bound_comparsion_vs_smoothness_alpha_%s'%(lambda_list[i])+'.png', dpi=100)
	plt.show()
