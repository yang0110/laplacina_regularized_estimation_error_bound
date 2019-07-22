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
item_num=500
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
for lam in [0,1,2,3,4,5]:
	user_f=dictionary_matrix_generator(user_num, random_user_f_2, lap, lam)
	fitness=np.linalg.norm(np.dot(normed_lap, user_f))
	smoothness=np.trace(np.dot(np.dot(user_f.T, normed_lap), user_f))
	fitness_list.extend([fitness])
	smoothness_list.extend([smoothness])

plt.figure(figsize=(5,5))
plt.plot(fitness_list, label='Fitness')
plt.plot(smoothness_list, label='Smoothness')
plt.legend(loc=0, fontsize=12)
plt.ylabel('Fitness and Smoothness', fontsize=12)
plt.xlabel('Alpha', fontsize=12)
plt.tight_layout()
plt.savefig(path+'fitness_and_smoothness'+'.png', dpi=100)
plt.show()


graph_user_f=np.zeros((user_num, dimension))
ridge_user_f=np.zeros((user_num, dimension))
ls_user_f=np.zeros((user_num, dimension))
emp_error=np.zeros((user_num, item_num))
error_bound=np.zeros((user_num, item_num))
graph_error_list=[]
global_error_bound_list=[]

cov=0.1*np.kron(normed_lap, np.identity(dimension))
bias=np.zeros(user_num*dimension)

user_cov={}
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
	print('item index=',i)
	x=item_f[i]
	v=np.outer(x,x)
	x_long_matrix=np.zeros((user_num, user_num*dimension))
	for u in range(user_num):
		user_v[u]+=v 
		user_xx[u]+=v 
		user_bias[u]+=np.dot(x, noisy_signal[u,i])
		user_noise[u]+=np.dot(x, noise[u,i])
		ridge_user_f[u]=np.dot(np.linalg.pinv(user_v[u]), user_bias[u])
		ls_user_f[u]=np.dot(np.linalg.pinv(user_xx[u]), user_bias[u])
		average=np.dot(user_f.T, -normed_lap[u])+user_f[u]
		graph_user_f[u]=ridge_user_f[u]+0.1*np.dot(np.linalg.pinv(user_xx[u]), average)
		emp_error[u,i]=np.linalg.norm(graph_user_f[u]-user_f[u])
		xx_inv=np.linalg.pinv(user_xx[u])
		error_bound[u,i]=np.linalg.norm(xx_inv)*(0.1*np.linalg.norm(user_f[u]-average)+np.linalg.norm(user_noise[u]))
		x_long_matrix[u, u*dimension:(u+1)*dimension]=x 
	cov+=np.dot(x_long_matrix.T, x_long_matrix)
	bias+=np.dot(x_long_matrix.T, noisy_signal[:,i])
	graph_est=np.dot(np.linalg.pinv(cov), bias).reshape((user_num, dimension))
	graph_error=np.linalg.norm(graph_est-user_f)
	graph_error_list.extend([graph_error])
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
	global_error_bound_list.extend([alpha*(rank+fitness)/(kappa+lambda_2)])


plt.figure(figsize=(5,5))
#plt.plot(np.sum(emp_error, axis=0)[10:], label='emp error')
plt.plot(graph_error_list[10:], label='Emperical Error')
plt.plot(global_error_bound_list[10:], '.-', markevery=0.2, label='Global Error Bound')
plt.plot(np.sum(error_bound, axis=0)[10:], '*-', markevery=0.2, label='Indiv Error Bound (Sum)')
#plt.title('Emperical Error vs Error Bound')
plt.xlabel('Sample size', fontsize=12)
plt.ylabel('estimation error (MSE)', fontsize=12)
plt.legend(loc=0, fontsize=12)
plt.show()
