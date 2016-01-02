import numpy as np
from scipy import sparse as sp
# note that we use the scikit-learn bundled version of joblib
from sklearn.externals import joblib

# --------------------------------------------------------------

def build_centroids( X, partition, k ):
	"""
	Build a set of K centroids based on the specified partition memberships.
	"""
	# NB: need to convert to a numpy array before we do this...
	memberships = np.array(partition)
	n_features = X.shape[1]
	centroids = np.empty((k, n_features), dtype=np.float64)
	for i in range(k):
		center_mask = memberships == i
		if sp.issparse(X):
			center_mask = np.where(center_mask)[0]
		centroids[i] = X[center_mask].mean(axis=0)
	return centroids

def clustermap_to_partition( cluster_map, doc_ids ):
	"""
	Convert a dictionary representing a clustering into a partition.
	"""
	cluster_names = list(cluster_map.keys())
	cluster_names.sort()
	# build document map
	partition = []
	doc_map = {}
	for i in range( len(doc_ids) ):
		partition.append( -1 )
		doc_map[ doc_ids[i] ] = i
	# now create partition
	for cluster_index in range( len(cluster_map) ):
		for doc_id in cluster_map[cluster_names[cluster_index]]:
			partition[doc_map[doc_id]] = cluster_index
	return partition
	
# --------------------------------------------------------------

def save_term_rankings( out_path, term_rankings, labels = None ):
	"""
	Save a list of multiple term rankings using Joblib.
	"""
	# no labels? generate some standard ones
	if labels is None:
		labels = []
		for i in range( len(term_rankings) ):
			labels.append( "C%02d" % (i+1) )
	joblib.dump((term_rankings,labels), out_path ) 

def load_term_rankings( in_path ):
	"""
	Load a list of multiple term rankings using Joblib.
	"""
	#print "Loading term rankings from %s ..." % in_path
	(term_rankings,labels) = joblib.load( in_path )
	return (term_rankings,labels)

def save_nmf_factors( out_path, W, H, doc_ids ):
	"""
	Save a NMF factorization result using Joblib.
	"""
	joblib.dump((W,H,doc_ids), out_path ) 

def load_nmf_factors( in_path ):
	"""
	Load a NMF factorization result using Joblib.
	"""
	(W,H,doc_ids) = joblib.load( in_path )
	return (W,H,doc_ids)

def save_partition( out_path, partition, doc_ids ):
	"""
	Save a disjoint partition (clustering) result using Joblib.
	"""
	joblib.dump((partition,doc_ids), out_path ) 


def load_partition( in_path):
	"""
	Load a disjoint partition (clustering) result using Joblib.
	"""
	(partition,doc_ids) = joblib.load( in_path )
	return (partition,doc_ids) 
