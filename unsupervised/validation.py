from prettytable import PrettyTable
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
import util, rankings

# --------------------------------------------------------------

class TermValidator:
	"""
	Validation measure, which compares the agreement between term rankings derived from the
	centroids produced using a 'ground truth' partition, with a specified set of test rankings
	generated on the same corpus.
	"""
	def __init__( self, X, terms, class_partition ):
		self.agreement_measure = rankings.RankingSetAgreement()
		centroids = util.build_centroids( X, class_partition, max(class_partition) + 1 )
		# sort centroids for each class
		self.class_rankings = []
		for class_index, centroid in enumerate(centroids):
			# build ranking of terms 
			ranking = [terms[i] for i in centroid.argsort()]
			ranking.reverse()
			self.class_rankings.append( ranking )

	def evaluate( self, test_rankings, top_values = [10] ):
		scores = {}
		for top in top_values:
			trunc_classes = rankings.truncate_term_rankings(self.class_rankings,top)
			trunc_test = rankings.truncate_term_rankings(test_rankings,top)
			sim = self.agreement_measure.similarity( trunc_classes, trunc_test )
			scores[ "terms-%03d" % (top) ] = sim
		return scores

# --------------------------------------------------------------

class DiversityValidator:
	"""
	Validation measure that calculates the average pairwise dissimilarity between 
	all unique pairs of term rankings in a given ranking set. 
	The default measure for calculate (dis)similarity is the Average Jaccard metric.
	"""
	def __init__( self, metric = rankings.AverageJaccard() ):
		self.metric = metric

	def evaluate( self, test_rankings, top_values = [10] ):
		scores = {}
		k = len(test_rankings)
		for top in top_values:
			trunc_rankings = rankings.truncate_term_rankings( test_rankings, top )
			pairs = 0
			diversity = 0.0
			for ranking_index1 in range(k):
				for ranking_index2 in range(ranking_index1 + 1, k):
					pair_dissimilarity = 1.0 - self.metric.similarity( trunc_rankings[ranking_index1], trunc_rankings[ranking_index2] )
					diversity += pair_dissimilarity
					pairs += 1
			scores[ "div-%03d" % (top) ] = diversity / pairs
		return scores

# --------------------------------------------------------------

class PartitionValidator:
	"""
	A validator that evaluates topic (cluster) memberships based on an external set of ground truth classes.
	Note: we assume both the topic and class memberships are disjoint (non-overlapping).
	"""
	def __init__( self, classes, doc_ids ):
		self.classes = classes
		self.doc_ids = doc_ids
		self.class_map = {}
		class_index = 0
		# convert classes to membership map
		for class_id in classes.keys():
			for doc_id in classes[class_id]:
				self.class_map[doc_id] = class_index
			class_index += 1

	def has_class_info( self ):
		return not( self.classes is None or len(self.classes) < 2 )

	def evaluate( self, partition, clustered_ids ):
		# no class info?
		if not self.has_class_info():
			return {}
		# get two clusterings that we can compare
		n = len(clustered_ids)
		classes_subset = np.zeros( n )
		for row in range(n):
			classes_subset[row] = self.class_map[clustered_ids[row]]		
		scores = {}
		scores["external-nmi"] = normalized_mutual_info_score( classes_subset, partition )
		scores["external-ami"] = adjusted_mutual_info_score( classes_subset, partition )
		scores["external-ari"] = adjusted_rand_score( classes_subset, partition )
		return scores

	def keys( self ):
		# no class info?
		if not self.has_class_info():
			return set()
		return set( "external-nmi", "external-ami", "external-ari" )

# --------------------------------------------------------------

class ScoreCollection:
	"""
	A utility class for keeping track of experiment scores produced by multiple validation measures 
	applied to different topic models.
	"""
	def __init__( self ):
		self.all_scores = {}
		self.all_score_keys = set()

	def add( self, experiment_key, scores ):
		for score_key in scores.keys():
			self.all_score_keys.add( score_key )
		self.all_scores[experiment_key] = scores

	def aggregate_scores( self ):
		if len(self.all_scores) == 0:
			return []
		vectors = {}
		for score_key in self.all_score_keys:
			vectors[score_key] = []
		for experiment_key in self.all_scores.keys():
			for score_key in self.all_scores[experiment_key].keys():
				vectors[score_key].append( self.all_scores[experiment_key][score_key] )
		mean_scores = {}
		std_scores = {}
		for score_key in self.all_score_keys:
			v = np.array( vectors[score_key] )
			mean_scores[score_key] = np.mean(v)
			std_scores[score_key] = np.std(v)
		return (mean_scores,std_scores)

	def create_table( self, include_mean = False, precision = 2 ):
		fmt = "%%.%df" % precision
		header = ["experiment"]
		score_keys = list(self.all_score_keys)
		score_keys.sort()
		header += score_keys
		tab = PrettyTable( header )
		tab.align["experiment"] = "l"
		experiment_keys = list( self.all_scores.keys() )
		experiment_keys.sort()
		for experiment_key in experiment_keys:
			row = [ experiment_key ]
			for score_key in score_keys:
				row.append( fmt % self.all_scores[experiment_key].get( score_key, 0.0 ) )
			tab.add_row( row )
		if include_mean:
			mean_scores, std_scores = self.aggregate_scores()
			row = [ "MEAN" ]
			for score_key in score_keys:
				row.append( fmt % mean_scores.get( score_key, 0.0 ) )
			tab.add_row( row )
		return tab 
