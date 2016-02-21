import math, string
import numpy as np
from prettytable import PrettyTable
import unsupervised.hungarian

# --------------------------------------------------------------
# Ranking Similarity 
# --------------------------------------------------------------

class JaccardBinary:
	""" 
	Simple binary Jaccard-based ranking comparison, which does not take into account rank positions. 
	"""
	def similarity( self, gold_ranking, test_ranking ):
		sx = set(gold_ranking)
		sy = set(test_ranking)
		numer = len( sx.intersection(sy) )
		if numer == 0:
			return 0.0
		denom = len( sx.union(sy) )
		if denom == 0:
			return 0.0
		return float(numer)/denom

	def __str__( self ):
		return "%s" % ( self.__class__.__name__ )

class AverageJaccard(JaccardBinary):
	""" 
	A top-weighted version of Jaccard, which takes into account rank positions. 
	This is based on Fagin's Average Overlap Intersection Metric.
	"""
	def similarity( self, gold_ranking, test_ranking ):
		k = min( len(gold_ranking), len(test_ranking) )
		total = 0.0
		for i in range(1,k+1):
			total += JaccardBinary.similarity( self, gold_ranking[0:i], test_ranking[0:i] )
		return total/k

# --------------------------------------------------------------
# Ranking Set Agreement
# --------------------------------------------------------------

class RankingSetAgreement:
	"""
	Calculates the agreement between pairs of ranking sets, using a specified measure of 
	similarity between rankings.
	"""
	def __init__( self, metric = AverageJaccard() ):
		self.metric = metric

	def similarity( self, rankings1, rankings2 ):
		"""
		Calculate the overall agreement between two different ranking sets. This is given by the
		mean similarity values for all matched pairs.
		"""
		self.results = None
		self.S = self.build_matrix( rankings1, rankings2 )
		score, self.results = self.hungarian_matching()
		return score

	def build_matrix( self, rankings1, rankings2 ):
		"""
		Construct the similarity matrix between the pairs of rankings in two 
		different ranking sets.
		"""
		rows = len(rankings1)
		cols = len(rankings2)
		S = np.zeros( (rows,cols) )
		for row in range(rows):
			for col in range(cols):
				S[row,col] = self.metric.similarity( rankings1[row], rankings2[col] )
		return S	

	def hungarian_matching( self ):
		"""
		Solve the Hungarian matching problem to find the best matches between columns and rows based on
		values in the specified similarity matrix.
		"""
		# apply hungarian matching
		h = unsupervised.hungarian.Hungarian()
		C = h.make_cost_matrix(self.S)
		h.calculate(C)
		results = h.get_results()
		# compute score based on similarities
		score = 0.0
		for (row,col) in results:
			score += self.S[row,col]
		score /= len(results)
		return (score, results)

# --------------------------------------------------------------
# Utilities
# --------------------------------------------------------------

def calc_relevance_scores( n, rel_measure ):
	""" 
	Utility function to compute a sequence of relevance scores using the specified function.
	"""
	scores = []
	for i in range(n):
		scores.append( rel_measure.relevance( i + 1 ) )
	return scores

def term_rankings_size( term_rankings ):
	"""
	Return the number of terms covered by a list of multiple term rankings.
	"""
	m = 0
	for ranking in term_rankings:
		if m == 0:
			m = len(ranking)
		else:
			m = min( len(ranking), m ) 
	return m

def truncate_term_rankings( orig_rankings, top ):
	"""
	Truncate a list of multiple term rankings to the specified length.
	"""
	if top < 1:
		return orig_rankings
	trunc_rankings = []
	for ranking in orig_rankings:
		trunc_rankings.append( ranking[0:min(len(ranking),top)] )
	return trunc_rankings

def format_term_rankings( term_rankings, labels = None, top = 10 ):
	"""
	Format a list of multiple term rankings using PrettyTable.
	"""
	from prettytable import PrettyTable
	# add header
	header = ["Rank"]
	if labels is None:
		for i in range( len(term_rankings) ):
			header.append("C%02d" % (i+1) )	
	else:
		for label in labels:
			header.append(label)	
	tab = PrettyTable(header)
	for field in header:
		tab.align[field] = "l"
	# add body
	for pos in range(top):
		row = [ str(pos+1) ]
		for ranking in term_rankings:
			# have we run out of terms?
			if len(ranking) <= pos:
				row.append( "" ) 
			else:
				row.append( ranking[pos] ) 
		tab.add_row( row )
	return tab

def format_term_rankings_long( term_rankings, labels = None, top = 10 ):
	"""
	Format a list of multiple term rankings using lists.
	"""
	if labels is None:
		labels = []
		for i in range( len(term_rankings) ):
			labels.append("C%02d" % (i+1) )	
	max_label_len = 0
	for label in labels:
		max_label_len = max(max_label_len,len(label))
	max_label_len += 1

	s = ""
	for i, label in enumerate(labels):
		s += label.ljust(max_label_len)
		s += ": "
		sterms = ""
		for term in term_rankings[i][0:top]:
			if len(sterms) > 0:
				sterms += ", "
			sterms += term
		s += sterms + "\n"
	return s

