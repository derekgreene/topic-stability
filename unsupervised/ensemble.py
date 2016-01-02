import math, operator
import unsupervised.rankings

# --------------------------------------------------------------
# Relevance Scores Implementations
# --------------------------------------------------------------

class RelevanceFunction:
	""" Generic rank relevance function, with unit weights for all ranks. """

	def relevance( self, rank ):
		"""	Computes a score for the specified rank, which is indexed from 1 """
		return 1.0

class ReciprocalRankRelevance:
	""" Relevance function with inverse rank-weighted scores. """

	def relevance( self, rank ):
		"""	Computes a score for the specified rank, which is indexed from 1 """
		return 1.0/rank

	def __str__( self ):
		return "%s" % ( self.__class__.__name__ )

class LogRelevance:
	""" DCG-based rank relevance function, with inverse log-weighted scores. """

	def __init__( self, base = 2):
		self.base = base

	def relevance( self, rank ):
		"""	Computes a score for the specified rank, which is indexed from 1 """
		return 1.0/math.log( rank + 1, self.base )

	def __str__( self ):
		return "%s(base=%d)" % ( self.__class__.__name__, self.base )

# --------------------------------------------------------------
# Ensembles
# --------------------------------------------------------------

class EnsembleRanking:
	def __init__( self, rel_measure ):
		self.rel_measure = rel_measure
		self.weights = {}
		self.runs = 0

	def add( self, ranking ):
		for i in range(len(ranking)):
			self.weights[ranking[i]] = self.weights.get( ranking[i], 0 ) + self.rel_measure.relevance( i + 1 )
		self.runs += 1

	def build_ranking( self, top = -1, include_weights = False ):
		sx = sorted(self.weights.iteritems(), key=operator.itemgetter(1))
		sx.reverse()
		# include everything?
		if top > 0:
			actual_top = top
		else:
			actual_top = len(sx)
		ranking = []
		for p in range( actual_top ):
			if include_weights:
				w = sx[p][1] / self.runs
				ranking.append( "%s (%.2f)" % ( sx[p][0], w ) )
			else:
				ranking.append( sx[p][0] )
		return ranking

	def get_score( self, term ):
		return self.weights.get( term, 0.0 ) / self.runs

	def term_count( self ):
		return len(self.weights)

	def terms( self ):
		return self.weights.keys()


class TopicEnsemble:
	def __init__( self, rel_measure = ReciprocalRankRelevance() ):
		self.matcher = unsupervised.rankings.RankingSetAgreement( unsupervised.rankings.AverageJaccard() )
		self.rel_measure = rel_measure
		self.reference_rankings = None
		self.ensemble_rankings = []
		self.ensemble_consistency = []
		self.doc_ensemble = None

	def add( self, rankings, partition ):
		# first set of rankings?
		if self.reference_rankings is None:
			for ranking in rankings:
				er = EnsembleRanking( self.rel_measure )
				er.add( ranking )
				self.ensemble_rankings.append( er )
				self.ensemble_consistency.append( 0.0 )
			self.reference_rankings = rankings
		# otherwise, add to existing
		else:
			self.matcher.similarity( self.reference_rankings, rankings )
			for pair in self.matcher.results:
				reference_topic_index = pair[0]
				other_topic_index = pair[1]
				self.ensemble_consistency[reference_topic_index] += self.matcher.S[reference_topic_index,other_topic_index]
				self.ensemble_rankings[reference_topic_index].add( rankings[other_topic_index] )

	def build_rankings( self, top = 10, include_weights = False ):
		current_term_rankings = []
		for er in self.ensemble_rankings:
			current_term_rankings.append( er.build_ranking( top, include_weights ) )
		return current_term_rankings





