import os, os.path, tempfile, shutil
import logging as log
from subprocess import call
import numpy as np
from scipy.stats.mstats import gmean

class MalletLDA:
	"""
	Wrapper class for Mallet. Requires that a binary version of Mallet 2.0 is available locally.
	"""
	def __init__( self, mallet_path, top = 100, seed = 1000, max_iters = 1000, alpha = 10.0, beta = 0.01, rerank_terms = False ):
		# settings
		self.mallet_path = mallet_path
		self.max_iters = max_iters
		self.num_threads = 4
		self.seed = seed
		self.top = top
		self.lda_alpha = alpha
		self.lda_beta = beta
		self.optimize_interval = 10
		self.rerank_terms = rerank_terms
		self.delete_temp_files = True
		# state
		self.partition = None

	def apply( self, X, k = 2 ):
		"""
		Apply topic modeling to the specific document-term matrix, using K topics.
		"""
		self.partition = None
		self.topic_rankings = None
		# create Mallet corpus
		dir_tmp = tempfile.mkdtemp()
		corpus_path = self.__write_documents( X, dir_tmp )
		mallet_data_path = self.__import_data( corpus_path, dir_tmp )
		if not os.path.exists( mallet_data_path ):
			raise Exception("Error: Failed to import data into Mallet format")
		# run Mallet
		mallet_terms_path, mallet_docs_path, mallet_weights_path = self.__run_mallet( k, mallet_data_path, dir_tmp )
		if not ( os.path.exists( mallet_terms_path ) and os.path.exists( mallet_docs_path ) and os.path.exists( mallet_weights_path ) ):
			raise Exception("Error: Failed to correctly run Mallet")
		# any pre-processing required?
		if self.rerank_terms:
			self.topic_rankings = self.__rerank_terms( X.shape[1], k, mallet_weights_path )
		else:
			self.topic_rankings = self.__parse_topics( mallet_terms_path )
		log.debug( "Generated ranking set with %d topic rankings" % len(self.topic_rankings) )
		self.partition =  self.__parse_document_weights( X.shape[0], mallet_docs_path )
		# now tidy up, if required
		if self.delete_temp_files:
			try:
				log.debug( "Removing temporary directory %s" % dir_tmp )
				shutil.rmtree(dir_tmp)
			except OSError as e:
				log.warning( "Failed to remove temporary directory - %s" % str(e) )

	def rank_terms( self, topic_index, top = -1 ):
		"""
		Return the top ranked terms for the specified topic, generated during the last LDA run.
		"""
		if self.topic_rankings is None:
			raise ValueError("No results for previous run available")
		if len(self.topic_rankings[topic_index]) < top:
			return self.topic_rankings[topic_index]
		return self.topic_rankings[topic_index][0:top]

	def generate_partition( self ):
		if self.partition is None:
			raise ValueError("No results for previous run available")
		return self.partition

	def __write_documents( self, X, dir_tmp ):
		"""
		Write documents to temporary file, for parsing by Mallet.
		"""
		log.debug( "Writing temporary files to %s" % dir_tmp )
		# Write documents, one per line
		corpus_path = os.path.join( dir_tmp, "corpus.txt" )
		f = open( corpus_path, "w")
		for row in range( X.shape[0] ):
			v = X.getrow( row )
			doc_tokens = []
			for pos in range(v.nnz):
				# just in case the data has been normalized...
				freq = max( 1, int(v.data[pos]) )
				token = "%d" % v.indices[pos]
				for i in range(freq):
					doc_tokens.append( token )
			f.write( " ".join( doc_tokens ) )
			f.write("\n")
		f.close()
		return corpus_path

	def __import_data( self, corpus_path, dir_tmp ):
		"""
		Run the Mallet pre-processing step.
		"""
		log.debug( "Importing data into Mallet format... " )
		mallet_data_path = os.path.join( dir_tmp, "corpus.mallet" )
		mallet_cmd = "%s import-file --keep-sequence --token-regex '\S+' --input %s --output %s" % ( self.mallet_path, corpus_path, mallet_data_path )
		call(mallet_cmd, shell=True)
		return mallet_data_path

	def __run_mallet( self, k, mallet_data_path, dir_tmp ):
		"""
		Run the Mallet LDA step.
		"""
		mallet_terms_path = os.path.join( dir_tmp, "topic_terms.txt" )
		mallet_docs_path = os.path.join( dir_tmp, "topic_docs.txt" )
		mallet_weights_path = os.path.join( dir_tmp, "topic_term_weights.txt" )
		mallet_args = [self.mallet_path,
			'train-topics',
			'--input',
			mallet_data_path,
			'--output-topic-keys',
			mallet_terms_path,
			'--output-doc-topics',
			mallet_docs_path,
			'--topic-word-weights-file',
			mallet_weights_path,
			'--num-topics',
			k, 
			'--num-iterations', 
			self.max_iters,
			'--num-top-words', 
			self.top,
			'--show-topics-interval 1000', 
			'--optimize-interval ', 
			self.optimize_interval,
			'--optimize-burn-in 200', 
			'--use-symmetric-alpha false', 
			'--alpha',
			self.lda_alpha,
			'--beta',
			self.lda_beta, 
			'--num-threads ',
			self.num_threads,
			'--random-seed',
			self.seed
			]
		log.debug( "Running Mallet (k=%d alpha=%.3f beta=%.3f optimize_interval=%d threads=%d seed=%s)... " % ( k, self.lda_alpha, self.lda_beta, self.optimize_interval, self.num_threads, self.seed ) )
		mallet_cmd = ' '.join([str(x) for x in mallet_args])
		#log.debug( mallet_cmd )
		call(mallet_cmd, shell=True)
		return (mallet_terms_path, mallet_docs_path, mallet_weights_path)

	def __parse_topics( self, mallet_terms_path ):
		rankings = []
		with open(mallet_terms_path) as f:
			lines = [line.rstrip() for line in f]
			for l in lines:
				if len(l) == 0 or l.startswith("#"):
					continue
				parts = l.split("\t")
				ranking = []
				if len(parts) > 2:
					for token in parts[2].strip().split(" "):
						term_index = int(token)
						ranking.append( term_index )
				rankings.append( ranking )
		return rankings

	def __parse_document_weights( self, num_docs, mallet_docs_path ):
		partition = list(0 for i in range(num_docs)) 
		log.debug("Reading LDA document weights from %s" % mallet_docs_path)
		with open(mallet_docs_path) as f:
			lines = [line.rstrip() for line in f]
			for l in lines:
				if len(l) == 0 or l.startswith("#"):
					continue
				parts = l.split("\t")
				if len(parts) < 3:
					continue
				doc_index = int(parts[0])
				# find the topic with the max weight
				weights = []
				for p in parts[2:]:
					weights.append( float(p) )
				weights = np.array(weights)
				partition[doc_index] = weights.argmax()
		return partition

	def __rerank_terms( self, num_terms, k, mallet_weights_path, eps = 1e-9 ):
		"""
		Implements the term re-weighting method proposed by Blei and Lafferty.
		"""
		log.debug( "Reweighting terms  ..." )
		# Parse weights for all terms and topics
		W = np.zeros( (num_terms, k) )
		with open(mallet_weights_path) as f:
			lines = [line.rstrip() for line in f]
			for l in lines:
				if len(l) == 0 or l.startswith("#"):
					continue
				parts = l.split("\t")
				if len(parts) < 3:
					continue
				topic_index = int(parts[0])
				term_index = int(parts[1])		
				W[term_index,topic_index] = float(parts[2])
		# Calculate geometric means
		gmeans = gmean( W, axis = 1 )
		# Reweight the terms
		# TODO: vectorize this
		for row in range(num_terms):
			if gmeans[row] <= eps or np.isnan( gmeans[row] ):
				continue
			for col in range(k):
				if W[row,col] <= eps:
					continue
				lp = np.log( W[row,col] / gmeans[row] )
				if np.isnan( lp ):
					continue
				W[row,col] = W[row,col] * lp
		# Get new term rankings per topic
		rankings = []
		for topic_index in range(k):
			ranking = np.argsort( W[:,topic_index] )[::-1].tolist()
			rankings.append( ranking )
		return rankings


