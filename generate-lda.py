#!/usr/bin/env python
import os, sys, random
from optparse import OptionParser
import numpy as np
import text.util, unsupervised.lda, unsupervised.rankings, unsupervised.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] corpus_file")
	parser.add_option("--seed", action="store", type="int", dest="seed", help="initial random seed", default=1000)
	parser.add_option("--kmin", action="store", type="int", dest="kmin", help="minimum number of topics", default=5)
	parser.add_option("--kmax", action="store", type="int", dest="kmax", help="maximum number of topics", default=5)
	parser.add_option("-r","--runs", action="store", type="int", dest="runs", help="number of runs", default=1)
	parser.add_option("--maxiters", action="store", type="int", dest="maxiter", help="maximum number of iterations", default=500)
	parser.add_option("-s", "--sample", action="store", type="float", dest="sample_ratio", help="sampling ratio of documents to include in each run (range is 0 to 1)", default=0.8)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="base output directory (default is current directory)", default=None)
	parser.add_option("-p", "--path", action="store", type="string", dest="mallet_path", help="path to Mallet 2 binary (required)", default=None)	
	parser.add_option("--reweight", action="store_true", dest="reweight_terms", help="reweight terms after apply LDA")
	(options, args) = parser.parse_args()
	if len(args) < 1:
		parser.error( "Must specify at least one corpus file" )	
	# Verify that we can find the Mallet binary.
	if options.mallet_path is None:
		parser.error( "Must specify path to Mallet 2 binary file using the option -p <file_path>" )	
	if not os.path.exists( options.mallet_path ):
		parser.error( "Cannot find specified Mallet 2 binary" )

	if options.dir_out is None:
		dir_out_base = os.getcwd()
	else:
		dir_out_base = options.dir_out

	# Load the cached corpus
	corpus_path = args[0]
	(X,terms,doc_ids,classes) = text.util.load_corpus( corpus_path )

	# Create implementation
	impl = unsupervised.lda.MalletLDA( options.mallet_path, top = min(100,len(terms)), max_iters = options.maxiter, reweight_terms = options.reweight_terms )

	n_documents = X.shape[0]
	n_sample = int( options.sample_ratio * n_documents )
	indices = np.arange(n_documents)
	print "* Sampling ratio = %.2f - %d/%d documents per run" % ( options.sample_ratio, n_sample, n_documents )

	# Generate all LDA topic models for the specified numbers of topics
	print "* Running experiments in range k=[%d,%d]" % ( options.kmin, options.kmax )
	for k in range(options.kmin, options.kmax+1):
		# Set random state
		np.random.seed( options.seed )
		random.seed( options.seed )			
		print "* Applying LDA k=%d runs=%d maxiters=%d (%s) ..." % ( k, options.runs, options.maxiter, impl.__class__.__name__ )
		dir_out_k = os.path.join( dir_out_base, "lda_k%02d" % k )
		if not os.path.exists(dir_out_k):
			os.makedirs(dir_out_k)		
		print "Results will be written to %s" % ( dir_out_k )
		# Run LDA
		for r in range(options.runs):
			print "Run %d/%d (seed=%s)" % (r+1, options.runs, options.seed )
			file_suffix = "%s_%03d" % ( options.seed, r+1 )
			# sub-sample data
			np.random.shuffle(indices)
			sample_indices = indices[0:n_sample]
			S = X[sample_indices,:]
			sample_doc_ids = []
			for doc_index in sample_indices:
				sample_doc_ids.append( doc_ids[doc_index] )
			# apply LDA, using a new random seed
			impl.seed = options.seed + r
			try:
				impl.apply( S, k )
			except Exception, error:	
				print str(error)
				print "Skipping LDA for run=%d k=%d" % (r,k)
				continue			
			# Get term rankings for each topic
			term_rankings = []
			for topic_index in range(k):		
				ranked_term_indices = impl.rank_terms( topic_index )
				term_ranking = [terms[i] for i in ranked_term_indices]
				term_rankings.append(term_ranking)
			print "Generated ranking set with %d topics covering up to %d terms" % ( len(term_rankings), unsupervised.rankings.term_rankings_size( term_rankings ) ) 
			# Write term rankings
			ranks_out_path = os.path.join( dir_out_k, "ranks_%s.pkl" % file_suffix )
			print "Writing term ranking set to %s" % ranks_out_path
			unsupervised.util.save_term_rankings( ranks_out_path, term_rankings )
			# Write document partition
			partition = impl.generate_partition()
			partition_out_path = os.path.join( dir_out_k, "partition_%s.pkl" % file_suffix )
			print "Writing document partition to %s" % partition_out_path
			unsupervised.util.save_partition( partition_out_path, partition, sample_doc_ids )			

	print "* Done"	  

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
