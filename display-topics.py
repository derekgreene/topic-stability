#!/usr/bin/env python
from optparse import OptionParser
import unsupervised.rankings, unsupervised.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] ranking_file1 ranking_file2 ...")
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to show", default=10)
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one ranking set file" )

	# Load each cached ranking set
	for in_path in args:
		print "Loading terms from %s ..." % in_path
		(term_rankings,labels) = unsupervised.util.load_term_rankings( in_path )
		m = unsupervised.rankings.term_rankings_size( term_rankings )
		print "Set has %d rankings covering up to %d terms" % ( len(term_rankings), m ) 
		print unsupervised.rankings.format_term_rankings( term_rankings, labels, min(options.top,m) )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
