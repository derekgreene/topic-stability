#!/usr/bin/env python
"""
Tool to read in a pre-processed corpus stored in binary Joblib (PKL) format, and write out the data as a set of plain text files.

The output files have the following format:
- *.mtx: The document-term matrix, represented as a sparse coordinate matrix in Matrix Market format.
- *.terms: List of terms in the corpus, with each line corresponding to a column of the sparse data matrix.
- *.docs: List of document identifiers, with each line corresponding to a row of the sparse data matrix.
- *.labels: Assignment of documents to the "ground truth" label, where each line corresponds to a different class label.
"""
import os, os.path, sys, codecs
import logging as log
from optparse import OptionParser
import numpy as np
import text.util
import scipy.io

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] corpus_file")
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="base output directory (default is current directory)", default=None)
	(options, args) = parser.parse_args()
	if len(args) < 1:
		parser.error( "Must specify one corpus file" )	
	log.basicConfig(level=20, format='%(message)s')

	if options.dir_out is None:
		dir_out_base = os.getcwd()
	else:
		dir_out_base = options.dir_out

	# Load the cached corpus
	corpus_path = args[0]
	log.info("Converting corpus from file %s ..." % corpus_path)
	corpus_name = os.path.splitext( os.path.basename( corpus_path ) )[0]
	(X,terms,doc_ids,classes) = text.util.load_corpus( corpus_path )
	log.info( "Read existing document-term matrix: %d documents, %d terms" % (X.shape[0], X.shape[1]) )

	# Write the MTX file
	out_path = os.path.join( dir_out_base, "%s.mtx" % corpus_name )
	scipy.io.mmwrite( out_path, X )
	log.info( "Wrote document-term matrix in MTX format to %s" % out_path )

	# Write the terms
	out_path = os.path.join( dir_out_base, "%s.terms" % corpus_name )
	with codecs.open(out_path, 'w', encoding="utf8", errors='ignore') as fout:
		for term in terms:
			fout.write("%s\n" % term)
	log.info( "Wrote %d terms to %s" % ( len(terms), out_path ) )

	# Write the document IDs
	out_path = os.path.join( dir_out_base, "%s.docs" % corpus_name )
	with codecs.open(out_path, 'w', encoding="utf8", errors='ignore') as fout:
		for doc_id in doc_ids:
			fout.write("%s\n" % doc_id)
	log.info( "Wrote %d document IDs to %s" % ( len(doc_ids), out_path ) )

	# Write the class labels, if any
	if not classes is None:
		out_path = os.path.join( dir_out_base, "%s.labels" % corpus_name )
		with codecs.open(out_path, 'w', encoding="utf8", errors='ignore') as fout:
			for class_name in classes.keys():
				fout.write("%s:" % class_name )
				fout.write( ",".join( classes[class_name] ) )
				fout.write("\n")
		log.info( "Wrote class labels for %d document IDs to %s" % ( len(doc_ids), out_path ) )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
