#! /usr/bin/env python	

import sys
import numpy as np
from astropy.io import fits


def main(infile):
	outfile = infile+"fixed.fits"
	data = fits.open(infile)
	numaxes=0
	for k in data[0].header:
		if k.startswith("CRVAL"):
			numaxes = max(numaxes, int(k[-1]))
		if k.startswith("HIST"):
			break
	if int(data[0].header['NAXIS']) != numaxes :
		print "claims to have {0} axes, but there are {1} in the fits header".format(data[0].header['NAXIS'],numaxes)
		shape = data[0].data.shape
		newshape = np.ones(numaxes)
		for i in xrange(1,len(shape)+1):
			newshape[-i] = shape[-i]
		data[0].data = np.resize(data[0].data,newshape)
		print "fixed"	
		data.writeto(infile,clobber=True)
	else:
		print "file doesn't appear to need fixing"


if __name__ == '__main__':
	if len(sys.argv)>1:
		infile = sys.argv[-1]
	else:
		print "Usage: python fixhead.py file.fits"
		sys.exit()
	main(infile)