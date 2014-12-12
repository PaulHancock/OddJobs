#! /usr/bin/env python

"""
fits_trim.py

A tool that does two things:
1 - converts pixels that are identically zero into masked pixels
2 - trims the fits image to the smallest rectangle that still contains all the data pixels

Author: Paul Hancock
Nov-2014
"""


import numpy as np
from astropy.io import fits
import sys

def main(fin,fout):
	"""
	"""
	hdulist = fits.open(fin)
	data = hdulist[0].data
	dshape = data.shape
	#remove axes that are empty
	data=data.squeeze()
	#turn pixels that are identically zero, into masked pixels
	data[np.where(data==0.)]=np.nan

	imin,imax=0,data.shape[1]-1
	jmin,jmax=0,data.shape[0]-1
	#select [ij]min/max to exclude rows/columns that are all zero
	for i in xrange(0,imax):
		if np.all(np.isnan(data[:,i])):
			imin=i
		else:
			break
	print imin,
	for i in xrange(imax,imin,-1):
		if np.all(np.isnan(data[:,i])):
			imax=i
		else:
			break
	print imax,
	for j in xrange(0,jmax):
		if np.all(np.isnan(data[j,:])):
			jmin=j
		else:
			break
	print jmin,
	for j in xrange(jmax,jmin,-1):
		if np.all(np.isnan(data[j,:])):
			jmax=j
		else:
			break
	print jmax

	data = data[jmin:jmax,imin:imax]
	#remember the shape of the data
	np.resize(data,dshape)
	hdulist[0].data = data
	#recenter the image so the coordinates are correct.
	hdulist[0].header['CRPIX1']-=imin
	hdulist[0].header['CRPIX2']-=jmin
	#save
	hdulist.writeto(fout,clobber=True)
	print "wrote",fout


if __name__ == '__main__':
	if len(sys.argv)>2:
		fin = sys.argv[-2]
		fout = sys.argv[-1]
		print fin,"=>",fout,"(mask and trim)"
	else:
		print "usage: python {0} infile.fits outfile.fits".format(__file__)
		sys.exit()
	main(fin,fout)