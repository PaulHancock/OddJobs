#! /usr/bin/env python

import sys

def main(infile,outfile):
	import aplpy

	gc = aplpy.FITSFigure(infile,figsize=(10,10))
	gc.show_colorscale(cmap='cubehelix',stretch='linear',vmin=-2,vmax=10)
	gc.show_colorbar()
	gc.colorbar.set_ticks([-2,0,2,4,6,8,10])
	gc.set_nan_color('gray')

	gc.add_grid()
	gc.grid.set_alpha(0.3)
	gc.grid.set_xspacing(15)
	gc.grid.set_yspacing(10)

	gc.set_tick_labels_font(size='small')
	gc.set_tick_labels_format(xformat='hh:mm',yformat='dd:mm')
	gc.save('out.png')
	gc.save('out.png')

def main2(infile,outfile):
	from astropy.io import fits
	from matplotlib import pyplot
	import numpy as np
	import os
	image = np.flipud(np.squeeze(fits.open(infile)[0].data))
	minmax = np.percentile(image.ravel(),[0.1,99.9])
	figure = pyplot.figure(figsize=(5,5))
	ax=figure.add_subplot(111)
	cbax = ax.imshow(image,cmap=pyplot.cm.cubehelix,vmin=minmax[0],vmax=minmax[1],interpolation='nearest')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title(os.path.basename(infile))
	cb=pyplot.colorbar(mappable=cbax,orientation='vertical',shrink=0.75)
	cb.set_label('Jy')
	figure.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95)
	pyplot.savefig(outfile)


if __name__ == '__main__':
	if not len(sys.argv)==3:
		print sys.argv
		print "need two files, infile and outfile"
		sys.exit()
	infile,outfile = sys.argv[-2:]
	main2(infile,outfile)
