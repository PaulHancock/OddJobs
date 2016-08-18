#! python

__author__ = "PaulHancock"
__date__ = "18/08/2016"

import numpy as np
import os
from scipy import interpolate
from scipy.interpolate import LinearNDInterpolator
from astropy import wcs
from astropy.io import fits
import sys
import argparse


def make_pix_models(fname, ra1='ra', dec1='dec', ra2='RAJ2000', dec2='DEJ2000', fitsname=None, plots=False):
    """
    Read a fits file which contains the crossmatching results for two catalogues.
    Catalogue 1 is the source catalogue (positions that need to be corrected)
    Catalogue 2 is the reference catalogue (correct positions)
    return rbf models for the ra/dec corrections
    :param fname: Filename
    :param ra1: column name for the ra degrees in catalogue 1 (source)
    :param dec1: column name for the dec degrees in catalogue 1 (source)
    :param ra2: column name for the ra degrees in catalogue 2 (reference)
    :param dec2: column name for the dec degrees in catalogue 2 (reference)
    :param plots: True = Make plots
    :return: (dramodel, ddecmodel)
    """
    raw_data = fits.open(fname)[1].data

    # get the wcs
    hdr = fits.getheader(fitsname)
    imwcs = wcs.WCS(hdr, naxis=2)

    # filter the data to only include SNR>10 sources
    flux_mask = np.where(raw_data['peak_flux']/raw_data['local_rms']>10)
    data = raw_data[flux_mask]

    #calculate the offsets in the ra/dec directions
    # catalog = Longitude(data[ra1], unit=u.degree), Latitude(data[dec1], unit=u.degree)
    # reference = Longitude(data[ra2], unit=u.degree), Latitude(data[dec2], unit=u.degree)

    cat_xy = imwcs.all_world2pix(zip(data[ra1], data[dec1]), 1)
    ref_xy = imwcs.all_world2pix(zip(data[ra2], data[dec2]), 1)

    diff_xy = ref_xy - cat_xy

    dxmodel = interpolate.Rbf(cat_xy[:, 0], cat_xy[:, 1], diff_xy[:, 0], function='linear', smooth=3)
    dymodel = interpolate.Rbf(cat_xy[:, 0], cat_xy[:, 1], diff_xy[:, 1], function='linear', smooth=3)

    if plots:
        from matplotlib import pyplot
        xmin, xmax = 0, hdr['NAXIS1']
        ymin, ymax = 0, hdr['NAXIS2']

        gx, gy = np.mgrid[xmin:xmax:(xmax-xmin)/20., ymin:ymax:(ymax-ymin)/20.]
        mdx = dxmodel(np.ravel(gx), np.ravel(gy))
        mdy = dymodel(np.ravel(gx), np.ravel(gy))

        x = cat_xy[:, 0]
        y = cat_xy[:, 1]
        dx = diff_xy[:, 0]
        dy = diff_xy[:, 1]

        fig = pyplot.figure(figsize=(14, 7))
        kwargs = {'angles':'xy', 'scale_units':'xy', 'scale':1, 'cmap':'hsv'} #, 'vmin':-180, 'vmax':180}
        angles = np.degrees(np.arctan2(dy, dx))
        ax = fig.add_subplot(1, 2, 1)
        cax = ax.quiver(x, y, dx*60, dy*60, angles, **kwargs)
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.set_title("Offsets")
        cbar = fig.colorbar(cax, orientation='horizontal')

        ax = fig.add_subplot(1, 2, 2)
        cax = ax.quiver(gx, gy, mdx*60, mdy*60, np.degrees(np.arctan2(mdy, mdx)), **kwargs)
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.set_title("model vectors")
        cbar = fig.colorbar(cax, orientation='horizontal')

        outname = os.path.splitext(fname)[0] +'.png'
        pyplot.savefig(outname, dpi=200)

    return dxmodel, dymodel


def correct_image(fname, dxmodel, dymodel, fout):
    """
    Read a fits image, and apply pixel-by-pixel corrections based on the
    given x/y models. Interpolate back to a regular grid, and then write an
    output file
    :param fname: input fits file
    :param dxodel: x model
    :param dymodel: x model
    :param fout: output fits file
    :return: None
    """
    im = fits.open(fname)
    data = np.squeeze(im[0].data)
    # create a map of (x,y) pixel pairs as a list of tuples
    xy = np.indices(data.shape, dtype=np.float32)
    xy.shape = (2, xy.shape[1]*xy.shape[2])

    x = np.array(xy[1, :])
    y = np.array(xy[0, :])

    # calculate the corrections in blocks of 100k since the rbf fails on large blocks
    print 'applying corrections',
    if len(x) > 100000:
        print 'in cycles'
        borders = range(0, len(x)+1, 100000)
        if borders[-1] != len(x):
            borders.append(len(x))
        for s1 in [slice(a, b) for a, b in zip(borders[:-1], borders[1:])]:
            x[s1] += dxmodel(x[s1], y[s1])
            # the x coords were just changed so we need to refer back to the original coords
            y[s1] += dymodel(xy[1, :][s1], y[s1])
    else:
        print 'all at once'
        x += dxmodel(x, y)
        y += dymodel(xy[1, :], y)

    print 'interpolating'
    ifunc = LinearNDInterpolator(np.transpose([x,y]), np.ravel(data))
    interpolated_map = ifunc(xy[1, :], xy[0, :])

    print 'writing'
    interpolated_map.shape = data.shape
    im[0].data = interpolated_map
    im.writeto(fout, clobber=True)
    print "wrote", fout
    return


def correct_catalogue(fname, dramodel, ddecmodel, fout):
    """
    Read a fits image, and apply pixel-by-pixel ra/dec corrections based on the
    given dra/ddec models. Interpolate back to a regular grid, and then write an
    output file
    :param fname: input fits file
    :param dramodel: ra model (in degrees)
    :param ddecmodel: dec model (in degrees)
    :param fout: output fits file
    :return: None
    """
    im = fits.open(fname)
    data = np.squeeze(im[0].data)
    imwcs = wcs.WCS(im[0].header, naxis=2)
    # create a map of (x,y) pixel pairs as a list of tuples
    xy = np.indices(data.shape[-2:])
    xy.shape = (2, xy.shape[1]*xy.shape[2])
    # convert x/y -> ra/dec
    print "pix2world"
    pos = imwcs.all_pix2world(xy.T, 1)
    ra = pos[:, 0]
    dec = pos[:, 1]

    # calculate the corrections in blocks of 100k since the rbf fails on large blocks
    print 'applying corrections'
    if len(ra) > 100000:
        borders = range(0, len(ra)+1, 100000)
        if borders[-1] != len(ra):
            borders.append(len(ra))
        for s1 in [slice(a, b) for a, b in zip(borders[:-1], borders[1:])]:
            ra[s1] += dramodel(ra[s1], dec[s1]) * scale
            dec[s1] += ddecmodel(ra[s1], dec[s1]) * scale
    else:
        ra += dramodel(ra, dec)
        dec += ddecmodel(ra, dec)

    # since ra/dec are views into pos, updaing ra/dec updates pos

    print 'world2pix'
    cxy = imwcs.all_world2pix(pos, 1)

    print 'interpolating'
    ifunc = LinearNDInterpolator(cxy, np.ravel(data))
    interpolated_map = ifunc(xy[0, :], xy[1, :])

    print 'writing'
    interpolated_map.shape = data.shape
    im[0].data = interpolated_map
    im.writeto(fout, clobber=True)
    print "wrote", fout
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group1 = parser.add_argument_group("input/output files")
    group1.add_argument("--xm", dest='xm', type=str, default=None,
                        help='A .fits binary table. The crossmatch between the reference and source catalogue.')
    group1.add_argument("--infits", dest='infits', type=str, default=None,
                        help="The fits image that is to be corrected.")
    group1.add_argument("--outfits", dest='outfits', type=str, default=None,
                        help="The output (corrected) fits image.")
    group2 = parser.add_argument_group("catalog column names")
    group2.add_argument("--ra1", dest='ra1', type=str, default='ra',
                        help="The column name for ra  (degrees) for source catalogue.")
    group2.add_argument("--dec1", dest='dec1', type=str, default='dec',
                        help="The column name for dec (degrees) for source catalogue.")
    group2.add_argument("--ra2", dest='ra2', type=str, default='RAJ2000',
                        help="The column name for ra  (degrees) for reference catalogue.")
    group2.add_argument("--dec2", dest='dec2', type=str, default='DEJ2000',
                        help="The column name for dec (degrees) for reference catalogue.")
    group3 = parser.add_argument_group("Other")
    group3.add_argument('--plot', dest='plot', default=False, action='store_true',
                        help="Plot the ra/dec offset models")
    group3.add_argument('--test', dest='test', default=False, action='store_true',
                        help="run test")

    results = parser.parse_args()

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    if results.test:
        dra, ddec = make_models('xmatched.fits', dec2='DECJ2000', plots=True)
        correct_image('small.fits', dra, ddec, 'small_corr.fits')
        sys.exit()

    if results.infits is not None or results.plot:
        dx, dy = make_pix_models(results.xm, results.ra1, results.dec1, results.ra2, results.dec2,
                                 results.infits, results.plot)
    if results.infits is not None:
        if os.path.exists(results.infits):
            correct_image(results.infits, dx, dy, results.outfits)
        else:
            print "File:{0} not found".format(results.infits)
    else:
        print "No fits file supplied not doing warping"