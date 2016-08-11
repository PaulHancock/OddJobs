#! python

__author__ = "PaulHancock"
__date__ = "11/08/2016"

import numpy as np
from scipy import interpolate
from scipy.interpolate import LinearNDInterpolator
from astropy import wcs
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle, Latitude, Longitude
import astropy.units as u
import sys
import argparse


def make_models(fname, ra1='ra', dec1='dec', ra2='RAJ2000', dec2='DEJ2000'):
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
    :return: (dramodel, ddecmodel)
    """
    raw_data = fits.open(fname)[1].data

    # filter the data to only include SNR>10 sources
    flux_mask = np.where(raw_data['peak_flux']/raw_data['local_rms']>10)
    data = raw_data[flux_mask]

    #calculate the offsets in the ra/dec directions
    catalog = Longitude(data[ra1], unit=u.degree), Latitude(data[dec1], unit=u.degree)
    reference = Longitude(data[ra2], unit=u.degree), Latitude(data[dec2], unit=u.degree)

    dra = (reference[0]-catalog[0]).degree
    ddec = (reference[1]-catalog[1]).degree

    dramodel = interpolate.Rbf(catalog[0].degree, catalog[1].degree, dra, function='linear', smooth=3)
    ddecmodel = interpolate.Rbf(catalog[0].degree, catalog[1].degree, ddec, function='linear', smooth=3)

    return dramodel, ddecmodel


def correct_image(fname, dramodel, ddecmodel, fout):
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
            ra[s1] += dramodel(ra[s1], dec[s1])
            dec[s1] += ddecmodel(ra[s1], dec[s1])
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
    group3 = parser.add_argument_group("testing")
    group3.add_argument('--test', dest='test', default=False, action='store_true',
                        help="run test")

    results = parser.parse_args()

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    if results.test:
        dra, ddec = make_models('xmatched.fits', dec2='DECJ2000')
        correct_image('small.fits', dra, ddec, 'small_corr.fits')
        sys.exit()

    dra, ddec = make_models(results.xm, results.ra1, results.dec1, results.ra2, results.dec2)
    correct_image(results.infits, dra, ddec, results.outfits)