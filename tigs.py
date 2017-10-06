#!/usr/bin/env python

from astropy.io import fits
import numpy as np
import sys
from functools import partial
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from MIST_codes.scripts.read_mist_models import convert_name

def get_val(hipid, data_table, dtab_hipids, datname='Tycho_V_Mag'):

    try:
        if str(hipid) in dtab_hipids:
            return data_table[datname][np.where(dtab_hipids == str(hipid))[0][0]]

        else:
            print("Hipparcos ID {:d} not found.".format(hipid))
            return -999

    except KeyError:
        # display list of possible magnitude names.
        print('Key \"{:s}\" does not exist. Here\'s a list of valid names:'.format(datname))
        for key in data_table.names:
            print(key)

        # stop execution
        sys.exit(0)

def get_dtab():

    # read TIGS fits file:
    hdulist = fits.open('TIGS_v10.2.fits.gz', memmap=True)
    dtab = hdulist[1].data

    return dtab

def get_mags(blue_filter, red_filter, ids, dtab):

    # magnitude arrays recovered will be ordered according to the order of the Hip ID array provided.

    print('Retrieving {:s} and {:s} magnitudes from TIGS...'.format(blue_filter, red_filter))

    hipids = dtab['HIP_ID']

    blue_mag = np.array([get_val(idnum, dtab, hipids, '{:s}_Mag'.format(blue_filter)) for idnum in ids])
    red_mag = np.array([get_val(idnum, dtab, hipids, '{:s}_Mag'.format(red_filter)) for idnum in ids])
    blue_err = np.array([get_val(idnum, dtab, hipids, '{:s}_Mag_Err'.format(blue_filter)) for idnum in ids])
    red_err = np.array([get_val(idnum, dtab, hipids, '{:s}_Mag_Err'.format(red_filter)) for idnum in ids])

    # eliminate "bad" indices suiting conditions below:
    bad_indices = np.where((abs(blue_mag) == 999) | (abs(red_mag) == 999) | (blue_err >= 0.1) | (red_err >= 0.1))
    blue_mag = np.delete(blue_mag, bad_indices)
    red_mag = np.delete(red_mag, bad_indices)
    blue_err = np.delete(blue_err, bad_indices)
    red_err = np.delete(red_err, bad_indices)
    br_err = np.sqrt(blue_err**2 + red_err**2)
    recovered_ids = np.delete(ids, bad_indices)

    # return blue, red; blue, red, blue-red errors; recovered ids
    return np.array([blue_mag, red_mag]), np.array([blue_err, red_err, br_err]), recovered_ids

def praesepe_FvL2009XTIGS(dtab, blue_filter, red_filter):

    praesepeids = np.genfromtxt('/n/home12/sgossage/Downloads/PraesepeHip_vanLeeuwen2.txt', usecols=(6,), skip_header=19)
    praesepeids = [int(idnum) for idnum in praesepeids if idnum != 42523] # 42523 is 40 Cnc

    mags, errs, ids = get_mags(blue_filter, red_filter, praesepeids, dtab)
    Bmag, Rmag = mags
    Berr, Rerr, BRerr = errs

    # strings describing filters used, could be more sophisticated:
    if 'Tycho' in blue_filter:
        filterstr = 'TychoBV'
    elif 'TMASS' in blue_filter:
        filterstr = '2MASSJK'

    savename_base = 'praesepe_vanlTIGS_{:s}'.format(filterstr)

    xlabel = r"${:s} - {:s}$".format(convert_name(blue_filter), convert_name(red_filter))
    ylabel = r"${:s}$".format(convert_name(red_filter))
    ax = plotphot(Bmag-Rmag, Rmag, xlabel, ylabel, savename_base)

    np.savetxt('phot_files/{:s}.phot'.format(savename_base), 
               X=np.c_[Bmag, Rmag], delimiter='\t',fmt="%f")

    np.savetxt('phot_files/{:s}.err'.format(savename_base),
                X=np.c_[Berr, Rerr, BRerr], delimiter='\t',fmt="%f")

    return np.array([Bmag, Rmag]), np.array([Berr, Rerr, BRerr])

def pleiades_FvL2009XTIGS(dtab, blue_filter, red_filter):
    pleiadesids = np.genfromtxt('/n/home12/sgossage/Downloads/PleiadesHip_vanLeeuwen2.txt', usecols=(6,), skip_header=21)
    pleiadesids = [int(idnum) for idnum in pleiadesids]

    mags, errs, ids = get_mags(blue_filter, red_filter, pleiadesids, dtab)
    Bmag, Rmag = mags
    Berr, Rerr, BRerr = errs

    # strings describing filters used, could be more sophisticated:
    if 'Tycho' in blue_filter:
        filterstr = 'TychoBV'
    elif 'TMASS' in blue_filter:
        filterstr = '2MASSJK'

    savename_base = 'pleiades_vanlTIGS_{:s}'.format(filterstr)

    xlabel = r"${:s} - {:s}$".format(convert_name(blue_filter), convert_name(red_filter))
    ylabel = r"${:s}$".format(convert_name(red_filter))
    ax = plotphot(Bmag-Rmag, Rmag, xlabel, ylabel, savename_base)

    np.savetxt('phot_files/{:s}.phot'.format(savename_base), 
               X=np.c_[Bmag, Rmag], delimiter='\t',fmt="%f")

    np.savetxt('phot_files/{:s}.err'.format(savename_base),
                X=np.c_[Berr, Rerr, BRerr], delimiter='\t',fmt="%f")

    return np.array([Bmag, Rmag]), np.array([Berr, Rerr, BRerr])

def hyades_deB2001XTIGS(dtab, blue_filter, red_filter):
    #hyadesids = np.genfromtxt('data_tables/debru01_p98_hipmain_with20894.tsv', usecols=(2,), skip_header=164, delimiter='\t')
    hyades_data = np.genfromtxt('data_tables/debruHIFI_allcols.tsv', delimiter = '\t', skip_header = 333, names=True)
    hyadesids = map(int, hyades_data['HIP'])

    # So, there's an issue where the Hyades is pretty close to us & its stars can have an appreciable range in flux just from
    # where they are in the cluster.

    # Add in each star's hipparcos secular parallax based distance modulus.
    # Then subtract out the average Hyades distance modulus -- 3.36
    # Will effectively remove the scatter and fit ~ with Mv, although not
    # entirely accurate. Treating each star as if it were at same distance from us.

    if blue_filter != 'B':
        mags, errs, ids = get_mags(blue_filter, red_filter, hyadesids, dtab)
        Bmag, Rmag = mags
        Berr, Rerr, BRerr = errs
    else:
        # this seperately fishes out the Hipparcos color and magnitudes since theyre stored different from everthing else.
        hipids = dtab['HIP_ID']
        HipBV = np.array([get_val(idnum, dtab, hipids, 'HIP_B_V') for idnum in hyadesids])
        HipHp = np.array([get_val(idnum, dtab, hipids, 'Hip_Hp_Mag') for idnum in hyadesids]) 
        
    # The magnitudes retrieved above are ordered according to the Hip ID order from the original data file.
    # However, maybe not all IDs were recovered...
    data_plxsh = {key:item for key, item in zip(map(int, hyades_data['HIP']), hyades_data['PlxsH'])}
    # Get the secular parallaxes:
    # convert from mas to as:
    #print(Bmag)
    for i, anid in enumerate(ids):
        theta = (data_plxsh[anid]) / 1000.0
        # distance based on parallax ~ 1/angle
        d = 1. / theta
        print(Bmag[i])
        print(-5*np.log10(d) + 5)
        # convert to abs mag
        Bmag[i] = Bmag[i] - 5*np.log10(d) + 5
        Rmag[i] = Rmag[i] - 5*np.log10(d) + 5

    # now bring it back to an ~apparent mag using dmod = 3.36 (average for Hyades).
    Bmag = Bmag + 3.36
    Rmag = Rmag + 3.36

    # strings describing filters used, could be more sophisticated:
    if 'Tycho' in blue_filter:
        filterstr = 'TychoBV'
    elif 'TMASS' in blue_filter:
        filterstr = '2MASSJK'
    elif blue_filter == 'B':
        filterstr = 'HpBV'

    savename_base = "hyades_debruTIGS_{:s}".format(filterstr)

    # plot CMD of the data:

    if blue_filter != 'B':
        xlabel = r"${:s} - {:s}$".format(convert_name(blue_filter), convert_name(red_filter))
        ylabel = r"${:s}$".format(convert_name(red_filter))
        ax = plotphot(Bmag-Rmag, Rmag, xlabel, ylabel, savename_base)

        np.savetxt('phot_files/{:s}.phot'.format(savename_base),
                   X=np.c_[Bmag, Rmag], delimiter='\t',fmt="%f")

        np.savetxt('phot_files/{:s}.err'.format(savename_base),
                    X=np.c_[Berr, Rerr, BRerr], delimiter='\t',fmt="%f")

        return np.array([Bmag, Rmag]), np.array([Berr, Rerr, BRerr])

    else:
        xlabel = r"${:s} - {:s}$".format('B', 'V')
        ylabel = r"${:s}$".format('H_p')
        ax = plotphot(HipBV, HipHp, xlabel, ylabel, savename_base)

        directv = np.genfromtxt('data_tables/debru01_p98_hipmain_with20894.tsv', usecols=(5,), skip_header=164, delimiter='\t')
        directbv = np.genfromtxt('data_tables/debru01_p98_hipmain_with20894.tsv', usecols=(12,), skip_header=164, delimiter='\t')
        ax = plotphot(directbv, directv, xlabel, r'$V$', savename_base = 'hyades_debruTIGS_BV')

        np.savetxt('phot_files/{:s}.phot'.format(savename_base),
                   X=np.c_[HipBV, HipHp], delimiter='\t',fmt="%f")

        return

def hyades_G13():

    # just taking mags and errs directly from data file, not cross matching with TIGS.

    data_filename = "data_tables/goldman2013_highfidelity.tsv"

    #G13J = np.genfromtxt('/n/home12/sgossage/Downloads/goldman_Hyades2MASS.txt', usecols=(5,), skip_header=59)
    #G13K = np.genfromtxt('/n/home12/sgossage/Downloads/goldman_Hyades2MASS.txt', usecols=(9,), skip_header=59)
    #G13Jerr = np.genfromtxt('/n/home12/sgossage/Downloads/goldman_Hyades2MASS.txt', usecols=(6,), skip_header=59)
    #G13Kerr = np.genfromtxt('/n/home12/sgossage/Downloads/goldman_Hyades2MASS.txt', usecols=(10,), skip_header=59)

    G13J = np.genfromtxt(data_filename, usecols=(13,), skip_header=76, delimiter='\t')
    G13K = np.genfromtxt(data_filename, usecols=(17,), skip_header=76, delimiter='\t')
    G13Jerr = np.genfromtxt(data_filename, usecols=(14,), skip_header=76, delimiter='\t')
    G13Kerr = np.genfromtxt(data_filename, usecols=(18,), skip_header=76, delimiter='\t')

    #plt.scatter(G13J - G13K, G13K)
    G13J = G13J[np.where((G13Jerr < 0.99) & (G13Kerr < 0.99))[0]]
    G13K = G13K[np.where((G13Jerr < 0.99) & (G13Kerr < 0.99))[0]]
    G13Jerrcut = G13Jerr[np.where((G13Jerr < 0.99) & (G13Kerr < 0.99))[0]]
    G13Kerrcut = G13Kerr[np.where((G13Jerr < 0.99) & (G13Kerr < 0.99))[0]]
    G13Jerr = G13Jerrcut
    G13Kerr = G13Kerrcut
    G13JKerr = np.sqrt(G13Jerr**2 + G13Kerr**2)
    #plt.scatter(G13J - G13K, G13K, marker='*')
    #plt.title('Hyades Goldman+ 13')
    #plt.xlabel('2MASS J - K')
    #plt.ylabel('2MASS K')
    #plt.gca().invert_yaxis()
    #plt.show()

    savename_base = 'hyades_goldman13HIFI_2MASSJK'

    xlabel = r"${:s} - {:s}$".format(convert_name(blue_filter), convert_name(red_filter))
    ylabel = r"${:s}$".format(convert_name(red_filter))
    ax = plotphot(G13J-G13K, G13K, xlabel, ylabel, savename_base = savename_base)

    np.savetxt('phot_files/{:s}.phot'.format(savename_base), X=np.c_[G13J, G13K], delimiter='\t',fmt="%f")
    np.savetxt('phot_files/{:s}.err'.format(savename_base), X=np.c_[G13Jerr, G13Kerr, G13JKerr], delimiter='\t',fmt="%f")

    return np.array([G13J, G13K]), np.array([G13Jerr, G13Kerr, G13JKerr])

def pleiades_S07():

    S07J = np.genfromtxt('/n/home12/sgossage/Downloads/Pleiades_Stauffer2007_JHKBVI.txt', usecols=(2,), skip_header=78)
    S07K = np.genfromtxt('/n/home12/sgossage/Downloads/Pleiades_Stauffer2007_JHKBVI.txt', usecols=(8,), skip_header=78)
    S07Jerr = np.genfromtxt('/n/home12/sgossage/Downloads/Pleiades_Stauffer2007_JHKBVI.txt', usecols=(3,), skip_header=78)
    S07Kerr = np.genfromtxt('/n/home12/sgossage/Downloads/Pleiades_Stauffer2007_JHKBVI.txt', usecols=(9,), skip_header=78)
    #colnum=18
    #names = []
    #with open('/n/home12/sgossage/Downloads/Pleiades_Stauffer2007_JHKBVI.txt', 'r') as s07f: 
    #    lines = s07f.readlines()
    #    lines = lines[78:-2]
    #    for line in lines:
    #        names.append(' '.join(line.split()[-5:]))
    #print(names)
    #names = np.array(names)
    #plt.scatter(S07J - S07K, S07K, label='Raw Stauffer+ \'07', zorder=0)
    S07J = S07J[np.where((S07Jerr < 0.99) & (S07Kerr < 0.99))[0]]
    S07K = S07K[np.where((S07Jerr < 0.99) & (S07Kerr < 0.99))[0]]
    S07Jerrcut = S07Jerr[np.where((S07Jerr < 0.99) & (S07Kerr < 0.99))[0]]
    S07Kerrcut = S07Kerr[np.where((S07Jerr < 0.99) & (S07Kerr < 0.99))[0]]
    #names = names[np.where((S07Jerr < 0.99) & (S07Kerr < 0.99))[0]]
    S07Jerr = S07Jerrcut
    S07Kerr = S07Kerrcut
    S07JKerr = np.sqrt(S07Jerr**2 + S07Kerr**2)
    #fig = plt.figure(figsize=(16,9))
    #plt.scatter(S07J - S07K, S07K, marker='*', label='S07, No J,K err >=0.99', zorder=1)
    #plt.scatter(pleiadesJ-pleiadesK, pleiadesK,marker='+',label='van L.+ 2009 X TIGS', c='r',s=120, zorder=2)
    #plt.title('Pleiades Stauffer+ 07')
    #plt.xlabel('2MASS J - K')
    #plt.ylabel('2MASS K')
    #idx=0
    #for i, mag in enumerate(S07K):
    #    if mag < 6.:
    #        idx+=1
    #        x = S07J[i] - mag
    #        y = mag
    #        plt.text(1.5, 4+0.5*idx, "{:s}".format(names[i]), fontsize=8)
    #        plt.plot([x, 1.45], [y, 4+0.5*idx],alpha=0.8)

    #plt.gca().invert_yaxis()
    #plt.legend(loc='lower left', prop={'size':6})
    #plt.show()

    savename_base = 'pleiades_stauffer07_2MASSJK'

    xlabel = r"${:s} - {:s}$".format(convert_name(blue_filter), convert_name(red_filter))
    ylabel = r"${:s}$".format(convert_name(red_filter))
    ax = plotphot(S07J-S07K, S07K, xlabel, ylabel, savename_base = savename_base)

    np.savetxt('phot_files/{:s}.phot'.format(savename_base), X=np.c_[S07J, S07K], delimiter='\t',fmt="%f")
    np.savetxt('phot_files/{:s}.err'.format(savename_base), X=np.c_[S07Jerr, S07Kerr, S07JKerr], delimiter='\t',fmt="%f")

    return np.array([S07J, S07K]), np.array([S07Jerr, S07Kerr, S07JKerr])

def pleiades_S07BV():

    S07B = np.genfromtxt('/n/home12/sgossage/Downloads/Pleiades_Stauffer2007_JHKBVI.tsv', 
                         usecols=(13,), skip_header=72, delimiter='\t', dtype=None)
    print(S07B)
    S07V = np.genfromtxt('/n/home12/sgossage/Downloads/Pleiades_Stauffer2007_JHKBVI.tsv', 
                         usecols=(14,), skip_header=72, delimiter='\t', dtype=None)
    #S07Jerr = np.genfromtxt('/n/home12/sgossage/Downloads/Pleiades_Stauffer2007_JHKBVI.txt', usecols=(3,), skip_header=78)
    #S07Kerr = np.genfromtxt('/n/home12/sgossage/Downloads/Pleiades_Stauffer2007_JHKBVI.txt', usecols=(9,), skip_header=78)


    S07Btemp = S07B[np.where((S07B < 99) & (S07V < 99))[0]]
    print(S07B)
    S07Vtemp = S07V[np.where((S07B < 99) & (S07V < 99))[0]]
    print(S07V)
    S07B = S07Btemp
    S07V = S07Vtemp

    #S07Vcut = S07B[np.where((S07B < 99) & (S07V < 99))[0]]
    #S07Kerrcut = S07Kerr[np.where((S07Jerr < 0.99) & (S07Kerr < 0.99))[0]]
    #names = names[np.where((S07Jerr < 0.99) & (S07Kerr < 0.99))[0]]
    #S07Jerr = S07Jerrcut
    #S07Kerr = S07Kerrcut
    #S07JKerr = np.sqrt(S07Jerr**2 + S07Kerr**2)

    savename_base = 'pleiades_stauffer07_BV'

    xlabel = r"${:s} - {:s}$".format(convert_name(blue_filter), convert_name(red_filter))
    ylabel = r"${:s}$".format(convert_name(red_filter))
    ax = plotphot(S07B-S07V, S07V, xlabel, ylabel, savename_base = savename_base)

    np.savetxt('phot_files/{:s}.phot'.format(savename_base), X=np.c_[S07B, S07V], delimiter='\t',fmt="%f")
    #np.savetxt('phot_files/{:s}.err'.format(savename_base), X=np.c_[S07Jerr, S07Kerr, S07JKerr], delimiter='\t',fmt="%f")

    return np.array([S07B, S07V])#, np.array([S07Jerr, S07Kerr, S07JKerr])

def praesepe_W14():

    W14J = np.genfromtxt('/n/home12/sgossage/Downloads/wang2014praesepe.tsv', usecols=(18,), skip_header=86, delimiter='\t')
    W14K = np.genfromtxt('/n/home12/sgossage/Downloads/wang2014praesepe.tsv', usecols=(22,), skip_header=86, delimiter='\t')
    W14Jerr = np.genfromtxt('/n/home12/sgossage/Downloads/wang2014praesepe.tsv', usecols=(19,), skip_header=86, delimiter='\t')
    W14Kerr = np.genfromtxt('/n/home12/sgossage/Downloads/wang2014praesepe.tsv', usecols=(23,), skip_header=86, delimiter='\t')
    W14J = W14J[~np.isnan(W14J)]
    W14K = W14K[~np.isnan(W14K)]
    W14Jerr = W14Jerr[~np.isnan(W14J)]
    W14Kerr = W14Kerr[~np.isnan(W14K)]
    #plt.scatter(W14J - W14K, W14K, label='Raw Wang+ 2014')
    W14J = W14J[np.where((W14Jerr < 0.99) & (W14Kerr < 0.99))[0]]
    W14K = W14K[np.where((W14Jerr < 0.99) & (W14Kerr < 0.99))[0]]
    W14Jerrcut = W14Jerr[np.where((W14Jerr < 0.99) & (W14Kerr < 0.99))[0]]
    W14Kerrcut = W14Kerr[np.where((W14Jerr < 0.99) & (W14Kerr < 0.99))[0]]
    W14Jerr = W14Jerrcut
    W14Kerr = W14Kerrcut
    W14JKerr = np.sqrt(W14Jerr**2 + W14Kerr**2)
 
   #plt.scatter(W14J - W14K, W14K, marker='*', label='W14, No J,K err >=0.99')
    #plt.scatter(praesepeJ-praesepeK, praesepeK,marker='+',label='van L.+ 2009 X TIGS', c='r',s=120, zorder=2)
    #for i, mag in enumerate(praesepeK):
    #    plt.text(praesepeJ[i] - mag + 0.01, mag, "{:d}".format(praesepeJKids[i]), fontsize=6)
    #plt.text(5.94 - 5.879 + 0.01, 5.879, "eps Cnc", fontsize=6)
    #plt.text(5.374 - 4.997 + 0.01, 4.997, "35 Cnc", fontsize=6)
    #plt.text(5.24 - 4.683 + 0.01, 4.683, "HD 73974", fontsize=6)
    #plt.text(5.18 - 4.39 + 0.01, 4.39, "HD 73598", fontsize=6)
    #plt.text(4.79 - 4.19 + 0.01, 4.19, "HD 73710", fontsize=6)
    #plt.text(4.74 - 4.23 + 0.01, 4.23, "39 Cnc", fontsize=6)
    #plt.title('Praesepe Wang+ 14')
    #plt.xlabel('2MASS J - K')
    #plt.ylabel('2MASS K')
    #plt.gca().invert_yaxis()
    #plt.legend(loc='lower left', prop={'size':6})
    #plt.show()

    savename_base = 'praesepe_wang14_2MASSJK'

    xlabel = r"${:s} - {:s}$".format(convert_name(blue_filter), convert_name(red_filter))
    ylabel = r"${:s}$".format(convert_name(red_filter))
    ax = plotphot(W14J-W14K, W14K, xlabel, ylabel, savename_base = savename_base)    

    np.savetxt('phot_files/{:s}.phot'.format(savename_base),
                X=np.c_[W14J, W14K], delimiter='\t',fmt="%f")

    np.savetxt('phot_files/{:s}.err'.format(savename_base), 
               X=np.c_[W14Jerr, W14Kerr, W14JKerr], delimiter='\t',fmt="%f")

    return np.array([W14J, W14K]), np.array([W14Jerr, W14Kerr, W14JKerr])

#def generate_phot(phot_tag, blue_filter, red_filter):

#    mags, errs = phot_funcs[phot_tag]()

#    return mags, errs

def plotphot(x, y, xlabel = None, ylabel = None, savename_base = None):

    # make a CMD of the data for inspection.
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    # red vs. blue - red, label w/ 3 no. of stars:
    ax.scatter(x, y, label='N = {:d}'.format(len(y)))
    # put code here to label axes
    #ax.set_ylabel(r"${:s}$".format(convert_name(red_filter)))
    #ax.set_xlabel(r"${:s} - {:s}$".format(convert_name(blue_filter), convert_name(red_filter)))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    ax.legend()

    if savename_base is not None:
        plt.savefig('phot_files/{:s}.png'.format(savename_base))

    return ax

if __name__ == '__main__':

    # bands in TIGS: e.g. Tycho_B_Mag, TMASS_K_Mag (not 'Ks'). HIP_Hp_Mag...see print(dtab.names)
    # Takes 3 input arguments: sys.argv[x] where x = 1 is function caller (below), x = 2, 3 is blue, red filter names.

    hdulist = fits.open('TIGS_v10.2.fits.gz', memmap=True)
    dtab = hdulist[1].data

    func_callers = ['PraesepeFvL09',
                    'PleiadesFvL09',
                    'HyadesdeB01',
                    'HyadesG13',
                    'PleiadesS07',
                    'PleiadesS07BV',
                    'PraesepeW14']

    phot_tag = sys.argv[1]
    if phot_tag not in func_callers:
        print("The supplied tag \"{:s}\" is invalid. Valid photometry tags are:\n".format(sys.argv[1]))
        for caller in func_callers:
            print(caller)
        print('\nPlease enter one of the above as the 1st argument.')
        sys.exit()

    # checks if filters have been specified as input arguments.
    try:
        for i, band in enumerate(sys.argv[2:4]):
            if '2MASS' in band:
                # TIGS designates '2MASS' via 'TMASS', so convert if nec.:
                sys.argv[2+i] = band.replace('2MASS', 'TMASS')

        blue_filter = sys.argv[2]
        red_filter = sys.argv[3]

    except IndexError:
        # photometry tags referencing Goldman+ 2013, Stauffer+ 2007, Wang+ 2014 do not require filter tags.
        # Using 2MASS tags just as dummy values (b/c at least they are the default values for non FvL sets).
        if any(np.array(func_callers)[0:4] == phot_tag):
            print('Please provide a blue and red filter (e.g. Tycho_B Tycho_V).')
            sys.exit()
        
        blue_filter = 'TMASS_J'
        red_filter = 'TMASS_K'

    args = (dtab, blue_filter, red_filter)
    

    funcs = [partial(praesepe_FvL2009XTIGS, *args), 
             partial(pleiades_FvL2009XTIGS, *args), 
             partial(hyades_deB2001XTIGS, *args), 
             hyades_G13, 
             pleiades_S07,
             pleiades_S07BV, 
             praesepe_W14]

    global phot_funcs

    phot_funcs = {key: item for key, item in zip(func_callers, funcs)}

    phot_funcs[phot_tag]() #generate_phot(phot_tag, blue_filter, red_filter)
