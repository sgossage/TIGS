#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import random
import sys

def pltdeB(show_tigs = False, absmag = False, hifihids=False, rawhids=False):

    # This is just for troubleshooting the de Bruijne+ 2001 catalogue.
    # Trying to ensure that I have just the high fidelity members.

    # First, HIFI data...
    hifidata = np.genfromtxt('data_tables/debruHIFI_allcols.tsv', delimiter='\t', 
                             skip_header = 333, names=True)

    # hifi stars' v mags:
    v = hifidata['Vmag']
    # the b-v color...
    bv = hifidata['BV']
 
    # debruRaw_morecols, 158
    #rawdata = np.genfromtxt('data_tables/debru01_p98_hipmain_with20894.tsv', delimiter='\t', 
    #                         skip_header = 164, names=True)
    rawdata = np.genfromtxt('data_tables/debruRaw_morecols.tsv', delimiter='\t', 
                             skip_header = 158, names=True)
    # raw v and bv (not cutting out any stars):
    raw_v = rawdata['Vmag']
    raw_bv = rawdata['BV']

    # if told to, plot the TIGS photometry:
    if show_tigs:
        tigs_b = np.genfromtxt('phot_files/hyades_debruTIGS_TychoBV.phot', usecols=(0,))
        tigs_v = np.genfromtxt('phot_files/hyades_debruTIGS_TychoBV.phot', usecols=(1,))
        tigs_bv = tigs_b - tigs_v

    # if told to, make it an absolute mag:
    if absmag:
        # convert from mas to as:
        hifi_theta = (hifidata['PlxsH']) / 1000.0
        raw_theta = (rawdata['PlxsH']) / 1000.0

        hifi_d = 1. / hifi_theta
        raw_d = 1. / raw_theta

        # abs mags
        v = v - 5*np.log10(hifi_d) + 5
        raw_v = raw_v - 5*np.log10(raw_d) + 5

    fig = plt.figure(figsize = (16,9))

    for plti in range(2):

        # always plot HIFI, and for other only plot on first subplot.
        hifion = True
        if plti == 0:
            rawon = True
        else:
            rawon = False
            show_tigs = False

        ax = fig.add_subplot(int('12{:d}'.format(plti+1)))

        # plot the CMD(s)...
        if rawon:
            # raw points 1st.
            ax.scatter(raw_bv, raw_v, s=8, label='Raw')

        if hifion:
            # cut "high fidelity" stars:
            ax.scatter(bv, v, marker='+', c = 'r', 
                       label='High Fidelity',zorder=-1)

        if show_tigs:
            # photometry from cross matching with TIGS catalogue:
            ax.scatter(tigs_bv, tigs_v, s=8, label='TIGS', c = 'c')    

        # labeling
        ax.set_xlabel(r'$B - V$')

        if absmag:
            ax.set_ylabel(r'$M_{V,\pi_{Hip}}$')
        else:
            ax.set_ylabel(r'$V$')

        ax.invert_yaxis()
        ax.legend()

        # display hipparcos ID nums for raw data points:
        if raw_hidson:
            hids = rawdata['HIP']

            dispHID(fig, ax, [raw_bv, raw_v], hids=hids,
                    regions=[[(0.1,0.5), (5.7,7)]])

        # display IDs of the HIFI data points:
        elif hifi_hidson:
            hids = hifidata['HIP']
 
            dispHID(fig, ax, [bv, v], hids = hids,
                    regions=[[(0.1,0.5), (5.7,7)]])

    plt.show()

    return

def dispHID(fig, ax, colors_mags, regions=[[]], 
            hids = np.genfromtxt('data_tables/debruRaw.tsv', 
                                 usecols=(0,), delimiter='\t', 
                                 skip_header=141)
           ):

    # colors_mags is an list of arrays [colors, mags].
    # data_table should match the table where the colors and magnitudes
    # were drawn from.

    colors = colors_mags[0]
    mags = colors_mags[1]

    for region in regions:
        # displays Hipparos IDs in a region of the CMD.
        # specify as a list of tuples or lists [xlims, ylims] w/
        # e.g. xlims = (xmin, xmax), etc., but order doesn't really
        # matter, could be xlims = (xmax, xmin).
        if region == []:
            ylims = ax.get_ylim()
            xlims = ax.get_xlim()
            region = [xlims, ylims]
        

        regxlims = region[0]
        regylims = region[1]
        # check for the magnitudes/colors that are within 
        # the specified region:
        indices = np.where((min(regxlims) < colors) & \
                           (colors < max(regxlims))  & \
                           (min(regylims) < mags)   & \
                           (mags < max(regylims)) 
                           )[0]

        # Go through the indices and grab the HIDs for valid stars:
        #hids = np.array(map(int, hids[indices]))
        texts = []
        for i in indices:
            txtx = colors[i]
            txty = mags[i]
            txtx, txty = scattertext(fig, txtx, txty,
                                     "{:d}".format(int(hids[i])),
                                     texts=texts, ax=ax)
            # write the HIDs stacked somewhere to the side?
            texts.append(ax.text(txtx, txty, "{:d}".format(int(hids[i]))))

    return

# look through the texts and check for overlapping positions
# upon encountering an overlap, move to offending text.
def scattertext(fig, txtx, txty, string, texts = [], ax=None, 
                xpad=0.08, ypad=0.25):
    
    if ax == None:
        ax = plt.gca()

    if texts == []:
        return (txtx, txty)

    initx, inity = (txtx, txty)

    i = 0
    ctr = 0
    bounces = 0
    while i < len(texts):

        # iteration breaker, shouldn't be reached or else
        # indicates that text has not been checked against
        # all others.
        if ctr >= 10000:
            break

        # clear previous text object.
        try:
            curr_text.remove()
        except NameError:
            pass

        # create new text object at proposed position
        curr_text = ax.text(txtx, txty, string)
        
        # check against i'th text object already on axis
        text = texts[i]

        # get bounding box limits.
        bbxlims, bbylims, vs = get_bblims(fig, ax, text)
        curr_bbxlims, curr_bbylims, curr_vs = get_bblims(fig, ax, curr_text)
        # for reference: vertices go ur, ul, ll, lr

        # if the new text position is within the bounds of a prev. text's
        # bounding box. Check if any vertex is contained within the other
        # BBox. Below checks each vertex of BBox. i= 0 means restart loop
        # and check against all texts again.
        if inbox(curr_vs[0], bbxlims, bbylims):
            # move down left:
            txtx = vs[2][0] - xpad
            txty = vs[2][1] + ypad
            i = 0

        elif inbox(curr_vs[1], bbxlims, bbylims):
            # move down right:
            txtx = vs[3][0] + xpad
            txty = vs[3][1] + ypad
            i = 0

        elif inbox(curr_vs[2], bbxlims, bbylims):
            # move up right:
            txtx = vs[0][0] + xpad
            txty = vs[0][1] - ypad
            i = 0

        elif inbox(curr_vs[3], bbxlims, bbylims):
            # move up left:
            txtx = vs[1][0] - xpad
            txty = vs[1][1] - ypad
            i = 0
     
        # check midpoint (fires pretty foten):
        elif inbox((curr_bbxlims.mean(), curr_bbylims.mean()), 
                    bbxlims, bbylims):
            # scatter around circle inscribing intersected BBox
            # i.e. circle w/ radius * 2 of what perfectly inscrinbing.
            bbxdist = bbxlims.max() - bbxlims.min()
            bbydist = bbylims.max() - bbylims.min()
            bbrad = np.sqrt((0.5*bbxdist)**2 + (0.5*bbydist)**2)
            theta = 2*np.pi*random.random()
            # instead, scatter the point randomly in a circle:
            txtx, txty = (2*bbrad*np.cos(theta) + bbxlims.mean(), 
                          2*bbrad*np.sin(theta) + bbylims.mean())
            i = 0 
            

        # if not within bounding box, move along to other texts.
        else:
            i += 1

        # iteration counter for safety purposes.
        ctr += 1

    # remove dummy text:
    curr_text.remove()

    # if the point was scattered away, draw a line from it to its data point.
    if initx != txtx or inity != txty and i != 0:
        ax.plot([initx, txtx], [inity, txty], c='k', zorder=-99, alpha = 0.2)

    # return new (x, y) position of scattered text:
    return (txtx, txty)

# checks if an (x, y) point lies within a box.
def inbox(point, boxx, boxy):#, xpad = 0.08, ypad = 0.2):
 
    if ((boxx.min()) < point[0] < (boxx.max())) & \
       ((boxy.min()) < point[1] < (boxy.max())):

        return True

    else:
        return False

# get limits of a text's bounding box:
def get_bblims(fig, ax, text):

    bbox = text.get_window_extent(renderer = find_renderer(fig))

    # here are the limits of a text's bounding box
    bblims = ax.transData.inverted().transform(bbox)
    bbxmax = bblims[1][0]
    bbxmin = bblims[0][0]
    bbymax = bblims[1][1]
    bbymin = bblims[0][1]

    vertices = [(bbxmax, bbymax),
                (bbxmin, bbymax),
                (bbxmin, bbymin),
                (bbxmax, bbymin)]

    return np.array([bbxmin, bbxmax]), np.array([bbymin, bbymax]), vertices

# finds the figure renderer:
def find_renderer(fig):
     if hasattr(fig.canvas, "get_renderer"):
         renderer = fig.canvas.get_renderer()
     else:
         import io
         fig.canvas.print_pdf(io.BytesIO())
         renderer = fig._cachedRenderer
     return renderer

if __name__ == '__main__':

    show_tigs = False
    hifi_hidson = False
    raw_hidson = False
    absmag = False

    for arg in sys.argv[1:]:
        if "-hifihids" == arg:
            hifi_hidson = True
        
        if "-rawhids" == arg:
            raw_hidson = True
        if "-abs" == arg:
            absmag = True
        if "-tigs" == arg:
            show_tigs = True

    pltdeB(show_tigs=show_tigs, absmag=absmag, hifihids = hifi_hidson, rawhids = raw_hidson)
