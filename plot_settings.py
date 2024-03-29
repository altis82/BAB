

import pylab, math

# Symbols
symbols = ['-','--','-.',':','.',',','o','^','v','<','>','s','+','x','D','d','1','2','3','4','h','H','p']
# Symbols + line
lps = [k+'-' for k in ['o','^','v','<','>','s','+','x','D','d','1','2','3','4','h','H','p']]
# Markers
markers = ['o','x', 's','v','d','^','<','>','+','*','h', 'p']
# Colors
colors= ['darkorange', 'r','purple', 'g', 'c', 'dodgerblue', 'm','darksalmon','b','y','darkgrey','c']
#colors= []
#for name, hex in matplotlib.colors.cnames.iteritems():
#    colors.append(name)


def get_figsize(fig_width_pt):
    inches_per_pt = 1.0/72.0                # Convert pt to inch
    golden_mean = (math.sqrt(5)-1.0)/2.0    # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]      # exact figsize
    return fig_size

# small sized image
ptiny = {'backend': 'ps',
          'axes.labelsize': 11,  
         
          'xtick.labelsize': 9,
          'ytick.labelsize': 9,
          'legend.borderpad': 0.4,    # empty space around the legend box
          'legend.fontsize': 7,
          'lines.markersize': 4.5,
          'lines.linewidth': 1.3,
          'font.size': 10,
          'text.usetex': False,
          'text.latex.unicode'  : True ,
          'figure.figsize': get_figsize(240)}


# small sized image
psmall = {'backend': 'ps',
          'axes.labelsize': 11,
          'font.size': 9,
          'xtick.labelsize': 9,
          'ytick.labelsize': 9,
          'legend.borderpad': 0.15,    # empty space around the legend box
          'legend.fontsize': 5,
          'lines.markersize': 5,
          'lines.linewidth': 1.3,
          'font.size': 8,
          'text.usetex': True,
          'figure.figsize': get_figsize(320)}

# medium sized images
pmedium = {'backend': 'ps',
          'axes.labelsize': 13,
          'text.fontsize': 13,
          'xtick.labelsize': 13,
          'ytick.labelsize': 13,
          'legend.borderpad': 0.20,     # empty space around the legend box
          'legend.fontsize': 10.5,
          'lines.markersize': 5,
          'font.size': 11,
          'text.usetex': True,
          'figure.figsize': get_figsize(500)}

# large sized images (default)
plarge = {'backend': 'ps',
          'axes.labelsize': 15,
          'text.fontsize': 15,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'legend.borderpad': 0.2,     # empty space around the legend box
          'legend.fontsize': 12,
          'lines.markersize': 5,
          'font.size': 12,
          'text.usetex': True,
          'figure.figsize': get_figsize(900)}

def set_mode(mode):
    if mode == "small":
        pylab.rcParams.update(psmall)
    elif mode == "tiny":
        pylab.rcParams.update(ptiny)
    elif mode == "medium":
        pylab.rcParams.update(pmedium)
    else:
        pylab.rcParams.update(plarge)

def set_figsize(fig_width_pt):
    pylab.rcParams['figure.figsize'] = get_figsize(fig_width_pt)

pylab.rcParams['text.usetex'] = True
