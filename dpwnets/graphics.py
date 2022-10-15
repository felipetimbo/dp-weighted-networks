import numpy as np

import os
import matplotlib.pyplot as plt 
import matplotlib.tri as mtri

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def line_plot(x, y, xlog=False, ylog=False, xlabel=None, ylabel=None, figsize=(12, 6), path=None):
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    ax.grid(True)
    gridlines = ax.get_ygridlines()
    # gridlines[5].set_linewidth(2.5)
    ax.plot(x, y, color="blue")
    if(xlog):
        plt.xscale('log')
    if(ylog):
        plt.yscale('log')
    if(xlabel):
        plt.xlabel(xlabel)
    if(ylabel):
        plt.ylabel(ylabel)
    if path:
        dir_path = os.path.dirname(os.path.realpath(path))
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(path, dpi=900)
        print('Graphic saved at: ' + path)
    else:
        plt.show()
    # plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.show()
    plt.clf()
    plt.close() 

def line_plot2(x, ys, path=None, line_legends=None, legend_path=None, 
              xlabel=None, ylabel=None, title=None, xlog=False, ylog=False,
              linestyles = [':', '--', '-.', (0, (3, 1, 1, 1, 1, 1)), ':', 'dashed', ':', '--', '-.', 'dashed'],
            #   colors = ['#000000', '#360CE8', '#4ECE00', '#FF0000', '#FF69B4', '#FFFF00', '#00009F', '#F3F0F0', '#AF10E0', '#F01F0F'],
              colors = ['#935806', '#a08214', '#bdb863', '#cce0b6', '#d89aef', '#92abf9', '#7073ac', '#442788', '#AF10E0', '#F01F0F'],
              markers = ['d','x','1','+','2','|','o','v','d','1'],
              figsize=(9, 5),
              ylim=None):

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    if markers is None:
        markers = ["None"]*len(ys)

    if linestyles is None:
        linestyles = ['-']*len(ys)
    
    if line_legends is None:
        line_legends = [None]*len(ys)
    
    lines = []
    for i,y in enumerate(ys):    
        lines.append(
            ax.plot(x, y, 
                linestyle=linestyles[i],
                linewidth=1.5, 
                color=colors[i],
                label=line_legends[i],
                marker=markers[i],
                markersize=4
                # alpha=0.5
            )
        )

    plt.legend()
    if ylog:
        plt.yscale('log')    
    # if xlog:
        # plt.xscale('log')    
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True)
    
    if path:
        dir_path = os.path.dirname(os.path.realpath(path))
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(path, dpi=300)
        print('Graphic saved at: ' + path)
    else:
        plt.show()
    plt.clf()
    plt.close()

def line_plot3(x, ys, path=None, line_legends=None, legend_path=None, 
              xlabel=None, ylabel=None, title=None, xlog=False, ylog=False,
              linestyles = [':', 'dashed', ':', '--', '-.', 'dashed'],
            #   colors = ['#000000', '#360CE8', '#4ECE00', '#FF0000', '#FF69B4', '#FFFF00', '#00009F', '#F3F0F0', '#AF10E0', '#F01F0F'],
              colors = ['#a89a1f', '#92abf9', '#7073ac', '#442788', '#AF10E0', '#F01F0F'],
              markers = ['d','|','o','v','d','1'],
              figsize=(9, 5),
              ylim=None):

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    if markers is None:
        markers = ["None"]*len(ys)

    if linestyles is None:
        linestyles = ['-']*len(ys)
    
    if line_legends is None:
        line_legends = [None]*len(ys)
    
    lines = []
    for i,y in enumerate(ys):    
        lines.append(
            ax.plot(x, y, 
                linestyle=linestyles[i],
                linewidth=1.5, 
                color=colors[i],
                label=line_legends[i],
                marker=markers[i],
                markersize=4
                # alpha=0.5
            )
        )

    plt.legend()
    if ylog:
        plt.yscale('log')    
    # if xlog:
        # plt.xscale('log')    
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True)
    
    if path:
        dir_path = os.path.dirname(os.path.realpath(path))
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(path, dpi=300)
        print('Graphic saved at: ' + path)
    else:
        plt.show()
    plt.clf()
    plt.close()


def histogram_(x, title="", xlabel="", ylabel="Frequency", path=None, color="#009edd",
                log=True, range_x=None, fig_size=(10,5), min_x_value=None, max_x_value=None, _bins=100):
    # plt.figaspect([10, 4])
    fig, ax = plt.subplots(figsize=fig_size, tight_layout=True)
    plt.hist(x, color=color, bins=_bins, range=range_x, alpha = 0.5, label="original")
    # plt.xlim(left=0.)
    if min_x_value is not None:
        plt.xlim(left=min_x_value)
    if max_x_value is not None:
        plt.xlim(right=max_x_value)
    
    # plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)    
    plt.grid(True)
    if log == True:
        plt.yscale('log')    
    if path:
        plt.savefig(path, dpi=900)
    else:
        plt.show()
    plt.clf()
    plt.close() 


def histogram(x, y, title="", xlabel="", ylabel="Frequency", path=None, line_legends=None,
                log=True, range_x=None, fig_size=(10,5), min_x_value=None, max_x_value=None, bins=100):
    # plt.figaspect([10, 4])
    fig, ax = plt.subplots(figsize=fig_size, tight_layout=True)
    n1,x1,_1  = plt.hist(x, histtype=u'step', color="#009edd", bins=bins, range=range_x, alpha = 0.01, label="random")
    n2,x2,_2  = plt.hist(y, histtype=u'step', color="#ff9edd", bins=bins, range=range_x, alpha = 0.01, label="based on num pub")
    
    bin_centers1 = 0.5*(x1[1:]+x1[:-1])
    bin_centers2 = 0.5*(x2[1:]+x2[:-1])
     
    plt.plot(bin_centers1,n1) 
    plt.plot(bin_centers2,n2) 

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(labels=labels)

    plt.xlim(left=0.)
    if min_x_value is not None:
        plt.xlim(left=min_x_value)
    if max_x_value is not None:
        plt.xlim(right=max_x_value)
    # plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)    
    plt.grid(True)
    if log == True:
        plt.yscale('log')    
    if path:
        plt.savefig(path, dpi=900)
    else:
        plt.show()
    plt.clf()
    plt.close() 

def histogram3(x, y, z, title="", xlabel="", ylabel="Frequency", path=None, line_legends=None,
                log=True, range_x=None, fig_size=(10,5), min_x_value=None, max_x_value=None, num_bins=100):
    # plt.figaspect([10, 4])
    fig, ax = plt.subplots(1, 1, figsize=fig_size, tight_layout=True)
    n1, x1,_1  = ax.hist(x, histtype=u'step', color="#ffffff", bins=num_bins, range=range_x, alpha = 0.0)
    n2, x2,_2  = ax.hist(y, histtype=u'step', color="#ffffff", bins=num_bins, range=range_x, alpha = 0.0)
    # n3, x3,_3  = ax.hist(z, histtype=u'step', color="#00ffff", bins=num_bins, range=range_x, alpha = 0.0)
    
    bin_centers1 = 0.5*(x1[1:]+x1[:-1])
    bin_centers2 = 0.5*(x2[1:]+x2[:-1])
    # bin_centers3 = 0.5*(x3[1:]+x3[:-1])
     
    ax.plot(bin_centers1,n1, color='m', label="original") 
    ax.plot(bin_centers2,n2, color='y', label="ts") 
    # ax.plot(bin_centers3,n3, color='y', label="ts") 

    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(labels="x1")
    ax.legend()

    plt.xlim(left=0.)
    if min_x_value is not None:
        plt.xlim(left=min_x_value)
    if max_x_value is not None:
        plt.xlim(right=max_x_value)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)    
    plt.grid(True)
    if log == True:
        plt.yscale('log')    
    if path:
        plt.savefig(path, dpi=900)
    else:
        plt.show()
    plt.clf()
    plt.close() 

def histogram3_old(x, y, z, title="", xlabel="", ylabel="Frequency", path=None, line_legends=None,
                log=True, range_x=None, fig_size=(10,5), min_x_value=None, max_x_value=None, num_bins=100):
    # plt.figaspect([10, 4])
    fig, ax = plt.subplots(figsize=fig_size, tight_layout=True)
    n1,x1,_1  = plt.hist(x, histtype=u'step', color="#ffffff", bins=num_bins, range=range_x, alpha = 0.5, label="original")
    n2,x2,_2  = plt.hist(y, histtype=u'step', color="#ffffff", bins=num_bins, range=range_x, alpha = 0.5, label="g_prime")
    n3,x3,_3  = plt.hist(z, histtype=u'step', color="#00ffff", bins=num_bins, range=range_x, alpha = 0.5, label="g")
    
    bin_centers1 = 0.5*(x1[1:]+x1[:-1])
    bin_centers2 = 0.5*(x2[1:]+x2[:-1])
    bin_centers3 = 0.5*(x3[1:]+x3[:-1])
     
    plt.plot(bin_centers1,n1, color='blue') 
    plt.plot(bin_centers2,n2, color='yellow') 
    plt.plot(bin_centers3,n3, color='red') 

    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(labels="x1")
    ax.legend()

    plt.xlim(left=0.)
    if min_x_value is not None:
        plt.xlim(left=min_x_value)
    if max_x_value is not None:
        plt.xlim(right=max_x_value)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)    
    plt.grid(True)
    if log == True:
        plt.yscale('log')    
    if path:
        plt.savefig(path, dpi=900)
    else:
        plt.show()
    plt.clf()
    plt.close() 

def hist(x, y, title="", xlabel="", ylabel="Frequency", path=None, line_legends=None,
                log=True, range_x=None, fig_size=(10,5), min_x_value=None, max_x_value=None, bins_=100):

    _, ax = plt.subplots(figsize=fig_size, tight_layout=True)

    # _, b, _ = plt.hist(x, range=range_x, bins=bins_)
    # _ = plt.hist(y, bins=b, alpha=0.5)

    plt.hist([x,y], bins=bins_, range=range_x, label=["original","perturbed"])

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(labels=labels)

    plt.xlim(left=0.)
    if max_x_value:
        plt.xlim(right=max_x_value)
    if min_x_value:
        plt.xlim(left=min_x_value)
    # plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)    
    plt.grid(True)
    if log == True:
        plt.yscale('log')    
    if path:
        plt.savefig(path, dpi=900)
    else:
        plt.show()
    plt.clf()
    plt.close() 

def bar_plot2(x, y, bins, title="", xlabel="", ylabel="", path=None, log=True, range_x=None,
            labels=['original', 'perturbed'], fig_size=(10,5), max_x_value=None):
    # plt.figaspect([10, 4])

    fig, ax = plt.subplots(figsize=fig_size, tight_layout=True)
    width = 0.3

    plt.bar(bins, x, width, label=labels[0])
    plt.bar(bins + width, y, width, label=labels[1])
    
    # ax.bar(y, x)
    # ax.bar(y, x)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(labels=labels)

    # plt.xlim(left=0.)
    if max_x_value:
        plt.xlim(right=max_x_value)
    # plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)    
    plt.grid(True)
    if log == True:
        plt.yscale('log')    
    if path:
        plt.savefig(path, dpi=900)
    else:
        plt.show()
    plt.clf()
    plt.close() 

def bar_plot3(x, y, z, bins, title="", xlabel="", ylabel="", path=None, log=True, range_x=None,
            labels=['original', 'perturbed'], fig_size=(10,5), max_x_value=None):
    
    fig, ax = plt.subplots(figsize=fig_size, tight_layout=True)
    width = 0.3

    plt.bar(bins - width, x, width, label=labels[0])
    plt.bar(bins, y, width, label=labels[1])
    plt.bar(bins + width, z, width, label=labels[2])
    
    # ax.bar(y, x)
    # ax.bar(y, x)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(labels=labels)

    # plt.xlim(left=0.)
    if max_x_value:
        plt.xlim(right=max_x_value)
    # plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)    
    plt.grid(True)
    if log == True:
        plt.yscale('log')    
    if path:
        plt.savefig(path, dpi=900)
    else:
        plt.show()
    plt.clf()
    plt.close() 

def bar_plot(x, y, title="", xlabel="", ylabel="", path=None, 
                log=False, range_x=None, fig_size=(10,5), max_x_value=None, num_bins=50):
    # plt.figaspect([10, 4])
    fig, ax = plt.subplots(figsize=fig_size)
    
    ax.bar(y, x, color="#009edd")

    # plt.xlim(left=0.)
    if max_x_value:
        plt.xlim(right=max_x_value)
    # plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)    
    plt.grid(True)
    if log == True:
        plt.yscale('log')    
    if path:
        plt.savefig(path, dpi=900)
    else:
        plt.show()
    plt.clf()
    plt.close() 

def box_plot(x, title="", xlabel="", ylabel="", path=None, outliers=False,
                log=True, range_x=None, fig_size=(10,5), max_x_value=None, num_bins=50):
    # plt.figaspect([10, 4])
    fig, ax = plt.subplots(figsize=fig_size, tight_layout=True)

    ax.boxplot(x, showfliers=outliers)

    plt.xlim(left=0.)
    if max_x_value:
        plt.xlim(right=max_x_value)
    # plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)    
    plt.grid(True)
    if log == True:
        plt.yscale('log')    
    if path:
        plt.savefig(path, dpi=900)
    else:
        plt.show()
    plt.clf()
    plt.close() 

def scatter_plot(x, y, xlog=False, ylog=False, xlabel=None, ylabel=None, figsize=(12, 6), 
                xlim=None, ylim=None, labels=None, path=None):
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    ax.scatter(x, y, color="blue")

    # gridlines = ax.get_ygridlines()
    # gridlines[5].set_linewidth(2.5)
    # ax.plot(x, y, color="blue")

    if(labels is not None):
        for i in range(len(labels)):
            x_ = x[i]
            y_ = y[i]
            plt.text(x_ * (1 + 0.002), y_ * (1 + 0.002) , labels[i], fontsize=8)
    if(xlim):
        plt.xlim(xlim)
    if(ylim):
        plt.ylim(ylim)
    if(xlog):
        plt.xscale('log')
    if(ylog):
        plt.yscale('log')
    if(xlabel):
        plt.xlabel(xlabel)
    if(ylabel):
        plt.ylabel(ylabel)
    if path:
        dir_path = os.path.dirname(os.path.realpath(path))
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(path, dpi=900)
        print('Graphic saved at: ' + path)
    else:
        plt.show()
    # plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.show()
    plt.clf()
    plt.close() 

def scatter_plot4D(x, y, z, c, xlog=False, ylog=False, xlabel=None, ylabel=None, zlabel=None, figsize=(12, 6), 
                xlim=None, ylim=None, labels=None, path=None):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(16, -19)
    img = ax.scatter(x, y, z, c=c, s=100, cmap= cm.get_cmap('jet_r'))
    fig.colorbar(img)

    # fig, ax = plt.subplots(figsize=figsize, tight_layout=True, projection='3d')
    # ax.scatter(x, y, z, c=c, cmap=plt.hot()) #color="blue")

    # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    # scamap = plt.cm.ScalarMappable(cmap='inferno')
    # fcolors = scamap.to_rgba(c[0].astype('float'))
    # ax.plot_surface(x, y, z, facecolors=fcolors, cmap='inferno')
    # fig.colorbar(scamap)

    # plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.88, wspace=0.2, hspace=0.2)

    if(labels is not None):
        for i in range(len(labels)):
            x_ = x[i]
            y_ = y[i]
            # z_ = z[i]
            plt.text(   x_ * (1 + 0.002), y_ * (1 + 0.002) , labels[i], fontsize=8)
    if(xlim):
        plt.xlim(xlim)
    if(ylim):
        plt.ylim(ylim)
    if(xlog):
        plt.xscale('log')
    if(ylog):
        plt.yscale('log')
    if(xlabel):
        ax.set_xlabel(xlabel)
    if(ylabel):
        ax.set_ylabel(ylabel)
    if(zlabel):
        ax.set_zlabel(zlabel)
    if path:
        dir_path = os.path.dirname(os.path.realpath(path))
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(path, dpi=900)
        print('Graphic saved at: ' + path)
    else:
        plt.show()
    # plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.show()
    plt.clf()
    plt.close() 

def surface_plot4D(x, y, z, c, xlog=False, ylog=False, xlabel=None, ylabel=None, zlabel=None, figsize=(12, 6), 
                xlim=None, ylim=None, labels=None, path=None):

    index_x = 0
    index_y = 1
    index_z = 2
    index_c = 3

    list_name_variables = ['x', 'y', 'z', 'c']
    triangles = mtri.Triangulation(x, y).triangles

    choice_calcuation_colors = 1
    if choice_calcuation_colors == 1: # Mean of the "c" values of the 3 pt of the triangle
        colors = np.mean( [c[triangles[:,0]], c[triangles[:,1]], c[triangles[:,2]]], axis = 0);
    elif choice_calcuation_colors == 2: # Mediane of the "c" values of the 3 pt of the triangle
        colors = np.median( [c[triangles[:,0]], c[triangles[:,1]], c[triangles[:,2]]], axis = 0);
    elif choice_calcuation_colors == 3: # Max of the "c" values of the 3 pt of the triangle
        colors = np.max( [c[triangles[:,0]], c[triangles[:,1]], c[triangles[:,2]]], axis = 0);

    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.view_init(16, -19)
    triang = mtri.Triangulation(x, y, triangles)
    surf = ax.plot_trisurf(triang, z, cmap = cm.get_cmap('Greys'), shade=False, linewidth=0.2)
    surf.set_array(colors)
    surf.autoscale()

    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(list_name_variables[index_c], rotation = 270)

    ax.set_xlabel(list_name_variables[index_x])
    ax.set_ylabel(list_name_variables[index_y])
    ax.set_zlabel(list_name_variables[index_z])

    # img = ax.scatter(x, y, z, c=-c[0].astype('float'), s=100, cmap=plt.jet())
    # fig.colorbar(img)

    if(xlim):
        plt.xlim(xlim)
    if(ylim):
        plt.ylim(ylim)
    if(xlog):
        plt.xscale('log')
    if(ylog):
        plt.yscale('log')
    # if(xlabel):
    #     ax.set_xlabel(xlabel)
    # if(ylabel):
    #     ax.set_ylabel(ylabel)
    # if(zlabel):
    #     ax.set_zlabel(zlabel)
    if path:
        dir_path = os.path.dirname(os.path.realpath(path))
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(path, dpi=900)
        print('Graphic saved at: ' + path)
    else:
        plt.show()
    # plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.show()
    plt.clf()
    plt.close() 