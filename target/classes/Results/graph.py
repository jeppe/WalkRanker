#-*- coding:utf-8 -*-

import json
import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter 
import numpy as np
from pylab import *
import sys

def loadGraphConf(config_file = "test.json", basedir=""):
	"""
		Load dataset config file
	"""
	configs = []
	json_file = open(basedir + config_file, "r")
	
	for line in json_file.readlines():
		line = line.strip()
		json_obj = json.loads(line)
		configs.append(json_obj)

	json_file.close()

	return configs

def drawIterGraph(configs, basedir=""):
	""" Draw graphs according to the configure file
	"""

	params = [r"walk length $\tau$", r"window size $max_{\rho}$",r"buffer size $\kappa$",r"subset size $n$",r"context weight $\beta$"]
	

	# ax = axes([0.25, 0.25 ,0.7, 0.7])
	for fig_num, cf in enumerate(configs):
		fig_file_name = cf["filename"]
		xLabel = cf["xlabel"]
		yLabel = "{0}".format(cf["ylabel"])
		xTicks = cf["xticks"]
		YelpY = cf["YY"]
		EpinionY = cf["EY"]
		fig = figure(figsize = (1.15, 1.15), facecolor = "white")
		# ax=fig.add_subplot(5,3,fig_num + 1)
		ax = axes([0.25, 0.25 ,0.6, 0.6])
		model_plots = list()

		# draw learning curve for different models
		X = [i for i in range(1,len(xTicks) + 1)]
		yplot, = plot(X, YelpY, linestyle="-", color="r", marker="x", linewidth = 0.5,markersize=1.5,label="Yelp")
		eplot, = plot(X, EpinionY, linestyle="-", color="b", marker="^", linewidth = 0.5,markersize=1.5,label="Epinions")

		model_plots.append(yplot)
		model_plots.append(eplot)
		# title(dataset, fontsize = 8)
		xlabel(params[xLabel], fontsize = 4.2)
		ylabel(yLabel, fontsize = 4.2)

		ax.set_xlim(0, max(X)+1)
		ax.set_ylim(min(YelpY + EpinionY) - 0.005, max(YelpY + EpinionY) + 0.005 )

		# xticks(X,xTicks,fontsize = 3.0)
		ax.set_xticks(X)
		ax.set_xticklabels(xTicks,fontsize=3.0)
		autoscale(enable=True, axis="x")
		yticks(fontsize = 2.8)
		ax.ticklabel_format(axis='y',style = 'sci',scilimits=(0.1,1))
		rc('font', size=3.5)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['left'].set_linewidth(0.5)
		ax.spines['bottom'].set_linewidth(0.5)
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		ax.yaxis.set_label_coords(-0.25,0.5)
		ax.xaxis.set_label_coords(0.5,-0.15)
		ax.tick_params(direction='out', length=1,pad=0)

		# ax.grid()
		legend(loc = 0, fontsize = 3.4)
		# show()
	# plt.tight_layout()
		savefig(fig_file_name)

confs = loadGraphConf("graph_data.json")
drawIterGraph(confs)
