# This code is submitted as work done so far for the NASA 2020 space apps challenge.
# The aim of this project was to use historical data, such as satellite data, to build
# a prediction model that could predict algae blooms in ocean spaces.
# The approach this project takes is to build simulations of oceans, and then fine tune
# these simulations to be as realistic as possible by matching them with real world data.
# With an accurate simulation, we can then run it into the future and predict algae blooms
# accordingly. The nature of the simulation means this project can double as an
# educational tool, given the ability to visualise it.

# Given the time required to run simulations on a large and accurate enough scale, this code is
# incomplete in many parts. This file shows what has been achieved so far, and lays
# out a skeleton for future work.



# A large part of this project was the VEROS simulations. These are given in a separate file.
# These simulations, as well as a lot of NASA satellite imagery, is output in NETCDF format.

import datetime as dt  # Python standard library datetime  module
import numpy as np
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import matplotlib.pyplot as plt
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid

# This opens .nc files, as outputted by VEROS simulations or NASA Earth Data.

nc_f = 'sal1.nc'  # Your filename
nc_fid = Dataset(nc_f, 'r', format='NETCDF')
nc_vars = [var for var in nc_fid.variables]

# The 3 functions below all assume a cartesian coordinate system. In VEROS, this means that 
# vs.coord_degree = False

# This can be used to visualise a 2D plane, such as a NASA satellite image.
def visualise_2D(data):
	plane = data

	plt.imshow(plane, interpolation='nearest')
	plt.show()

	plt.savefig('ocean.png')

# This can be used to visualise a 2D plane, such as the salinity of the ocean's surface, over time.
# You could also use this method to visualise a slice of a 3D structure by passing a sliced numpy
# array as input.
def visualise_2D_time(data):
	for s in range(data.shape[0]):
		plane = data[s,:,:]

		plt.imshow(plane, interpolation='nearest')
		plt.show()

		plt.savefig('bar' + str(s) + '.png')

# This can be used to visualise a 3D cube of voxels. At the moment, it means you can only see the outsides
# of a shape, but if the data ever made it necessary, this method could be modified with a high pass filter
# so that more interesting internal shapes were more apparent. 
def visualise_3D_time(data):
	for s in range(data.shape[0]):
		plane = data[s,:,:,:]
		# Could remove this, this was just convenient for the simulations that were run
		# during this challenge
		plane = np.rot90(plane, 1, axes=(1,2))

		fig = plt.figure()
		ax = fig.gca(projection='3d')

		cmap = plt.get_cmap("viridis")
		norm= plt.Normalize(plane.min(), plane.max())
		ax.view_init(elev=1, azim=1)
		ax.voxels(plane, facecolors=cmap(norm(plane)), edgecolor="black")

		# plt.show()

		plt.savefig('ocean' + str(s) + '.png')

# To fine tune ocean model parameters, some optimisation technique, perhaps evolutionary algorithms, can
# be used. In most cases, we would be dealing with a given topology (like the sea around the UAE), as well
# as other ocean variables. The topology may well need to be fine tuned given that it is likely going to
# be heavily simplified and can affect the simulation's ability to be accurate.

# The most crucial part of such an approach would probably be the loss function. Given NASA satellite images,
# and the equivalent output from the simulation, determining the error between the simulation and the real
# data is the most important part of producing an accurate simulation. As such, here is a proposed loss
# function:

# we may want to emphasise or de-emphasise different variables in the loss function.
loss_multipliers = {"salt" : 2.0, "temp" : 0.5}

# we may want to reduce the significance of errors as they get deeper below the ocean surface. If so, this is an
# example of how this could be achieved. In this example, the importance of an error is inversely proportional
# to the square of the depth at which it occurred.
def depth_multiplier(depth):
	return 1.0 / (depth ** 2.0)

# this method takes two equally sized ocean sections, at the same date and time, and calculates the error
# between them. It uses the two multipliers given, along with the square of the error, to calculate the
# final loss which would go on to guide an optimisation algorithm such as an evolutionary algorithm.
def loss(simulation, real):

	loss = 0

	for v in real.variables:
		real_v = real[v]
		sim_v = simulation[v]
		for r in range(real.shape[0]):
			s_row = sim_v[r, :, :]
			r_row = real_v[r, :, :]
			for c in range(real.shape[1]):
				s_col = s_row[c, :]
				r_col = r_row[c, :]
				for d in range(len(r_col)):
					multiplier = 1
					if v in loss_multipliers.keys():
						multiplier = loss_multipliers[v]

					loss += multiplier * depth_multiplier(d) * ((r_col[d] - s_col[d])**2)

visualise_2D(Dataset('sal1.nc', 'r', format='NETCDF')["smap_sss"])
visualise_3D_time(Dataset('sal17.nc', 'r', format='NETCDF')["salt"])