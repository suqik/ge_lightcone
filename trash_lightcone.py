from astropy.cosmology import FlatLambdaCDM as LCDM, z_at_value
import astropy.units as u
from astropy import cosmology
from scipy.interpolate import splev, splrep
from scipy.integrate import quad
import h5py
import numpy as np
import readgadget
from ctypes	import *
import datetime
import random
import gc

path = '/home/suqikuai777/Program/ge_lightcone/'
prefix = 'snapshot' # prefix of snapshot file. E.g.: file name "snapshot_001" should set prefix='snapshot'

CFunc = CDLL('/home/suqikuai777/Program/ge_lightcone/libwrite.so')

# # test code
# num_mock = int(1e8)

def read_header(snap):
	snapshot_fname = path+prefix+'_{0:03d}'.format(snap)
	return readgadget.header(snapshot_fname)

# def ge_redshift_boundary(snap_num):
# 	zz = np.zeros(snap_num)
# 	for snap in range(snap_num):
# 		head = read_header(14-snap) # z of snapshot_014 is 0
# 		zz[snap]= head.redshift

# 	z_max = zz[:-1:] + (zz[1::] - zz[:-1:])/2
# 	z_max = np.append(z_max, 2*zz[-1] - z_max[-1])
	
# 	return z_max

def z_distance(Omega0, OmegaL, h0, opt='dist_to_z'):
	sol = 299792.458 # speed of light, km/s
	z_range = np.linspace(0.0, 1.0, 11) # can be tested, initially choose 10 points
	D_range = np.array([], dtype=float)

	for z_u in z_range:
		D_range = np.append(D_range, quad(lambda x: sol / (100*h0*np.sqrt(Omega0 * (1+x) ** 3 + OmegaL)), 0.0, z_u)[0])
	D_range *= h0 # units from Mpc to Mpc/h

	if opt == 'dist_to_z':
		spl = splrep(D_range, z_range, s=0.0)
		return spl
	if opt == 'z_to_dist':
		spl = splrep(z_range, D_range, s=0.0)
		return spl


def choose_pos_vel(ids, snapshot_label, typs):
	'''
	choose the positions/velocities of particles in near snapshots.
	params:
	ids: int array. ids of the particles needed.
	snapshot_label: int array. label of snapshot. note that snapshot_label=1 represents first snapshot, 
	                i.e. the snapshot at z=1*snaps_interval (=0.1 in this case).
	types: character string. can be "POS " or "VEL ", represents choose position or velocity, respectively.
	'''

	# # test code
	# global num_mock
	# id1 = list(range(num_mock))
	# np.random.shuffle(id1)
	# id1 = np.array(id1)
	snapshot_label = snap_label_set[int(snapshot_label)]
	id1 = readgadget.read_block(path+prefix+"_{0:03d}".format(snapshot_label), "ID  ", [1])-1

	id1_dict = np.zeros_like(id1)
	for i in range(len(id1)):
		id1_dict[int(id1[i])] = i
	
	del id1
	gc.collect()

	id1_dict = id1_dict[ids]

	# # test code
	# type1 = (np.random.rand(num_mock*3)*400).reshape(num_mock, 3)
	type1 = readgadget.read_block(path+prefix+"_{0:03d}".format(snapshot_label), typs, [1])[:,0:2]/1e3
	type1 = type1[id1_dict]

	return type1

def adjust_position(pos_i, vel_i, pos_f, vel_f, z_i, z_f, z_real):
	# calculate physical time at certain redshift. Note that the start time is not important, 
	# only depends on the time intervals between z_initial and z_final.
	t1 = cosmo.age(z_i) # time at initial redshift, Gyr
	t2 = cosmo.age(z_f) # time at final redshift, Gyr
	t_real = cosmo.age(z_real)
	
	# change the units of velocity, from km/s to Mpc/h/Gyr
	vel_i = (vel_i*(u.km/u.s)).to(u.Mpc/u.Gyr)*h0
	vel_f = (vel_f*(u.km/u.s)).to(u.Mpc/u.Gyr)*h0
	pos_i = pos_i*u.Mpc
	pos_f = pos_f*u.Mpc
	
	# cubic polynomial interpolation function
	pos = (((t1 - t2) * (vel_i + vel_f) - 2 * pos_i + 2 * pos_f) / (t1 - t2) ** 3) * t_real ** 3 + \
			((-t1 ** 2 * (vel_i + 2 * vel_f) + t1 * (t2 * (vel_f - vel_i) + 3 * (pos_i - pos_f)) \
		 + t2 * (2 * t2 * vel_i + t2 * vel_f + 3 * (pos_i - pos_f))) / (t1 - t2) ** 3) * t_real ** 2 + \
		((-t2 ** 3 * vel_i + t1 ** 3 * vel_f + t1 ** 2 * t2 * (2 * vel_i + vel_f) - t1 * t2 * \
		(t2 * vel_i + 2 * t2 * vel_f + 6 * (pos_i - pos_f))) / (t1 - t2) ** 3) * t_real + \
		((t2 * (t1 * (t2 - t1) * (t2 * vel_i + t1 * vel_f) - t2 * (t2 - 3 * t1) * pos_i) + \
		t1 ** 2 * (t1 - 3 * t2) * pos_f) / (t1 - t2) ** 3)
	
	return pos

def select_particles(dist_p, d_th, boxsize, nvec, crossj, pos_interp=True):
	'''
	Process to select particles in a slice.
	Params:
		dist_p: Float. Distance of center of the slice from observer. Unit in Mpc/h.
		d_th: Float. Thick of the slice. Unit in Mpc/h.
		boxsize: Float. Boxsize of simulation box. Unit in Mpc/h.
		snap0: Character string. Path of the snapshot file at z=0. Used in constructing the whole universe.
		nvec: 3-D numpy array. Direction of line of sight.
		crossj: Bool. Justice if the slice crosses the snapshot.
	'''
	
	# define the direction of LOS
	nvec = nvec / np.sqrt(np.einsum("...i, ...i", nvec, nvec))
	nvec_p = np.array([nvec[2], 0, -nvec[0]]) # vector perpendicular to nvec
	theta = np.arctan(nvec[0] / nvec[2]) # tangent of the skewed angle

	# Determine position in simulation box
	slice_up_los = (dist_p + d_th / 2 + boxsize * np.sin(theta)) # up limit of the slice, along the LOS
	slice_low_los = (dist_p - d_th / 2)
	slice_up = slice_up_los * np.cos(theta)
	slice_low = slice_low_los * np.cos(theta)

	# input data
	global snap0
	ptype = [1] # represents CDM
	pos = readgadget.read_block(snap0, "POS ", ptype)/1e3 # Mpc
	ids = readgadget.read_block(snap0, "ID  ", ptype)-1 # starting from 0

	# # test code
	# global num_mock
	# pos = (np.random.rand(num_mock*3)*400).reshape(num_mock, 3)
	# ids = list(range(num_mock))
	# np.random.shuffle(ids)
	# ids = np.array(ids)

	# transverse the position of the particles from the box to real distance
	int_dist = float(slice_low // boxsize) * boxsize
	if crossj == True:
		pos[(pos[:, 2] > (slice_low - int_dist))] += int_dist
		pos[(pos[:, 2] < (slice_low - int_dist))] += int_dist + boxsize
		pos[:, 0:2] %= boxsize
	else:
		pos[:, 2] += int_dist
	# judge if the chosen particles cross the snapshot
	# note that z_at_value determine redshift by comoving distance in Mpc units, not Mpc/h
	global snap_label_set
	z_low_los = z_at_value(cosmo.comoving_distance, slice_low_los/h0*u.Mpc)
	z_up_los = z_at_value(cosmo.comoving_distance, slice_up_los/h0*u.Mpc)
	snapshot_label = z_low_los // snaps_interval # ith snapshot, not "snapshot_00i".

	if int(z_low_los // snaps_interval) != int(z_up_los // snaps_interval):
		print("particles z cross snapshot. separate strategy.")
		flag1 = 0
		flag2 = 0
		
		ids = ids[((np.dot(pos, nvec_p) < 0) & (slice_low_los < np.dot(pos, nvec)) & (np.dot(pos, nvec) < slice_low_los + d_th)) |\
			((np.dot(pos, nvec_p) > 0) & (slice_low_los + boxsize * np.sin(theta) < np.dot(pos, nvec)) & (np.dot(pos, nvec) < slice_up_los))]
		pos = pos[((np.dot(pos, nvec_p) < 0) & (slice_low_los < np.dot(pos, nvec)) & (np.dot(pos, nvec) < slice_low_los + d_th)) |\
			((np.dot(pos, nvec_p) > 0) & (slice_low_los + boxsize * np.sin(theta) < np.dot(pos, nvec)) & (np.dot(pos, nvec) < slice_up_los))]
		
		# Strategies: two steps. separate the slice into two parts.
		# choose ids of the part that located in the near snapshot.
		pos_p1 = pos[(np.dot(pos, nvec) < (cosmo.comoving_distance(snaps_interval * (snapshot_label + 1))*h0).value)]

		if pos_interp == True :
			if len(pos_p1) != 0 :
				flag1 = 1
				ids_p1 = ids[(np.dot(pos, nvec) < (cosmo.comoving_distance(snaps_interval * (snapshot_label + 1))*h0).value)]

				splz = z_distance(Omegam0, 1-Omegam0, h0, opt='dist_to_z')
				z_real = splev(np.dot(pos_p1, nvec), splz)
				pos_p1 = pos_p1[:, 2] # only conserve the z_direct information, used in coordinate transform

				pos1 = choose_pos_vel(ids_p1, snapshot_label, "POS ")
				vel1 = choose_pos_vel(ids_p1, snapshot_label, "VEL ")
				pos2 = choose_pos_vel(ids_p1, snapshot_label+1, "POS ")
				vel2 = choose_pos_vel(ids_p1, snapshot_label+1, "VEL ")

				del ids_p1
				gc.collect()

				real_pos = np.zeros((pos1.shape[0], pos1.shape[1]))
				for lab in range(2):
					real_pos[:,lab] = adjust_position(pos1[:,lab], vel1[:,lab], pos2[:,lab], vel2[:,lab],\
						snap_redshift_set[int(snapshot_label)], snap_redshift_set[int(snapshot_label)+1], z_real) 

				del pos1
				gc.collect()
				del vel1
				gc.collect()
				del pos2
				gc.collect()
				del vel2
				gc.collect()

				# coordinate transform
				z_real = pos_p1[np.newaxis, :]
				real_pos = np.concatenate((real_pos, z_real.T), axis=1)
				real_pos[:, 0] = real_pos[:, 0] * np.cos(theta) - real_pos[:, 2] * np.sin(theta)
				real_pos = np.delete(real_pos, -1, axis=1)
				real_pos[:, 0] = (real_pos[:, 0] + boxsize * np.cos(theta)) % (boxsize * np.cos(theta))		

				print("part 1 done.")
			del pos_p1
			gc.collect()
		else:
			if len(pos_p1) != 0:
				flag1 = 1
				pos_p1[:, 0] = pos_p1[:, 0] * np.cos(theta) - pos_p1[:, 2] * np.sin(theta)
				real_pos = pos_p1[:, 0:2]			

				del pos_p1
				gc.collect()

		# same routine again for part farther snapshot
		pos_p2 = pos[(np.dot(pos, nvec) > (cosmo.comoving_distance(snaps_interval * (snapshot_label + 1))*h0).value)]

		if pos_interp == True:
			if len(pos_p2) != 0 : 
				flag2 = 1
				ids_p2 = ids[(np.dot(pos, nvec) > (cosmo.comoving_distance(snaps_interval * (snapshot_label + 1))*h0).value)]

				splz = z_distance(Omegam0, 1-Omegam0, h0, opt='dist_to_z')
				z_real = splev(np.dot(pos_p2, nvec), splz)
				pos_p2 = pos_p2[:, 2]

				pos1 = choose_pos_vel(ids_p2, snapshot_label+1, "POS ")
				vel1 = choose_pos_vel(ids_p2, snapshot_label+1, "VEL ")
				pos2 = choose_pos_vel(ids_p2, snapshot_label+2, "POS ")
				vel2 = choose_pos_vel(ids_p2, snapshot_label+2, "VEL ")

				del ids_p2
				gc.collect()

				real_pos_2 = np.zeros((pos1.shape[0], pos1.shape[1]))
				for lab in range(2):
					real_pos_2[:,lab] = adjust_position(pos1[:,lab], vel1[:,lab], pos2[:,lab], vel2[:,lab],\
						snap_redshift_set[int(snapshot_label)+1], snap_redshift_set[int(snapshot_label)+2], z_real) 
				
				del pos1
				gc.collect()
				del vel1
				gc.collect()
				del pos2
				gc.collect()
				del vel2
				gc.collect()

				z_real = pos_p2[np.newaxis, :]
				real_pos_2 = np.concatenate((real_pos_2, z_real.T), axis=1)
				real_pos_2[:, 0] = real_pos_2[:, 0] * np.cos(theta) - real_pos_2[:, 2] * np.sin(theta)
				real_pos_2 = np.delete(real_pos_2, -1, axis=1)
				real_pos_2[:, 0] = (real_pos_2[:, 0] + boxsize * np.cos(theta)) % (boxsize * np.cos(theta))		
				print("part 2 done.")

			del pos_p2
			gc.collect()
			
		else:
			if len(pos_p2) != 0:
				pos_p2[:, 0] = pos_p2[:, 0] * np.cos(theta) - pos_p2[:, 2] * np.sin(theta)
				pos_p2 = pos_p2[:, 0:2]
				real_pos_2 = np.concatenate((real_pos, pos_p2), axis=0)

				del pos_p2
				gc.collect()

			if flag1 == 1 and flag2 == 1:
				real_pos = np.concatenate((real_pos, real_pos_2), axis=0)
			if flag1 == 0 and flag2 == 1:
				real_pos = real_pos_2
			if flag1 == 0 and flag2 == 0:
				print("Error: no particles can be found in the slice!")
				exit()
	else:
		print("particles z do not cross snapshot.")

		# choose ids of needed particles
		ids = ids[((np.dot(pos, nvec_p) < 0) & (slice_low_los < np.dot(pos, nvec)) & (np.dot(pos, nvec) < slice_low_los + d_th)) |\
			((np.dot(pos, nvec_p) > 0) & (slice_low_los + boxsize * np.sin(theta) < np.dot(pos, nvec)) & (np.dot(pos, nvec) < slice_up_los))]

		# calculate redshift (in box) of these particles
		pos = pos[((np.dot(pos, nvec_p) < 0) & (slice_low_los < np.dot(pos, nvec)) & (np.dot(pos, nvec) < slice_low_los + d_th)) |\
			((np.dot(pos, nvec_p) > 0) & (slice_low_los + boxsize * np.sin(theta) < np.dot(pos, nvec)) & (np.dot(pos, nvec) < slice_up_los))]
		splz = z_distance(Omegam0, 1-Omegam0, h0, opt='dist_to_z')
		z_real = splev(np.dot(pos, nvec), splz)
		
		if pos_interp == True:

			pos = pos[:, 2] # only conserve the z_direction information

			pos1 = choose_pos_vel(ids, snapshot_label, "POS ")
			vel1 = choose_pos_vel(ids, snapshot_label, "VEL ")
			pos2 = choose_pos_vel(ids, snapshot_label+1, "POS ")
			vel2 = choose_pos_vel(ids, snapshot_label+1, "VEL ")
			del ids
			gc.collect()

			real_pos = np.zeros((pos1.shape[0], pos1.shape[1]))
			for lab in range(2):
				real_pos[:,lab] = adjust_position(pos1[:,lab], vel1[:,lab], pos2[:,lab], vel2[:,lab],\
					snap_redshift_set[int(snapshot_label)], snap_redshift_set[int(snapshot_label)+1], z_real) 
			del pos1
			gc.collect()
			del vel1
			gc.collect()
			del pos2
			gc.collect()
			del vel2
			gc.collect()

			# splz = z_distance(Omegam0, 1-Omegam0, h0, opt='z_to_dist')
			# z_real = splev(z_real, splz) # pos_z. Note that in order to reduce the usage of store space, I do not change the name of the variable
			z_real = pos[np.newaxis, :]
			real_pos = np.concatenate((real_pos, z_real.T), axis=1)
			real_pos[:, 0] = real_pos[:, 0] * np.cos(theta) - real_pos[:, 2] * np.sin(theta)
			real_pos = np.delete(real_pos, -1, axis=1)
			real_pos[:, 0] = (real_pos[:, 0] + boxsize * np.cos(theta)) % (boxsize * np.cos(theta))
		else:
			real_pos = pos[:, 0] * np.cos(theta) - pos[:, 2] * np.sin(theta)
			del pos
			gc.collect()
		
	print("choose position done.")
	return real_pos

def store_particles_pos(p_real_pos, pmass, boxsize, slice_d, slice_redshift, slice_i):
	class io_struct(Structure):
		_fields_ = [('npart', c_int), 
					('mass', c_double), 
					('boxsize', c_double), 
					('distance', c_double), 
					('redshift', c_double)]

	output_head = io_struct(npart=int(len(p_real_pos)), mass=pmass, boxsize=boxsize*1e3, distance=slice_d*1e3, redshift=slice_redshift)
	p_real_pos_2darray = np.ctypeslib.as_ctypes(p_real_pos)

	CFunc.write_to_gadget_slice(slice_i, pointer(output_head), p_real_pos_2darray)

def single_routine(slice_label, nvector):
	# load the information of the simulation
	snap_info = read_header(9)
	step = 60.0 # Mpc/h
	thick = 10.0 # Mpc/h
	boxsize = snap_info.boxsize/1e3 # Mpc/h

	sinth = nvector[0] / (np.sqrt(nvector[0]**2+nvector[2]**2))
	costh = nvector[2] / (np.sqrt(nvector[0]**2+nvector[2]**2))
	
	#calculate redshift of slice
	distance = step * slice_label
	slice_z = z_at_value(cosmo.comoving_distance, distance/h0*u.Mpc, zmax=1) # redshift of the slice
	# judge if the slice cross the snapshot
	if float((((distance - thick / 2.0) * costh) // boxsize + 1)) * boxsize < (distance + thick / 2.0 + boxsize * sinth) * costh :
		crossif = True
	else:
		crossif = False
	
	print("slice distance is {:g} Mpc/h. corresponding redsift: {:g}".format(distance, slice_z))
	print("slice crosses snapshot or not: ", crossif)
	print("begin choosing particles positions ...")
	particles_real_pos = select_particles(distance, thick, boxsize, nvector, crossif)
	print("begin storing particles positions ...")
	store_particles_pos(particles_real_pos, snap_info.massarr[1]*1e10, boxsize, distance, slice_z, slice_label)

# Cosmological params
h0 = 0.6727
H0 = 100*h0
Omegam0 = 0.3156
Omegab0 = 0.0491
cosmo = LCDM(H0, Omegam0, Ob0=Omegab0)

# slice params
slice_num = 1
snaps_interval = 0.1 # redshift intervals
nvector = np.array([2.0, 0.0, 7.0]) # direction of LoS

#load label set of snapshots
label_0 = 14 # label of snapshot at z=0
label_end = 4 # label of snapshot at source redshift
tot_num = label_0 - label_end + 1
snap_label_set = np.linspace(label_0, label_end, tot_num, dtype=int)
snap_redshift_set = np.linspace(0.0, 1.0, tot_num, dtype=float)
snap0 = path+prefix+'_{0:03d}'.format(label_0) # name of snapshot at z=0

# Main function
for index in range(slice_num):
	index += 1
	print("begin to generate slice {:d} ...".format(index))
	single_routine(index, nvector=nvector)
