ge_lightcone
============

A code to generate 2D slices from Gadget-2 type simulation snapshots.

Direction of the Line of Sight (LoS) can be defined.

Output files can be found in dictionary slices/

Note that it may be needed to change the work path in the file write.c, and regenerate the .so lib.

Format of the output file:

First five blocks stores basic information of the slice, they are:
(int) num_particles (double) mass_particles (double) boxsize (units: kpc) (double) distance (units: kpc) (double) redshift.

The following blocks stores the position of the particles, whose unit is Mpc.

Note that the slice may not be a square if the LoS do not be chosen along one axis of the snapshot, which is recommended to avoid repeated structure.

Bug report: suqikuai777@gmail.com
