#author: Daniel Johnson

# In this file, we read in RayGalGroup weak lensing simulation data, extract a certain number of lines of sight, interpolate to calculate the shear at the specific redshifts associated with each of the SLACS lenses, and calculate and store the line of sight shear for these lines of sight. This requires the RayGalGroup convergence and shear HEALPIX maps to be downloaded - these can be found at https://cosmo.obspm.fr/public-datasets/raygalgroupsims-relativistic-halo-catalogs/healpix-maps/.

######################################################## Imports ##########################################################

import numpy as np
import os
import random
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from astropy.io import fits
import healpy as hp
from multiprocessing import Pool, cpu_count
import csv
import pickle

iterations = 5000                #the number of lines of sight to generate per redshift set
project_name = 'RayGalGroup'

################################################ Cosmology #############################################################

from astropy.cosmology import WMAP7 as cosmo

def chi(z):                                             #the comoving distance at redshift z
    return cosmo.comoving_distance(z).value

###################################### Saving data from fits files #######################################################

path_data = '../weak_lensing_maps/'     #location of the weak lensing simulation files

# All the redshift values to be used for different depth maps
all_redshifts = ['0.200000', '0.250000', '0.300000', '0.350000', '0.450000', '0.700000', '0.950000', '1.200000', '1.450000', '1.700000', '1.950000', '3.000000','4.300000','6.900000','8.200000','9.500000']

# Filename numbers corresponding to the different depths/resolutions (2048 for full-sky, 4096 for intermediate, etc.)
nsides = ['2048', '4096', '8192']

def process_z(z_str, i, nside_out=2048):
    z_val = round(float(z_str), 2)
    chi_val = chi(z_val)

    # Choose appropriate nside for input file
    if z_val < 0.7:
        nside_in = 2048
    elif z_val < 3:
        nside_in = 4096
    else:
        nside_in = 8192

    filename = (
        f'catalog_boxlen2625_n4096_lcdmw7v2_00000_nside{nside_in}_{i}_lambda_normal_0.000100_z_{z_str}_homogeneous.fits'
    )

    with fits.open(path_data + filename) as hdul:
        values = np.array(hdul[1].data.field(0))

    # Downsample to target resolution if needed
    if nside_in != nside_out:
        values = hp.ud_grade(values, nside_out=nside_out, order_in='RING', order_out='RING')

    # Mask invalid pixels
    values = values[values > -1e25] 

    print(f'Loaded z = {z_str}, i = {i}')
    return z_val, chi_val, values  # return downsampled map



# Prepare the input arguments: all combinations of (z_str, i)
args = [(z, i) for z in all_redshifts for i in range(3)]  # i = 0, 1, 2

if __name__ == '__main__':
    with Pool(cpu_count()) as pool:
        results = pool.starmap(process_z, args)  # pass two arguments to process_z

    # Filter out failed loads
    results = [r for r in results if r is not None]

    # Separate results into kappa, shear1, shear2
    kappa_maps = []
    shear1_maps = []
    shear2_maps = []
    z_values = []
    comoving_distances = []

    for (z, i), (z_val, com_dist, map_data) in zip(args, results):
        if i == 0:
            kappa_maps.append(map_data)
        elif i == 1:
            shear1_maps.append(map_data)
        elif i == 2:
            shear2_maps.append(map_data)

        # Only append z and comoving_distance once per z (avoid duplicates)
        if i == 0:
            z_values.append(z_val)
            comoving_distances.append(com_dist)

    print("All maps loaded.")

################################################ Finding usable indices ###########################################################

# Initialize an empty list to store indices of non-empty lines of sight (LOS) for each depth
indices_los = []


# Assume if a pixel is empty in one subdepth, it's empty in all subdepths of that depth
for j, pixel_value in enumerate(kappa_maps[len(kappa_maps)-1]):
    if pixel_value > -1e25:  # Non-empty pixels
        indices_los.append(j)

print("Found all non-empty pixel indices for all depths.")

######################################################## Useful functions #########################################################

# Symmetric derivative function
def derivative(func, x, h=1e-5):
    return (func(x + h) - func(x - h)) / (2 * h)

# Second derivative function
def scnd_derivative(func, x, h=1e-5):
    return (derivative(func, x + h, h) - derivative(func, x - h, h)) / (2 * h)

# Function to calculate the ds terms using integration
def calc_ds(interpolated_function, z_s, z_d):
    def second_derivative(chii):
        return scnd_derivative(interpolated_function, chii)

    def weight_function(chii):
        return ((chi(z_s) - chii) * (chii - chi(z_d))) / (chii * (chi(z_s) - chi(z_d)))

    def integrand_ds(chii):
        return second_derivative(chii) * weight_function(chii)
    
    return quad(integrand_ds, chi(z_d), chi(z_s), epsabs=1e-7, epsrel=1e-7)[0]

######################################################## storing line of sight effects #########################################################

def los_effects(z_lens, z_source):

    # Random valid LOS index
    index_los = random.choice(indices_los)

    # Collect relevant convergence and comoving distance data

    comoving2interpolate = [0]
    
    convergence2interpolate = [0]
    convergenceproduct2interpolate = [0]

    shear1interpolate = [0]
    shear1product2interpolate = [0]
    
    shear2interpolate = [0]
    shear2product2interpolate = [0]

    #loop through the redshifts
    for i in range(len(kappa_maps)):
        
        comoving2interpolate.append(comoving_distances[i])
        
        convergence2interpolate.append(kappa_maps[i][index_los])
        convergenceproduct2interpolate.append(comoving_distances[i] * kappa_maps[i][index_los])
        
        shear1interpolate.append(shear1_maps[i][index_los])
        shear1product2interpolate.append(comoving_distances[i] * shear1_maps[i][index_los])
        
        shear2interpolate.append(shear2_maps[i][index_los])
        shear2product2interpolate.append(comoving_distances[i] * shear2_maps[i][index_los])

    # Interpolated function
    convergence_interpolated_product = PchipInterpolator(comoving2interpolate, convergenceproduct2interpolate)
    shear1_interpolated_product = PchipInterpolator(comoving2interpolate, shear1product2interpolate)
    shear2_interpolated_product = PchipInterpolator(comoving2interpolate, shear2product2interpolate)

    # Calculate os and od terms
    kappa_os = convergence_interpolated_product(chi(z_source)) / chi(z_source)
    kappa_od = convergence_interpolated_product(chi(z_lens)) / chi(z_lens)
    
    shear1_os = shear1_interpolated_product(chi(z_source)) / chi(z_source)
    shear1_od = shear1_interpolated_product(chi(z_lens)) / chi(z_lens)
    
    shear2_os = shear2_interpolated_product(chi(z_source)) / chi(z_source)
    shear2_od = shear2_interpolated_product(chi(z_lens)) / chi(z_lens)

    # Calculate ds terms using calc_ds
    kappa_ds = calc_ds(convergence_interpolated_product, z_source, z_lens)

    shear1_ds = calc_ds(shear1_interpolated_product, z_source, z_lens)
    
    shear2_ds = calc_ds(shear2_interpolated_product, z_source, z_lens)

    shear1_LOS = shear1_os + shear1_od - shear1_ds
    shear2_LOS = shear2_os + shear2_od - shear2_ds

    shear_mag_LOS = np.sqrt(shear1_LOS**2 + shear2_LOS**2)
    
    terms = {'kappa_os': kappa_os,
             'kappa_od': kappa_od,
             'kappa_ds': kappa_ds,
             'shear1_os': shear1_os,
             'shear1_od': shear1_od,
             'shear1_ds': shear1_ds,
             'shear2_os': shear2_os,
             'shear2_od': shear2_od,
             'shear2_ds': shear2_ds,
             'shear1_LOS': shear1_LOS,
             'shear2_LOS': shear2_LOS,
             'shear_mag_LOS': shear_mag_LOS
                }
    
    return terms

#################################### the data #######################

# Read CSV into dictionary
data_dict = {}

with open("SLACS_measurements.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)  # Each row becomes a dict
    for row in reader:
        for key, value in row.items():
            data_dict.setdefault(key, []).append(value)

#################################### Saving a sample of line of sight effects #############################################

def process_lens(name, z_l, z_s, project_name, iterations):
    """Compute and save sampled_effect for a single lens."""
    output_dir = f"Data/{project_name}/"
    output_file = os.path.join(output_dir, name)

    os.makedirs(output_dir, exist_ok=True)
    
    sampled_effect = []
    for i in range(iterations):
        sampled_effect.append(los_effects(float(z_l), float(z_s)))

    with open(output_file, "wb") as f:
        pickle.dump(sampled_effect, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = [
        (data_dict['Name'][j], data_dict['z_l'][j], data_dict['z_s'][j], project_name, iterations)
        for j in range(len(data_dict['z_l']))
    ]

    with Pool(cpu_count()) as pool:
        results = pool.starmap(process_lens, args)
