import os
import pyapr
from skimage import io as skio
import numpy as np
from glob import glob

IP_TH = 300
GRAD_TH = 30
SIGMA_TH = 230


def convert_to_apr(fpath):
    
    img = skio.imread(fpath)

    while img.ndim < 3:
        img = np.expand_dims(img, axis=0)

    apr = pyapr.APR()
    parts = pyapr.ShortParticles()
    par = pyapr.APRParameters()
    converter = pyapr.converter.ShortConverter()
    
    # Initialize APRParameters (only Ip_th, grad_th and sigma_th are set manually)    
    par.auto_parameters = False
    par.rel_error = 0.1
    par.gradient_smoothing = 2
    par.min_signal = 200
    par.noise_sd_estimate = 5
    
    #Setting the APRParameters (only Ip_th, grad_th and sigma_th) using the global values
    par.Ip_th = IP_TH
    par.grad_th = GRAD_TH
    par.sigma_th = SIGMA_TH

    converter.set_parameters(par)
    converter.set_verbose(True)
    
    # # Compute APR and sample particle values
    converter.get_apr(apr, img)
    parts.sample_image(apr, img)
    # apr, parts = pyapr.converter.get_apr_interactive(img, dtype=img.dtype, params=par, verbose=True)

    # Display the APR
    # pyapr.viewer.parts_viewer(apr, parts)

    # Write the resulting APR to file
    print("Writing APR to file ... \n")
    fpath_apr = "./data/temp/"+fpath[-8:-4]+".apr"

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Write APR and particles to file
    aprfile.open(fpath_apr, 'WRITE')
    aprfile.write_apr(apr)
    aprfile.write_particles('particles', parts)

    # Compute compression and computational ratios
    file_sz = aprfile.current_file_size_MB()
    print("APR File Size: {:7.2f} MB \n".format(file_sz))
    print("Total number of particles: {}".format(apr.total_number_particles()))
    mcr = os.path.getsize(fpath) * 1e-6 / file_sz
    cr = img.size/apr.total_number_particles()

    print("Memory Compression Ratio: {:7.2f}".format(mcr))
    print("Compuational Ratio: {:7.2f}".format(cr))

    aprfile.close()

    return 0


def main():
    
    DATA_ROOT_DIR = "./data/Fluo-C3DH-A549/01"
    
    #Ensure the temp folder is empty before converting images
    filelist = glob('./data/temp/*')
    for f in filelist:
        os.remove(f)
    
    # Read in the images
    image_paths_with_masks = glob(DATA_ROOT_DIR + '_GT/SEG/*.tif')
    for idx, images in enumerate(image_paths_with_masks):
        image_path = DATA_ROOT_DIR + '/t{}'.format(image_paths_with_masks[idx][-7:-4] + '.tif')
        print("Converting the image {} to APR format.. ".format(image_path))
        convert_to_apr(image_path)

    print("Done. \n")

if __name__ == '__main__':
    main()