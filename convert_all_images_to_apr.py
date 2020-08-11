import os
import pyapr
from skimage import io as skio
import numpy as np
from glob import glob

IP_TH = 300
GRAD_TH = 30
SIGMA_TH = 230


def convert_to_apr_point_cloud(fpath_image, fpath_mask):
    
    img = skio.imread(fpath_image)
    mask = skio.imread(fpath_mask)

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

    # Converting APR into Pointcloud
    org_dims = apr.org_dims()  # (Ny, Nx, Nz)
    #py_recon = np.empty((org_dims[2], org_dims[1], org_dims[0]), dtype=np.uint16)
    max_level = apr.level_max()

    apr_it = apr.iterator()
    
    v_min = min(parts)
    v_max = max(parts)
    point = []
    for idx, part in enumerate(parts):
        parts[idx] = int(((parts[idx] - v_min)/(v_max - v_min)) * 255)
    
    # loop over levels up to level_max-1
    for level in range(apr_it.level_min(), apr_it.level_max()+1):

        step_size = 2 ** (max_level - level)

        for z in range(apr_it.z_num(level)):
            for x in range(apr_it.x_num(level)):
                for idx in range(apr_it.begin(level, z, x), apr_it.end()):
                    y = apr_it.y(idx)  # this is slow
                    
                    y_start = y * step_size
                    x_start = x * step_size
                    z_start = z * step_size
                    
                    point += [[x_start, y_start, z_start, parts[idx], mask[z_start, x_start, y_start]]]
                    
    point_cloud = np.array(point)


    # Write the resulting APR to file
    print("Writing Point Cloud to file ... \n")
    start_idx = 8
    if "_2" in fpath_image : start_idx = start_idx + 2
    fpath_pointcloud = "./data/APRPointCloud/test/"+fpath_image[-start_idx:-4]+".txt"
    np.savetxt(fpath_pointcloud, point_cloud, delimiter = ',')
    
    # # Initialize APRFile for I/O
    # aprfile = pyapr.io.APRFile()
    # aprfile.set_read_write_tree(True)

    # # Write APR and particles to file
    # aprfile.open(fpath_apr, 'WRITE')
    # aprfile.write_apr(apr)
    # aprfile.write_particles('particles', parts)

    # # Compute compression and computational ratios
    # file_sz = aprfile.current_file_size_MB()
    # print("APR File Size: {:7.2f} MB \n".format(file_sz))
    # print("Total number of particles: {}".format(apr.total_number_particles()))
    # mcr = os.path.getsize(fpath_image) * 1e-6 / file_sz
    # cr = img.size/apr.total_number_particles()

    # print("Memory Compression Ratio: {:7.2f}".format(mcr))
    # print("Compuational Ratio: {:7.2f}".format(cr))

    # aprfile.close()

    return 0


def main():
    
    DATA_ROOT_DIR = "./data/Fluo-C3DH-A549/02"
    
    #Ensure the temp folder is empty before converting images
    filelist = glob('./data/APRPointCloud/test/*')
    for f in filelist:
        os.remove(f)
    
    # Read in the images
    image_paths_with_masks = glob(DATA_ROOT_DIR + '_GT/SEG/*.tif')
    for idx, mask_image_path in enumerate(image_paths_with_masks):
        start_idx = 7
        if "_2" in mask_image_path: start_idx = start_idx + 2
        image_path = DATA_ROOT_DIR + '/t{}'.format(image_paths_with_masks[idx][-start_idx:-4] + '.tif')
        print("Converting the image {} to APR format.. ".format(image_path))
        convert_to_apr_point_cloud(image_path, mask_image_path)

    print("Done. \n")

if __name__ == '__main__':
    main()
