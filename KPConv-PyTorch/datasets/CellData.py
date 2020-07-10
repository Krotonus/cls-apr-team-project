#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling S3DIS dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import numpy as np
import pickle
import torch
import math
from multiprocessing import Lock

# Path functions
import sys
sys.path.append('../')

# OS functions
from os import listdir
from os.path import exists, join, isdir

# Dataset parent class
from datasets.common import PointCloudDataset
from torch.utils.data import Sampler, get_worker_info
from utils.mayavi_visu import *

from datasets.common import grid_subsampling
from utils.config import bcolors


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class CellDataset(PointCloudDataset):
    """Class to handle S3DIS dataset."""

    def __init__(self, config, set='training', use_potentials=True, load_data=True):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """
        PointCloudDataset.__init__(self, 'S3DIS')

        ############
        # Parameters
        ############

        # Dict from labels to names
        self.label_to_names = {0: 'background',
                               1: 'foreground'}

        # Initialize a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.array([])

        # Dataset folder
        self.path = './data/Fluo-C3DH-A549/01'

        # Type of task conducted on this dataset
        self.dataset_task = 'cloud_segmentation'

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Training or test set
        self.set = set

        # Using potential or random epoch generation
        self.use_potentials = use_potentials
        
        # Path of the label files
        self.train_path = '_GT/SEG'

        # List of files to process
        # Note: the variable ply_path. Might need to change it to better reflect our task.
        ply_path = self.path + self.train_path

        # Proportion of validation scenes
        # Note: Might be filenames of the different tiff files.
        self.cloud_names = ['t000', 't002', 't006', 't008', 't012', 't013',
                            't014', 't015', 't019', 't022', 't025', 't026',
                            't027', 't028', 't029',]
        self.all_splits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self.validation_split = 7

        # Number of models used per epoch
        if self.set == 'training':
            self.epoch_n = config.epoch_steps * config.batch_num
        elif self.set in ['validation', 'test', 'ERF']:
            self.epoch_n = config.validation_size * config.batch_num
        else:
            raise ValueError('Unknown set for Cell data: ', self.set)

        # Stop data is not needed
        if not load_data:
            return

        ###################
        # Prepare tiff files
        ###################

        #self.prepare_S3DIS_ply()
        #Note: Do we even need this function??
        #self.prepare_cell_data()

        ################
        # Load ply files
        ################

        # List of training files
        self.files = []
        for i, f in enumerate(self.cloud_names):
            if self.set == 'training':
                if self.all_splits[i] != self.validation_split:
                    self.files += [self.path + '/' + f + '.tif']
            elif self.set in ['validation', 'test', 'ERF']:
                if self.all_splits[i] == self.validation_split:
                    self.files += [self.path + '/' + f + '.tif']
            else:
                raise ValueError('Unknown set for Cell data: ', self.set)

        if self.set == 'training':
            self.cloud_names = [f for i, f in enumerate(self.cloud_names)
                                if self.all_splits[i] != self.validation_split]
        elif self.set in ['validation', 'test', 'ERF']:
            self.cloud_names = [f for i, f in enumerate(self.cloud_names)
                                if self.all_splits[i] == self.validation_split]

        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')
        

        # Initiate containers
        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.pot_trees = []
        self.num_clouds = 0
        self.test_proj = []
        self.validation_labels = []

        # Start loading
        self.load_subsampled_clouds()

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize potentials
        if use_potentials:
            self.potentials = []
            self.min_potentials = []
            self.argmin_potentials = []
            for i, tree in enumerate(self.pot_trees):
                self.potentials += [torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)]
                min_ind = int(torch.argmin(self.potentials[-1]))
                self.argmin_potentials += [min_ind]
                self.min_potentials += [float(self.potentials[-1][min_ind])]

            # Share potential memory
            self.argmin_potentials = torch.from_numpy(np.array(self.argmin_potentials, dtype=np.int64))
            self.min_potentials = torch.from_numpy(np.array(self.min_potentials, dtype=np.float64))
            self.argmin_potentials.share_memory_()
            self.min_potentials.share_memory_()
            for i, _ in enumerate(self.pot_trees):
                self.potentials[i].share_memory_()

            self.worker_waiting = torch.tensor([0 for _ in range(config.input_threads)], dtype=torch.int32)
            self.worker_waiting.share_memory_()
            self.epoch_inds = None
            self.epoch_i = 0

        else:
            self.potentials = None
            self.min_potentials = None
            self.argmin_potentials = None
            N = config.epoch_steps * config.batch_num
            self.epoch_inds = torch.from_numpy(np.zeros((2, N), dtype=np.int64))
            self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
            self.epoch_i.share_memory_()
            self.epoch_inds.share_memory_()

        self.worker_lock = Lock()

        # For ERF visualization, we want only one cloud per batch and no randomness
        if self.set == 'ERF':
            self.batch_limit = torch.tensor([1], dtype=torch.float32)
            self.batch_limit.share_memory_()
            np.random.seed(42)

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.cloud_names)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        if self.use_potentials:
            return self.potential_item(batch_i)
        else:
            return self.random_item(batch_i)

    def potential_item(self, batch_i, debug_workers=False):

        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0

        info = get_worker_info()
        if info is not None:
            wid = info.id
        else:
            wid = None

        while True:

            t += [time.time()]

            if debug_workers:
                message = ''
                for wi in range(info.num_workers):
                    if wi == wid:
                        message += ' {:}X{:} '.format(bcolors.FAIL, bcolors.ENDC)
                    elif self.worker_waiting[wi] == 0:
                        message += '   '
                    elif self.worker_waiting[wi] == 1:
                        message += ' | '
                    elif self.worker_waiting[wi] == 2:
                        message += ' o '
                print(message)
                self.worker_waiting[wid] = 0

            with self.worker_lock:

                if debug_workers:
                    message = ''
                    for wi in range(info.num_workers):
                        if wi == wid:
                            message += ' {:}v{:} '.format(bcolors.OKGREEN, bcolors.ENDC)
                        elif self.worker_waiting[wi] == 0:
                            message += '   '
                        elif self.worker_waiting[wi] == 1:
                            message += ' | '
                        elif self.worker_waiting[wi] == 2:
                            message += ' o '
                    print(message)
                    self.worker_waiting[wid] = 1

                # Get potential minimum
                cloud_ind = int(torch.argmin(self.min_potentials))
                point_ind = int(self.argmin_potentials[cloud_ind])

                # Get potential points from tree structure
                pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)

                # Center point of input region
                center_point = pot_points[point_ind, :].reshape(1, -1)

                # Add a small noise to center point
                if self.set != 'ERF':
                    center_point += np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape)

                # Indices of points in input region
                pot_inds, dists = self.pot_trees[cloud_ind].query_radius(center_point,
                                                                         r=self.config.in_radius,
                                                                         return_distance=True)

                d2s = np.square(dists[0])
                pot_inds = pot_inds[0]

                # Update potentials (Tukey weights)
                if self.set != 'ERF':
                    tukeys = np.square(1 - d2s / np.square(self.config.in_radius))
                    tukeys[d2s > np.square(self.config.in_radius)] = 0
                    tukeys = torch.from_numpy(tukeys)
                    self.potentials[cloud_ind][pot_inds] += tukeys
                    min_ind = torch.argmin(self.potentials[cloud_ind])
                    self.min_potentials[[cloud_ind]] = self.potentials[cloud_ind][min_ind]
                    self.argmin_potentials[[cloud_ind]] = min_ind

            t += [time.time()]

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)


            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            t += [time.time()]

            # Number collected
            n = input_inds.shape[0]

            # Collect labels and colors
            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_ind][input_inds]
            if self.set in ['test', 'ERF']:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])

            t += [time.time()]

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                input_colors *= 0

            # Get original height as additional feature
            input_features = input_colors.astype(np.float32)#np.hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np.float32)

            t += [time.time()]

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

            # Randomly drop some points (act as an augmentation process and a safety for GPU memory consumption)
            # if n > int(self.batch_limit):
            #    input_inds = np.random.choice(input_inds, size=int(self.batch_limit) - 1, replace=False)
            #    n = input_inds.shape[0]

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        #Edited here
        if self.config.in_features_dim == 1 or self.config.in_features_dim == 2:
            stacked_features = features[:, :1]
        #Edited here
        elif self.config.in_features_dim == 4:
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config.in_features_dim == 5:
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        t += [time.time()]

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)

        t += [time.time()]

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        if debug_workers:
            message = ''
            for wi in range(info.num_workers):
                if wi == wid:
                    message += ' {:}0{:} '.format(bcolors.OKBLUE, bcolors.ENDC)
                elif self.worker_waiting[wi] == 0:
                    message += '   '
                elif self.worker_waiting[wi] == 1:
                    message += ' | '
                elif self.worker_waiting[wi] == 2:
                    message += ' o '
            print(message)
            self.worker_waiting[wid] = 2

        t += [time.time()]

        # Display timings
        debugT = False
        if debugT:
            print('\n************************\n')
            print('Timings:')
            ti = 0
            N = 5
            mess = 'Init ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Pots ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Sphere .... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Collect ... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Augment ... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += N * (len(stack_lengths) - 1) + 1
            print('concat .... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('input ..... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('stack ..... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('\n************************\n')
        return input_list

    def random_item(self, batch_i):

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0

        while True:

            with self.worker_lock:

                # Get potential minimum
                cloud_ind = int(self.epoch_inds[0, self.epoch_i])
                point_ind = int(self.epoch_inds[1, self.epoch_i])

                # Update epoch indice
                self.epoch_i += 1

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)

            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)

            # Add a small noise to center point
            if self.set != 'ERF':
                center_point += np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape)

            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            # Number collected
            n = input_inds.shape[0]

            # Collect labels and colors
            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_ind][input_inds]
            if self.set in ['test', 'ERF']:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                input_colors *= 0

            # Get original height as additional feature
            input_features = np.hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np.float32)

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

            # Randomly drop some points (act as an augmentation process and a safety for GPU memory consumption)
            # if n > int(self.batch_limit):
            #    input_inds = np.random.choice(input_inds, size=int(self.batch_limit) - 1, replace=False)
            #    n = input_inds.shape[0]

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config.in_features_dim == 5:
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)
        
        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        return input_list
    
    def prepare_cell_data(self):

        print('\nPreparing cell data tiff files')
        t0 = time.time()

        # Folder for the ply files
        ply_path = join(self.path, self.train_path)
        if not exists(ply_path):
            makedirs(ply_path)

        for cloud_name in self.cloud_names:

            # Pass if the cloud has already been computed
            cloud_file = join(ply_path, cloud_name + '.ply')
            if exists(cloud_file):
                continue

            # Get rooms of the current cloud
            cloud_folder = join(self.path, cloud_name)
            room_folders = [join(cloud_folder, room) for room in listdir(cloud_folder) if isdir(join(cloud_folder, room))]

            # Initiate containers
            cloud_points = np.empty((0, 3), dtype=np.float32)
            cloud_colors = np.empty((0, 3), dtype=np.uint8)
            cloud_classes = np.empty((0, 1), dtype=np.int32)

            # Loop over rooms
            for i, room_folder in enumerate(room_folders):

                print('Cloud %s - Room %d/%d : %s' % (cloud_name, i+1, len(room_folders), room_folder.split('/')[-1]))

                for object_name in listdir(join(room_folder, 'Annotations')):

                    if object_name[-4:] == '.txt':

                        # Text file containing point of the object
                        object_file = join(room_folder, 'Annotations', object_name)

                        # Object class and ID
                        tmp = object_name[:-4].split('_')[0]
                        if tmp in self.name_to_label:
                            object_class = self.name_to_label[tmp]
                        elif tmp in ['stairs']:
                            object_class = self.name_to_label['clutter']
                        else:
                            raise ValueError('Unknown object name: ' + str(tmp))

                        # Correct bug in S3DIS dataset
                        if object_name == 'ceiling_1.txt':
                            with open(object_file, 'r') as f:
                                lines = f.readlines()
                            for l_i, line in enumerate(lines):
                                if '103.0\x100000' in line:
                                    lines[l_i] = line.replace('103.0\x100000', '103.000000')
                            with open(object_file, 'w') as f:
                                f.writelines(lines)

                        # Read object points and colors
                        object_data = np.loadtxt(object_file, dtype=np.float32)

                        # Stack all data
                        cloud_points = np.vstack((cloud_points, object_data[:, 0:3].astype(np.float32)))
                        cloud_colors = np.vstack((cloud_colors, object_data[:, 3:6].astype(np.uint8)))
                        object_classes = np.full((object_data.shape[0], 1), object_class, dtype=np.int32)
                        cloud_classes = np.vstack((cloud_classes, object_classes))

            # Save as ply
            write_ply(cloud_file,
                      (cloud_points, cloud_colors, cloud_classes),
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

        print('Done in {:.1f}s'.format(time.time() - t0))
        return

    def load_subsampled_clouds(self):

        # Parameter
        dl = self.config.first_subsampling_dl

        # Create path for files
        tree_path = join(self.path, 'input_{:.3f}'.format(dl))
        if not exists(tree_path):
            makedirs(tree_path)
        
        def extract_label_path(file_path):
            return '_GT/SEG/man_seg'+file_path[-7:]
        
        ##############
        # Load KDTrees
        ##############

        for i, file_path in enumerate(self.files):

            # Restart timer
            t0 = time.time()

            # Get cloud name
            cloud_name = self.cloud_names[i]

            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))
            label_path = self.path + extract_label_path(file_path)
            # Read tiff file
            data = read_tiff(file_path, label_path)
            points = np.vstack((data[:, 0], data[:, 1], data[:, 2])).T
            points = points.astype(np.float32)
            # Single color in the cell data set
            # colors = np.vstack((data['red'], data['green'], data['blue'])).T
            intensity = data[:, 3].astype(np.uint8).reshape((-1, 1))
            labels = data[:, 4].astype(np.int32).reshape((-1, 1))

            
            # Subsample cloud
            sub_points, sub_colors, sub_labels = grid_subsampling(points,
                                                                  features=intensity,
                                                                  labels=labels,
                                                                  sampleDl=dl)

            # Did not Rescale float color and squeeze label
            
            #sub_colors = sub_colors / 255
            sub_labels = np.squeeze(sub_labels)

            # Get chosen neighborhoods
            search_tree = KDTree(sub_points, leaf_size=10)
            #search_tree = nnfln.KDTree(n_neighbors=1, metric='L2', leaf_size=10)
            #search_tree.fit(sub_points)

            # Save KDTree
            with open(KDTree_file, 'wb') as f:
                pickle.dump(search_tree, f)


            # Fill data containers
            self.input_trees += [search_tree]
            self.input_colors += [sub_colors]
            self.input_labels += [sub_labels]

            size = sub_colors.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size * 1e-6, time.time() - t0))

        ############################
        # Coarse potential locations
        ############################

        # Only necessary for validation and test sets
        if self.use_potentials:
            print('\nPreparing potentials')

            # Restart timer
            t0 = time.time()

            pot_dl = self.config.in_radius / 10
            cloud_ind = 0

            for i, file_path in enumerate(self.files):

                # Get cloud name
                cloud_name = self.cloud_names[i]

                # Name of the input files
                coarse_KDTree_file = join(tree_path, '{:s}_coarse_KDTree.pkl'.format(cloud_name))

                # Check if inputs have already been computed
                if exists(coarse_KDTree_file):
                    # Read pkl with search tree
                    with open(coarse_KDTree_file, 'rb') as f:
                        search_tree = pickle.load(f)

                else:
                    # Subsample cloud
                    sub_points = np.array(self.input_trees[cloud_ind].data, copy=False)
                    coarse_points = grid_subsampling(sub_points.astype(np.float32), sampleDl=pot_dl)

                    # Get chosen neighborhoods
                    search_tree = KDTree(coarse_points, leaf_size=10)

                    # Save KDTree
                    with open(coarse_KDTree_file, 'wb') as f:
                        pickle.dump(search_tree, f)

                # Fill data containers
                self.pot_trees += [search_tree]
                cloud_ind += 1

            print('Done in {:.1f}s'.format(time.time() - t0))

        ######################
        # Reprojection indices
        ######################

        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        # Only necessary for validation and test sets
        if self.set in ['validation', 'test']:

            print('\nPreparing reprojection indices for testing')

            # Get validation/test reprojection indices
            for i, file_path in enumerate(self.files):

                # Restart timer
                t0 = time.time()

                # Get info on this cloud
                cloud_name = self.cloud_names[i]

                # File name for saving
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))

                # Try to load previous indices
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    data = read_tiff(file_path, label_path)
                    points = np.vstack((data[:, 0], data[:, 1], data[:, 2])).T
                    points = points.astype(np.float32)
                    labels = data[:, 4].astype(np.int32).reshape((-1, 1))

                    # Compute projection inds
                    idxs = self.input_trees[i].query(points, return_distance=False)
                    #dists, idxs = self.input_trees[i_cloud].kneighbors(points)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
                self.validation_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        print()
        return

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        def extract_label_path(file_path):
            return '_GT/SEG/man_seg'+file_path[-7:]
        label_path = self.path + extract_label_path(file_path)
        data = read_tiff(file_path, label_path)
        points = np.vstack((data[:, 0], data[:, 1], data[:, 2])).T
        points = points.astype(np.float32)
        return points


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class CellDataCustomBatch:
    """Custom batch definition with memory pinning for S3DIS"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]
        # Number of layers
        L = (len(input_list) - 7) // 5

        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.cloud_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.center_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.input_inds = torch.from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.center_inds = self.center_inds.pin_memory()
        self.input_inds = self.input_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.cloud_inds = self.cloud_inds.to(device)
        self.center_inds = self.center_inds.to(device)
        self.input_inds = self.input_inds.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list



def CellDataCollate(batch_data):
    return CellDataCustomBatch(batch_data)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Debug functions
#       \*********************/


def debug_upsampling(dataset, loader):
    """Shows which labels are sampled according to strategy chosen"""


    for epoch in range(10):

        for batch_i, batch in enumerate(loader):

            pc1 = batch.points[1].numpy()
            pc2 = batch.points[2].numpy()
            up1 = batch.upsamples[1].numpy()

            print(pc1.shape, '=>', pc2.shape)
            print(up1.shape, np.max(up1))

            pc2 = np.vstack((pc2, np.zeros_like(pc2[:1, :])))

            # Get neighbors distance
            p0 = pc1[10, :]
            neighbs0 = up1[10, :]
            neighbs0 = pc2[neighbs0, :] - p0
            d2 = np.sum(neighbs0 ** 2, axis=1)

            print(neighbs0.shape)
            print(neighbs0[:5])
            print(d2[:5])

            print('******************')
        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_timing(dataset, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.config.batch_num
    estim_N = 0

    for epoch in range(10):

        for batch_i, batch in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.cloud_inds) - estim_b) / 100
            estim_N += (batch.features.shape[0] - estim_N) / 10

            # Pause simulating computations
            time.sleep(0.05)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > -1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f} - {:.0f}'
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1],
                                     estim_b,
                                     estim_N))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_show_clouds(dataset, loader):


    for epoch in range(10):

        clouds = []
        cloud_normals = []
        cloud_labels = []

        L = dataset.config.num_layers

        for batch_i, batch in enumerate(loader):

            # Print characteristics of input tensors
            print('\nPoints tensors')
            for i in range(L):
                print(batch.points[i].dtype, batch.points[i].shape)
            print('\nNeigbors tensors')
            for i in range(L):
                print(batch.neighbors[i].dtype, batch.neighbors[i].shape)
            print('\nPools tensors')
            for i in range(L):
                print(batch.pools[i].dtype, batch.pools[i].shape)
            print('\nStack lengths')
            for i in range(L):
                print(batch.lengths[i].dtype, batch.lengths[i].shape)
            print('\nFeatures')
            print(batch.features.dtype, batch.features.shape)
            print('\nLabels')
            print(batch.labels.dtype, batch.labels.shape)
            print('\nAugment Scales')
            print(batch.scales.dtype, batch.scales.shape)
            print('\nAugment Rotations')
            print(batch.rots.dtype, batch.rots.shape)
            print('\nModel indices')
            print(batch.model_inds.dtype, batch.model_inds.shape)

            print('\nAre input tensors pinned')
            print(batch.neighbors[0].is_pinned())
            print(batch.neighbors[-1].is_pinned())
            print(batch.points[0].is_pinned())
            print(batch.points[-1].is_pinned())
            print(batch.labels.is_pinned())
            print(batch.scales.is_pinned())
            print(batch.rots.is_pinned())
            print(batch.model_inds.is_pinned())

            show_input_batch(batch)

        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_batch_and_neighbors_calib(dataset, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)

    for epoch in range(10):

        for batch_i, input_list in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Pause simulating computations
            time.sleep(0.01)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> Average timings (ms/batch) {:8.2f} {:8.2f} '
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1]))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)
