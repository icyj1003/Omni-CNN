import numpy as np
import os


def show_all_files_in_directory(input_path, extension):
    "This function reads the path of all files in directory input_path"
    files_list = []
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(extension):
                files_list.append(os.path.join(path, file))
    return files_list


def detecting_related_file_paths(path, categories, episodes):
    find_all_paths = [
        "/".join(a.split("/")[:-1]) for a in show_all_files_in_directory(path, "rf.npz")
    ]  # rf for example
    selected = []
    for Cat in categories:  # specify categories as input
        for ep in episodes:
            selected = selected + [
                s
                for s in find_all_paths
                if Cat in s.split("/") and "episode_" + str(ep) in s.split("/")
            ]
    print("Getting {} data out of {}".format(len(selected), len(find_all_paths)))

    return selected


# path = 'FLASH_Dataset_3_Processed'
# all_npz_files = show_all_files_in_directory(path,'.npz')
# all_npy_files = show_all_files_in_directory(path,'.npy')
# save_path = "flash_GPS_Image_LiDAR/task_common"

# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# ##all files have the same length
# all_lidar_npz_files = [a for a in all_npz_files if 'lidar.npz' in a.split('/')]
# all_img_npz_files = [a for a in all_npz_files if 'img.npz' in a.split('/')]
# all_gps_npz_files = [a for a in all_npz_files if 'gps.npz' in a.split('/')]
# all_rf_npz_files = [a for a in all_npz_files if 'rf.npz' in a.split('/')]
# all_ranperm_npy_files = [a for a in all_npy_files if 'ranperm.npy' in a.split('/')]

# for (lidar_file, img_file, gps_file, rf_file, randperm_file) in zip(all_lidar_npz_files, all_img_npz_files, all_gps_npz_files, all_rf_npz_files, all_ranperm_npy_files):
#     # print('Processing npz file',npz_file)
#     # print('Processing rf file',rf_file)
#     # print('Processing randperm file',randperm_file)
#     lidar_npz = np.load(lidar_file)['lidar']
#     img_npz = np.load(img_file)['img']
#     gps_npz = np.load(gps_file)['gps']
#     rf_npz = np.load(rf_file)['rf']
#     randperm = np.load(randperm_file)
#     # print('lidar_npz',lidar_npz.shape)
#     # print('rf_npz',rf_npz.shape)
#     # print('randperm',randperm.shape)
#     try:
#         train_X = np.concatenate((train_X, lidar_npz[randperm[:int(0.8*len(lidar_npz))]]),axis = 0)
#         train_y = np.concatenate((train_y, rf_npz[randperm[:int(0.8*len(lidar_npz))]]),axis = 0)
#         val_X = np.concatenate((val_X, lidar_npz[randperm[int(0.8*len(lidar_npz)):int(0.9*len(lidar_npz))]]),axis = 0)
#         val_y = np.concatenate((val_y, rf_npz[randperm[int(0.8*len(lidar_npz)):int(0.9*len(lidar_npz))]]),axis = 0)
#         test_X = np.concatenate((test_X, lidar_npz[randperm[int(0.9*len(lidar_npz)):]]),axis = 0)
#         test_y = np.concatenate((test_y, rf_npz[randperm[int(0.9*len(lidar_npz)):]]),axis = 0)
#     except NameError:
#         train_X = lidar_npz[randperm[:int(0.8*len(lidar_npz))]]
#         train_y = rf_npz[randperm[:int(0.8*len(lidar_npz))]]
#         val_X = lidar_npz[randperm[int(0.8*len(lidar_npz)):int(0.9*len(lidar_npz))]]
#         val_y = rf_npz[randperm[int(0.8*len(lidar_npz)):int(0.9*len(lidar_npz))]]
#         test_X = lidar_npz[randperm[int(0.9*len(lidar_npz)):]]
#         test_y = rf_npz[randperm[int(0.9*len(lidar_npz)):]]

#     print('train_X',train_X.shape)
#     print('train_y',train_y.shape)
#     print('val_X',val_X.shape)
#     print('val_y',val_y.shape)
#     print('test_X',test_X.shape)
#     print('test_y',test_y.shape)

# train_dictionary = os.path.join(save_path,'train')
# if not os.path.exists(train_dictionary):
#     os.makedirs(train_dictionary)

# val_y_dictionary = os.path.join(save_path,'val')
# if not os.path.exists(val_y_dictionary):
#     os.makedirs(val_y_dictionary)

# test_dictionary = os.path.join(save_path,'test')
# if not os.path.exists(test_dictionary):
#     os.makedirs(test_dictionary)

# np.save(train_dictionary+'/X_lidar.npy',train_X)
# np.save(train_dictionary+'/y.npy',train_y)
# np.save(val_y_dictionary+'/X_lidar.npy',val_X)
# np.save(val_y_dictionary+'/y.npy',val_y)
# np.save(test_dictionary+'/X_lidar.npy',test_X)
# np.save(test_dictionary+'/y.npy',test_y)

#     # randperm = np.random.permutation(len(gps_npz))
#     # # 'data/a.npy'
#     # save_directory = '/'.join(npz_file.split('/')[:-1])
#     # print('save directory',save_directory)
#     # np.save(save_directory+'/ranperm.npy',randperm)
