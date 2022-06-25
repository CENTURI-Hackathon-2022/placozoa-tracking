def find_outer_edge(path_in, path_out): 
    
    from tifffile import imread, imwrite
    import scipy.ndimage as nd
    from pathlib import Path
    from skimage.morphology import disk
    import numpy as np
    import morphsnakes as ms
    from scipy.ndimage.morphology import binary_fill_holes
    from skimage.filters import rank
    
    path_in = Path(path_in)
    path_out = Path(path_out)

    image = imread(path_in)
    
    output_array = np.zeros(image.shape, dtype = bool)
    
    for t in range(0,image.shape[0]):

        im_single_t = image[t,:,:]
        im_filtered_minimum =  rank.minimum(im_single_t, disk(4))
        im_ms = ms.morphological_chan_vese(im_filtered_minimum, 10)
        ms_filled = binary_fill_holes(im_ms)

        #detect if its is segmented the right way around (expecting that the background has most area touching the image border)
        #otherwise invert the image
        amount_edge_false = ms_filled[ms_filled[0,:] == False].shape[0] + ms_filled[ms_filled[-1,:] == False].shape[0] + ms_filled[ms_filled[:,0] == False].shape[0] + ms_filled[ms_filled[:,-1] == False].shape[0]
        amount_edge_true = ms_filled[ms_filled[0,:] == True].shape[0] + ms_filled[ms_filled[-1,:] == True].shape[0] + ms_filled[ms_filled[:,0] == True].shape[0] + ms_filled[ms_filled[:,-1] == True].shape[0]
        if amount_edge_true < amount_edge_false:
            pass
        else:
            ms_filled = np.invert(ms_filled) 
        
        #label connected components in the binary mask
        labels, num_features = nd.label(ms_filled)
        label_unique = np.unique(labels)

        #count pixels of each component and sort them by size, excluding the background
        vol_list = []
        for label in label_unique:
            if label != 0:
                vol_list.append(np.count_nonzero(labels == label))
        
        #create binary array of only the largest component
        binary_mask = np.zeros(labels.shape)
        binary_mask = np.where(labels == vol_list.index(max(vol_list))+1, 1, 0)
        
        output_array[t,:,:] = binary_mask
    imwrite(path_out, output_array)
        
    return output_array