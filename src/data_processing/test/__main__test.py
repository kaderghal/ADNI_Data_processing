import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")





img = nib.load('mri.nii')
img_data = img.get_fdata()
print(img_data.shape)

img_data = np.nan_to_num(img_data)

slice_0 = img_data[43, :, :]
slice_1 = img_data[:, 71, :]
slice_2 = img_data[:, :, 44]


show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for MRI image")  
plt.show()



