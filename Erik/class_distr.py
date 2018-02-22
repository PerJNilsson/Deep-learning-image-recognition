import glob, os
import matplotlib.pyplot as plt

PATH = 'Data/Final_Training/Images'
all_img_paths = []

for x in range(43):
    all_img_paths.append(glob.glob(os.path.join(PATH , '*0' + str(x) + '/*.ppm')))

class_frequency = []
for x in range(len(all_img_paths)):
    class_frequency.append(len(all_img_paths[x]))

plt.hist(class_frequency)
plt.xlabel('Class')
plt.ylabel('No of imgs')
plt.show()