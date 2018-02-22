Training one tests the effect of  the h, s and v channels from the HSV representation of rgb-images.

The GTSRB training set must be placed at 'Data/Final_Training/Images' relative to the location of the files or change
the training_path variable in main1.py to the correct destination.

Empy folder at 'Trained_models/' + training_name and 'Logs/' + training_name must be set up for callbacks to work.
See the training method in helper_func.py for more info

case1 - only basic preprocessing. crop and rescale to standard size
case2 - added histogram normalization in the v channel
case3 - histogram norm in h channel
case4 - hist norm in s channel
case5 - hist norm in h and s channels
case6 - hist norm in h and v channels
case7 - hist norm in s and v channels
case8 - hist norm in h, s and v channels