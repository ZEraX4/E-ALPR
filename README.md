# E-ALPR
 E-ALPR is an Automatic Licence Plate Recognition System for Egyptian Plates.
## How to use:
 First, you need to generate some characters data to train the recognition model on by running 
 the `Generate.py` script with `--fonts` parameter taking the fonts directory that include the 
 fonts you want to generate image for, and `--out` parameter referring to the output directory.
 
 The default size of the generated images is 40x40 and 50 images per font, you can change it by 
 passing the optional `--size [SIZE]` and `--count [COUNT]` parameters.
 
 Here's an example:
 ```bash
 python Generate.py --fonts Fonts --out data
 ```
 