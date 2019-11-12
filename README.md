# E-ALPR
 E-ALPR is an Automatic Licence Plate Recognition System for Egyptian Plates.
 
 It uses the tiny version of the famous [Yolo](https://pjreddie.com/darknet/yolo/) Model to predict 
 the place of the plate then does some image processing to extract the characters from the plate 
 then passes it to a generated [Tensorflow](https://www.tensorflow.org) model for recognizing the 
 character using classification.
## How to use:
 First, you need to generate some characters data to train the recognition model on by running 
 the `Generate.py` script with `--fonts` parameter taking the fonts directory that include the 
 fonts you want to generate image for, and `--out` parameter referring to the output directory.
 
 The default size of the generated images is 40x40 and 50 images per font, you can change it by 
 passing the optional `--size [SIZE]` and `--count [COUNT]` parameters.
 
 NOTE: the `E-ALPR.py` and `E-ALPR_GUI.py` scripts uses 40x40 image size only for now, you have 
 to make adjustments to the code to make it compatible with the size you want.
 
 Here's an example:
 ```bash
 python Generate.py --fonts Fonts --out data
 ```
 After the data is generated, you need to train the recognition model. All you need to do is pass the
 `--path` parameter to `Train.py` script, like this:
 ```bash
 python Train.py --path data 
 ``` 
 this will train a normal [Tensorflow](https://www.tensorflow.org)  model for usage with the non-gui 
 script, if you need to use the gui script you need to pass the `--lite` parameter to create a lite 
 model.
 ```bash
 python Train.py --path data --lite
 ```
 Now, you can use either the `E-ALPR.py` script or `E-ALPR_GUI.py` script to predict and recognize 
 the plate.
 
 To use the non-gui version, you just need to pass the image to `-i` parameter and the created model
 to `-m` parameter.
 ```bash
 python E-ALPR.py -i Test/1.jpg -m model.h5
 ```
 If you want to predict on a video, you can use `-v` parameter or `-c` for a camera.
 
 If you want to see what happens in the process, just add `-d` debugging parameter.
 ## Special Thanks to:
 Adrian Rosebrock and his amazing blog "PyImageSearch" that helped me a lot in making this project.
 
 You can find the Blog here: [PyImageSearch](https://www.pyimagesearch.com)
 ## Licence:
 [MIT](https://choosealicense.com/licenses/mit/)