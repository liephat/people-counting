# Counting people in a video

## Requirements

Following python version and libraries were used to test the program:

- Python 3.6
- imutils
- numpy
- scipy
- OpenCV

Install libraries by running following commands in your python 3.6 environment:
```
pip install imutils numpy scipy
pip install opencv-python
```
## Run instructions

Make sure that videos are stored in a folder '/video' in the directory of the program.

Run the program by typing 
```
python people_counting.py
```
in your console.

Optional arguments can be used, e.g. to define a folder to save output videos.

| argument | description |
| -------- | ----------- |
| -m, --model | path to pre-trained Caffe model |
| -p, --prototxt | path to Caffe 'deploy' prototxt file |
| -i, --input | path to input video folder |
| -o, --output | path to optional output video folder |
| -r, --results | path to results file |
| -c, --confidence | minimum probability for filtering weak detections |

Note: Every argument has a default value, so you only have to use them when you want to modify a variable.