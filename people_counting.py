from simplecentroidtracker import SimpleCentroidTracker
import cv2
import csv
import os
import imutils
import argparse
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str, default="models/res10_300x300_ssd_iter_140000.caffemodel", help="path to pre-trained Caffe model")
ap.add_argument("-p", "--prototxt", type=str, default="models/deploy.prototxt", help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-i", "--input", type=str, default="videos", help="path to input video folder")
ap.add_argument("-o", "--output", type=str, default="output", help="path to optional output video folder")
ap.add_argument("-r", "--results", type=str, default="results.csv", help="path to results file")
ap.add_argument("-c", "--confidence", type=float, default=0.7, help="minimum probability for filtering weak detections")

args = vars(ap.parse_args())

# Set path variables to input and output video folders and Caffe model files
input_path = os.path.abspath(os.path.join(os.path.curdir, args["input"]))
output_path = os.path.abspath(os.path.join(os.path.curdir, args["output"]))
model_path = os.path.abspath(os.path.join(os.path.curdir, args["model"]))
prototxt_path = os.path.abspath(os.path.join(os.path.curdir, args["prototxt"]))

# Create folder for output videos if not exists
if not os.path.isdir(output_path):
    os.mkdir(output_path)

# Create csv file for results
with open(args["results"], "w", newline="") as csv_file:
    writer = csv.writer(csv_file, delimiter=";")
    writer.writerow(["video", "count"])

for video in os.listdir(input_path):

    # Load serialized pre-trained model
    print("[INFO] Loading model ...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    print(f"[INFO] Reading {video} ...")
    # Get input video file path
    input_video = cv2.VideoCapture(os.path.abspath(os.path.join(input_path, video)))

    # Get video properties of input video (those of the output video have to be the same)
    fps = input_video.get(cv2.CAP_PROP_FPS)
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] fps: {fps} fps, resoultion {width}x{height}")

    # Create output video file
    output_file = os.path.abspath(os.path.join(output_path, f"{video[:-4]}.avi"))
    print(f"[INFO] Writing to {output_file}")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frame_count = 0

    # Create a tracking object for current input video
    ct = SimpleCentroidTracker(max_disappeared=500, max_distance=300)

    while True:
        # Grab a single frame of video
        ret, frame = input_video.read()
        frame_count += 1

        # Quit when the input video file ends
        if not ret:
            break

        # Construct an input blob for the frame by resizing to a fixed 300x300 pixels and then normalizing it
        blob = cv2.dnn.blobFromImage(frame, 1.0, (width, height), (104.0, 177.0, 123.0))

        # Obtain detections
        net.setInput(blob)
        detections = net.forward()

        rects = []

        # Iterate through all detections
        for i in range(0, detections.shape[2]):

            # Check whether the detection has yielded the minimum confidence and draw a box around the face
            if detections[0, 0, i, 2] > args["confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                rects.append(box.astype("int"))

                (start_x, start_y, end_x, end_y) = box.astype("int")
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

        # Update centroid tracker with detected objects
        objects = ct.update(rects)

        # Draw a number and centroid for each tracked object and give info about total count
        for (object_id, centroid) in objects.items():
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(object_id), (centroid[0], centroid[1] - 10), font, 0.5, (255, 255, 255), 2),
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"Total: {ct.get_count()}", (50, height - 50), font, 2, (0, 0, 255), 2),

        # Write the resulting image to the output video file
        print(f"[INFO] Writing frame {frame_count} / {length}")
        output_video.write(frame)

        # Show the output frame
        show_frame = imutils.resize(frame, width=900)

        cv2.imshow(f"Count people in {video}", show_frame)
        key = cv2.waitKey(1) & 0xFF
        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    print(f"[INFO] Total count: {ct.get_count()}")

    # Append total count for current video to results file
    with open(args["results"], "a", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow([video[:-4], ct.get_count()])

    input_video.release()
    cv2.destroyAllWindows()