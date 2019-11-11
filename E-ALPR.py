import argparse
import sys
from collections import Counter

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

MODEL = None

parser = argparse.ArgumentParser(description='Licence Plate Recognition')
parser.add_argument('-i', '--image', help='Path to image file.')
parser.add_argument('-v', '--video', help='Path to video file.')
parser.add_argument('-m', '--model', help='Path to model file.', required=True)
parser.add_argument('-d', '--debug', help='Verbose output.', action='store_true')
parser.add_argument('-c', '--cam', help='Predict on camera.', action='store_true')
parser.add_argument('--conf', default=0.5, help='Confidence Threshold.')
parser.add_argument('--nms', default=0.5, help='Non-Maxima Suppression Threshold.')
args = parser.parse_args()

confThreshold = args.conf
nmsThreshold = args.nms

inpWidth = 416  # Width of YoloTiny network's input image
inpHeight = 416  # Height of YoloTiny network's input image

# Load the configuration
modelConfiguration = "YoloModel/yolov3-tiny.cfg"
modelWeights = "YoloModel/yolov3-tiny.backup"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

outputFile = "out.avi"
cap = None
vid_writer = None
font = ImageFont.truetype('fonts/tradbdo.ttf', 40)

YoloClasses = 'LP'
alphabet = {
    "a": "أ", "b": "ب", "t": "ت", "th": "ث", "g": "ج", "hh": "ح", "kh": "خ", "d": "د", "the": "ذ",
    "r": "ر", "z": "ز", "c": "س", "sh": "ش", "s": "ص", "dd": "ض", "tt": "ط", "zz": "ظ", "i": "ع",
    "gh": "غ", "f": "ف", "q": "ق", "k": "ك", "l": "ل", "m": "م", "n": "ن", "h": "ه", "w": "و",
    "y": "ي", "0": "٠", "1": "١", "2": "٢", "3": "٣", "4": "٤", "5": "٥", "6": "٦", "7": "٧",
    "8": "٨", "9": "٩"
}

classes = list(alphabet.keys())


# Get the names of the output layers
def getOutputsNames(n):
    # Get the names of all the layers in the network
    layersNames = n.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in n.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(fr, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(fr, (left, top), (right, bottom), (255, 255, 255), 3)
    # Get the label for the class name and its confidence
    lab = '%s:%.2f' % (YoloClasses[classId], conf)
    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(fr, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                  (0, 0, 255), cv2.FILLED)
    cv2.putText(fr, lab, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(fr, outs, confT, nmsT):
    frameHeight = fr.shape[0]
    frameWidth = fr.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for o in outs:
        for detection in o:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confT:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confT, nmsT)
    cropped = None
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = max(box[0], 0)
        top = max(box[1], 0)
        width = max(box[2], 0)
        height = max(box[3], 0)
        if height > width:
            continue
        cropped = fr[top:(top + height), left:(left + width)]
        drawPred(fr, classIds[i], confidences[i], left, top, left + width, top + height)

    return len(indices) > 0, cropped


def printDict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            printDict(v)
        else:
            print("{0} : {1:5.2f}%, ".format(k, v), end='')


def predict_image(img, model=None):
    if not model:
        raise ValueError("You Need to Submit a Model File or a Model Object")

    digits, marked = mark(img)
    prediction = {}
    plate = ''
    for i in range(len(digits)):
        try:
            resized = cv2.resize(square(digits[i]), (40, 40), interpolation=cv2.INTER_AREA)
            if args.debug:
                cv2.imshow(str(i), resized)
            result = model.predict(np.array(resized[tf.newaxis, ..., tf.newaxis], dtype='f'))
            prediction[i] = {}
            prediction[i][classes[int(np.argmax(result))]] = float(np.max(result) * 100)
            plate += (alphabet[classes[int(np.argmax(result))]] + ' ')
        except AssertionError:
            print("empty")

    printDict(prediction)
    print()
    return plate, marked


def square(img):
    assert type(img) == np.ndarray
    d, r = divmod(abs(img.shape[0] - img.shape[1]), 2)
    if img.shape[0] > img.shape[1]:
        return cv2.copyMakeBorder(img, 0, 0, d if not r else d + 1, d, cv2.BORDER_CONSTANT, 0)
    else:
        return cv2.copyMakeBorder(img, d if not r else d + 1, d, 0, 0, cv2.BORDER_CONSTANT, 0)


def mark(img):
    chars = {}
    digits = []
    copy = img.copy()
    # Convert to Gray
    gray = cv2.inRange(img, (0, 0, 0), (150, 70, 255))
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=3 if img.shape[0] > 250 else 1)

    if args.debug:
        cv2.imshow("gray", gray)
        cv2.imshow("opening", opening)

    # Finding characters
    cnt, he = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    k = [key for (key, value) in Counter([x[3] for x in he[0]]).items() if value >= 5]
    print(k)
    t1, t2, _ = img.shape
    for r in k:
        for i, v in enumerate(cnt):
            if he[0][i][3] == r:
                x, y, w, h = cv2.boundingRect(v)
                if .5 * t1 > h > .1 * t1 and .3 * t2 > w > .01 * t2:
                    chars[x] = opening[y:y + h, x:x + w]
                    cv2.rectangle(copy, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
        if len(chars) < 5:
            chars = {}
            copy = img.copy()
        else:
            break
    if len(chars) >= 1:
        for i, key in enumerate(sorted(chars.keys())):
            digits.append(chars[key])
            if args.debug:
                cv2.imshow(str(i), chars[key])
        cv2.imshow("final", copy)
    return digits, copy


if __name__ == '__main__':

    if args.model:
        import tensorflow as tf  # Import is here so we don't waste time if there's no model
        MODEL = tf.keras.models.load_model(args.model)
    else:
        parser.print_help()
        sys.exit(1)

    if args.cam:
        # Web-cam input
        cap = cv2.VideoCapture("http://192.168.137.51:8080/video")

    elif args.video:
        cap = cv2.VideoCapture(args.video)
        outputFile = args.video[:-4] + '_out.avi'

    elif args.image:
        cap = cv2.VideoCapture(args.image)
        outputFile = args.image[:-4] + '_out.jpg'

    else:
        parser.print_help()
        sys.exit(2)

    # Get the video writer initialized to save the output video
    if args.video or args.cam:
        vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                     (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap and cv2.waitKey(1):
        if cv2.waitKey(1) == ord('q'):
            break
        # Get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv2.waitKey(3000)  # I love you 3000 ^_^
            break

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # Sets the input to the network
        net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        run = net.forward(getOutputsNames(net))
        # Remove the bounding boxes with low confidence
        rec, plateImg = postprocess(frame, run, confThreshold, nmsThreshold)
        cv2.imshow("Capture", cv2.resize(frame, (400, 300)))
        # If there's still a plate, recognize the characters
        if rec and plateImg is not None:
            out, final = predict_image(plateImg, model=MODEL)
            frame[0:final.shape[0], 0:final.shape[1], :] = final
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            draw.text((round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 3), 100), out, font=font, fill=(0, 255, 0, 0))
            frame = np.array(image)

        # Write the frame with the detection boxes
        if args.image:
            cv2.imwrite(outputFile, frame.astype(np.uint8))
        else:
            vid_writer.write(frame.astype(np.uint8))

    if not args.image:
        vid_writer.release()
    if cap:
        cap.release()
    cv2.waitKey()
    cv2.destroyAllWindows()
