import cv2
import time
import configargparse
import numpy as np
from face_lib import face_lib
FL = face_lib()
from urllib.request import urlopen

#get image from an URL function
def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    # return the image
    return image

#parser func
def parser_args():
    parser = configargparse.ArgParser(description="Face Recognition NFV FIL HUST",
                            formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--source_rtmp",
                        help="source stream path")
    return parser.parse_args()

#calculate average fps after exit 
def average_fps(arr):
    sum = 0
    count = 1
    for fps in arr:
        sum += fps
        count += 1
    return sum/count


if __name__ == "__main__":
    #parser init
    args = parser_args()
    source_rtmp = args.source_rtmp


    #Source Streaming path
    #path = f"rtmp://localhost/live/stream"
    path = 0
    
    #setup
    font = cv2.FONT_HERSHEY_DUPLEX
    #source video
    cap = cv2.VideoCapture(path)
    WIDTH_INPUT_STREAMING = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT_INPUT_STREAMING = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS_INPUT_STREAMING = int(cap.get(cv2.CAP_PROP_FPS))

    #get image from an URL
    img = url_to_image('https://lh6.googleusercontent.com/X-GkxplHlP8_XvSxtWIIhwEHVuq_OTYNp7omk029HbpTKyQMSPUbgCdGgixr_AnfY_E=w2400')
    REFERENCE_IMAGE = img 
    SUM_FRAME_HANDLE = 1
    SUM_FRAME_HAVE_TRUE_OUTPUT = 0 
    SUM_FRAME_HAVE_FACE = 1 

    # Variables to calculate fps
    print_fps_period = 1
    frame_count = 0
    arr = []
    t0 = time.monotonic()
    
    #loop frame by frame
    while(True):
        ret, frame = cap.read()
        if ret == True:
            notify = 'False'
            
            #check if there are faces in frame and count it
            face_in_frame = FL.get_faces(frame)
            if face_in_frame:
                SUM_FRAME_HAVE_FACE += 1
            
            face_exist, no_faces_detected = FL.recognition_pipeline(frame,REFERENCE_IMAGE)
            if face_exist:
                notify = 'True'
                SUM_FRAME_HAVE_TRUE_OUTPUT += 1
            print("Recognition: {}".format(notify), end="\r")
            #cv2.putText(frame,notify,(25,25), font, 1.0, (255,255,255), 1)
            
            #fps calculate
            SUM_FRAME_HANDLE += 1
            frame_count += 1
            td = time.monotonic() - t0
            cv2.imshow("aa",frame)
            if td > print_fps_period:
                current_fps = frame_count / td
                arr += [current_fps]
                print("FPS: {:6.2f}".format(current_fps), end="\r")
                frame_count = 0
                t0 = time.monotonic()
            
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        #break th loop
        else:
            break

recognition_rate = (SUM_FRAME_HAVE_TRUE_OUTPUT/SUM_FRAME_HAVE_FACE)*100
print("AVERAGE FPS: ",average_fps(arr))
print("recognition rate: {:6.2f}".format(recognition_rate),"%")
print("Sum frame have face:", SUM_FRAME_HAVE_FACE)
print("Sum frame handled: ", SUM_FRAME_HANDLE)

# After the loop release the cap object
cap.release()
