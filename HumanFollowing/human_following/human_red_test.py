import common as cm
import cv2
import numpy as np
from PIL import Image
import time
from threading import Thread
import serial


#hang so pid
Kp = 30
Ki = 0
Kd = 0

#prev_error
prev_error = 0
integral = 0

ser = serial.Serial('COM7' , 115200, timeout=1.0)
ser.reset_input_buffer()

def send_pwm(pwm):
    command = f"{pwm}\n"
    ser.write(command.encode('utf-8'))

def back():
    print("moving back!!!!!!")
    ser.write("b\n".encode('utf-8'))
    
def right():
    print("moving right!!!!!!")
    ser.write("r\n".encode('utf-8'))
    
def left():
    print("moving left!!!!!!")
    ser.write("l\n".encode('utf-8'))
    
def forward():
    print("moving forward!!!!!!")
    ser.write("f\n".encode('utf-8'))
    
def stop():
    print("stop!!!!!!")
    ser.write("s\n".encode('utf-8'))

def pid_control(x_deviation):
    global prev_error, integral

    error = x_deviation
    derivative = error - prev_error
    integral += error
    prev_error = error

    # PID output
    delta = Kp * error + Ki * integral + Kd * derivative
    delta = int(delta)

    # Gửi PWM qua Serial
    send_pwm(delta)    

import sys
sys.path.insert(0, '../../human')


threshold=0.2
top_k=3 #first five objects with prediction probability above threshhold (0.2) to be considered
#edgetpu=0

model_dir = 'E:/human/all_models'
model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
model_edgetpu = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
lbl = 'coco_labels.txt'

tolerance=0.1
x_deviation=0
y_deviation = 0
y_max=0
arr_track_data=[0,0,0,0,0,0]
y_center = 0;

object_to_track='person'

#---------Flask----------------------------------------
from flask import Flask, Response
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    #return "Default Message"
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    #global cap
    return Response(main(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    
#-----initialise motor speed-----------------------------------

#------------------------------------------
# Hàm kiểm tra vùng màu đỏ trong bounding box
def is_wearing_red(frame, bbox):
    """
    Kiểm tra xem bounding box có chứa màu đỏ chiếm ưu thế không.
    bbox: tọa độ [x_min, y_min, x_max, y_max] đã chuẩn hóa (0~1).
    """
    height, width, _ = frame.shape
    x_min, y_min, x_max, y_max = [int(val * width) if i % 2 == 0 else int(val * height)
                                  for i, val in enumerate(bbox)]
    
    # Cắt vùng ảnh của đối tượng
    person_img = frame[y_min:y_max, x_min:x_max]

    if person_img.size == 0:
        return False

    # Chuyển sang không gian HSV để dễ tách màu
    hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)

    # Định nghĩa khoảng màu đỏ (có 2 dải màu trong HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Tạo mask màu đỏ
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Tính tỷ lệ pixel màu đỏ trong bbox
    red_ratio = cv2.countNonZero(mask) / (person_img.shape[0] * person_img.shape[1])
    
    return red_ratio > 0.15  # >15% là màu đỏ => có thể điều chỉnh


def track_object(objs, labels):
    global x_deviation, y_deviation, y_max, tolerance, arr_track_data, y_center, frame

    if len(objs) == 0:
        print("no objects to track")
        stop()
        arr_track_data = [0, 0, 0, 0, 0, 0]
        return

    target_obj = None
    for obj in objs:
        label = labels.get(obj.id, obj.id)
        if label == "person":
            if is_wearing_red(frame, obj.bbox):  # Dùng frame hiện tại
                target_obj = obj
                break

    if target_obj is None:
        print("No person with red clothing found")
        stop()
        arr_track_data = [0, 0, 0, 0, 0, 0]
        return

    x_min, y_min, x_max, y_max = list(target_obj.bbox)
    x_diff = x_max - x_min
    y_diff = y_max - y_min

    obj_x_center = round(x_min + x_diff / 2, 3)
    obj_y_center = round(y_min + y_diff / 2, 3)

    print("Center: [", obj_x_center, obj_y_center, "]")
    y_center = obj_y_center
    x_deviation = round(0.5 - obj_x_center, 3)
    y_deviation = round(obj_y_center - 0.5, 3)
    y_max = round(y_max, 3)

    print("{", x_deviation, y_max, "}")

    thread = Thread(target=move_robot)
    thread.start()

    arr_track_data[0] = obj_x_center
    arr_track_data[1] = obj_y_center
    arr_track_data[2] = x_deviation
    arr_track_data[3] = y_max
    arr_track_data[4] = obj_y_center

def move_robot():
    global x_deviation, y_max, tolerance, arr_track_data, y_center
    
    print("moving robot .............!!!!!!!!!!!!!!")
    print(x_deviation, tolerance, arr_track_data)
    
    y=1-y_max #distance from bottom of the frame
    
    if(abs(x_deviation)<tolerance):
        delay1=0
        if(y_center < 0.55):
            cmd="Stop"
            stop()
        else:
            cmd="forward"
            forward()
    
    else:
        if(x_deviation>=tolerance):
            cmd="Move Left" 
            delay1=get_delay(x_deviation) * 1.35
            left()
            pid_control(x_deviation)
            time.sleep(delay1)
            stop()
                
        if(x_deviation<=-1*tolerance):
            cmd="Move Right"
            delay1=get_delay(x_deviation) * 1.35
            x_deviation = -1 * x_deviation
            pid_control(x_deviation)
            right()
            time.sleep(delay1)
            stop()    

    arr_track_data[4]=cmd
    arr_track_data[5]=delay1

def get_delay(deviation):
    deviation=abs(deviation)
    if(deviation>=0.4):
        d=0.080
    elif(deviation>=0.35 and deviation<0.40):
        d=0.060
    elif(deviation>=0.20 and deviation<0.35):
        d=0.050
    else:
        d=0.040
    return d
    
def main():
    global frame
    cap = cv2.VideoCapture(0)
    mdl = model
        
    interpreter, labels =cm.load_model(model_dir,mdl,lbl,0)
    
    fps=1
    arr_dur=[0,0,0]
    
    while True:
        start_time=time.time()
        
        #----------------Capture Camera Frame-----------------
        start_t0=time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
       
        arr_dur[0]=time.time() - start_t0
        #----------------------------------------------------
       
        #-------------------Inference---------------------------------
        start_t1=time.time()
        cm.set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = cm.get_output(interpreter, score_threshold=threshold, top_k=top_k)
        
        arr_dur[1]=time.time() - start_t1
        #----------------------------------------------------
       
       #-----------------other------------------------------------
        start_t2=time.time()
        track_object(objs,labels)#tracking  <<<<<<<
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
        cv2_im = append_text_img1(cv2_im, objs, labels, arr_dur, arr_track_data)
       # cv2.imshow('Object Tracking - TensorFlow Lite', cv2_im)
        
        ret, jpeg = cv2.imencode('.jpg', cv2_im)
        pic = jpeg.tobytes()
        
        #Flask streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + pic + b'\r\n\r\n')
       
        arr_dur[2]=time.time() - start_t2
        fps = round(1.0 / (time.time() - start_time),1)
        print("*********FPS: ",fps,"************")

    cap.release()
    cv2.destroyAllWindows()

def append_text_img1(cv2_im, objs, labels, arr_dur, arr_track_data):
    height, width, channels = cv2_im.shape
    font=cv2.FONT_HERSHEY_SIMPLEX

    global tolerance

    #draw black rectangle on top
    cv2_im = cv2.rectangle(cv2_im, (0,0), (width, 24), (0,0,0), -1)

    #write processing durations
    cam=round(arr_dur[0]*1000,0)
    inference=round(arr_dur[1]*1000,0)
    other=round(arr_dur[2]*1000,0)
    text_dur = 'Camera: {}ms   Inference: {}ms   other: {}ms'.format(cam,inference,other)
    cv2_im = cv2.putText(cv2_im, text_dur, (int(width/4)-30, 16),font, 0.4, (255, 255, 255), 1)

    #write FPS
    total_duration=cam+inference+other
    fps=round(1000/total_duration,1)
    text1 = 'FPS: {}'.format(fps)
    cv2_im = cv2.putText(cv2_im, text1, (10, 20),font, 0.7, (150, 150, 255), 2)


    #draw black rectangle at bottom
    cv2_im = cv2.rectangle(cv2_im, (0,height-24), (width, height), (0,0,0), -1)

    #write deviations and tolerance
    str_tol='Tol : {}'.format(tolerance)
    cv2_im = cv2.putText(cv2_im, str_tol, (10, height-8),font, 0.55, (150, 150, 255), 2)

    x_dev=arr_track_data[0]
    str_x='X: {}'.format(x_dev)
    if(abs(x_dev)<tolerance):
        color_x=(0,255,0)
    else:
        color_x=(0,0,255)
    cv2_im = cv2.putText(cv2_im, str_x, (110, height-8),font, 0.55, color_x, 2)

    y_dev=arr_track_data[1]
    str_y='Y: {}'.format(y_dev)
    if(abs(y_dev)>0.9):
        color_y=(0,255,0)
    else:
        color_y=(0,0,255)
    cv2_im = cv2.putText(cv2_im, str_y, (220, height-8),font, 0.55, color_y, 2)

    #write command, tracking status and speed
    cmd=arr_track_data[4]
    cv2_im = cv2.putText(cv2_im, str(cmd), (int(width/2) + 10, height-8),font, 0.68, (0, 255, 255), 2)

    delay1=arr_track_data[5]
    str_sp='Speed: {}%'.format(round(delay1/(0.1)*100,1))
    cv2_im = cv2.putText(cv2_im, str_sp, (int(width/2) + 185, height-8),font, 0.55, (150, 150, 255), 2)

    if(cmd==0):
        str1="No object"
    elif(cmd=='Stop'):
        str1='Acquired'
    else:
        str1='Tracking'
    cv2_im = cv2.putText(cv2_im, str1, (width-140, 18),font, 0.7, (0, 255, 255), 2)

    #draw center cross lines
    cv2_im = cv2.rectangle(cv2_im, (0,int(height/2)-1), (width, int(height/2)+1), (255,0,0), -1)
    cv2_im = cv2.rectangle(cv2_im, (int(width/2)-1,0), (int(width/2)+1,height), (255,0,0), -1)

    #draw the center red dot on the object (this is the tracked center - normalized)
    cv2_im = cv2.circle(cv2_im, (int(arr_track_data[0]*width),int(arr_track_data[1]*height)), 7, (0,0,255), -1)

    #draw the tolerance box
    cv2_im = cv2.rectangle(cv2_im, (int(width/2-tolerance*width),0), (int(width/2+tolerance*width),height), (0,255,0), 2)

    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)

        box_color, text_color, thickness=(0,150,255), (0,255,0),1


        text3 = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        if(labels.get(obj.id, obj.id)=="person"):
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), box_color, thickness)
            cv2_im = cv2.putText(cv2_im, text3, (x0, y1-5),font, 0.5, text_color, thickness)

            # Tính toán và vẽ tâm của từng vật thể phát hiện được (chuẩn hóa)
            center_x_normalized = (x0 + x1) / (2 * width)
            center_y_normalized = (y0 + y1) / (2 * height)
            center_x_pixel = int(center_x_normalized * width)
            center_y_pixel = int(center_y_normalized * height)
            cv2_im = cv2.circle(cv2_im, (center_x_pixel, center_y_pixel), 5, (0, 255, 255), -1) # Vẽ tâm màu vàng

            # Hiển thị tọa độ tâm chuẩn hóa
            text_center_normalized = f"({center_x_normalized:.3f}, {center_y_normalized:.3f})"
            cv2_im = cv2.putText(cv2_im, text_center_normalized, (x1 + 10, y1 - 30), font, 0.5, (255, 0, 255), 1) # Màu hồng

    return cv2_im

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True) # Run FLASK
    main()

