import numpy as np
import cv2
import time


def draw_flow(img, flow, step=16):
    # h, w = img.shape[:2]
    # y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    # fx, fy = flow[y,x].T

    # lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    # lines = np.int32(lines + 0.5)

    # img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    # for (x1, y1), (_x2, _y2) in lines:
    #     cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)

    hsv = np.zeros((h, w, 3), np.uint8)

    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


cap = cv2.VideoCapture('./granny.mp4')

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


def draw_mask(flow, img, feather_amount=10):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)

    hsv = np.zeros((h, w, 3), np.uint8)

    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    
    # с lower надо поиграть
    lower = np.array([10,10,10])
    upper = np.array([255,255,255])
    mask = cv2.inRange(hsv, lower, upper)

    feathered_mask = cv2.GaussianBlur(mask, (feather_amount*2+1, feather_amount*2+1), 0)
    feathered_mask = feathered_mask[:, :, np.newaxis]
    # background = np.zeros_like(img, dtype=img.dtype)
    # feathered_image = img * feathered_mask + background * (1 - feathered_mask)
    
    cnts = cv2.findContours(feathered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = img[y:y+h, x:x+w]
        img[y:y+h, x:x+w] = cv2.fastNlMeansDenoisingColored(ROI,None, 10, 10, 7, 15)   #filter
        

    return img


while True:
    suc, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()


    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    prevgray = gray


    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)

    cv2.imshow('flow', draw_flow(img, flow))
    # cv2.imshow('flow HSV', draw_hsv(flow))
    cv2.imshow('flow mask', draw_mask(flow, img))


    key = cv2.waitKey(5)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()