#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False,maxHands = 2, detectionCon=0.5, trackCon =0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands #identify hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon,  self.trackCon) #identify hands
        self.mpDraw = mp.solutions.drawing_utils #drawing identified hands
        
        
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)



        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for idk, lm in enumerate (myHand.landmark):
                 #print(idk,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(idk,cx,cy)
                self.lmList.append([idk,cx,cy])
                #if idk == 0:
                     #cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        
        
        return self.lmList
    def fingersUp(self):
        fingers = []
        
        
        #Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1] [1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
         
        #4 fingers
        for idk in range(1,5):
            if self.lmList[self.tipIds[idk]][2] < self.lmList[self.tipIds[idk] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
                
                
                
                
                
                
                

    


def main():
    pTime = 0  #previous time
    cTime = 0 #current time
    
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        sucess, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
    
    
        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            
            
        cv2.imshow("image",img)
        cv2.waitKey(1)   
            
            

        

    
    
    
    
if __name__ =="__main__":
    main()


# In[ ]:




