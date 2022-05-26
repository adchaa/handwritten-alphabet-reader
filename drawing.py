import sys
import cv2
import pygame
from pygame.locals import *
import numpy as np
from tensorflow.keras import models
import tensorflow.keras as ker
def show_pred(p):
    for i in range(len(p[0])):
        print(chr(i+65)," =",p[0,i])
        

model = models.load_model("letter_regination.h5")
pygame.init()
screen =pygame.display.set_mode((700,700))
pygame.display.set_caption('letter Recognition')
font = pygame.font.Font('freesansbold.ttf', 32)
reset_text = pygame.font.Font('freesansbold.ttf', 16).render("press R to reset",True,(255,255,255))
screen.blit(reset_text,(1,1))
arr_x=[]
arr_y=[]
draw = False
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEBUTTONDOWN:
            draw=True
        if event.type == MOUSEMOTION and draw:
            x,y = event.pos
            pygame.draw.circle(screen,(255,255,255),(x,y),5)
            arr_x.append(x)
            arr_y.append(y)
            
            
        if event.type == MOUSEBUTTONUP:
            draw=False
            left = min(arr_x)-5
            top = min (arr_y)-5
            wid = max(arr_x)- min(arr_x) +10
            he = max(arr_y)- min(arr_y) +10
            arr_img = np.array(pygame.PixelArray(screen))[left:left+wid,top:top+he].T.astype(np.float32)
            image =cv2.resize(arr_img, (28,28))
            image = np.pad(image,(10,10), 'constant', constant_values=0)
            image = ker.utils.normalize(cv2.resize(image, (28,28)), axis=1)
            pygame.draw.rect(screen,(255,0,0),pygame.Rect(left,top,wid,he),2)
            arr_x= []
            arr_y = []
            p = model.predict(image.reshape(1,28,28,1))
            show_pred(p)
            print(np.argmax(p))
            predicted_number=np.argmax(p)
            text = font.render(chr(predicted_number+65), True, (0,255,0),(255,0,0) )
            textRect = text.get_rect()
            textRect.center = (left+(wid//2), top-15)
            screen.blit(text, textRect)
        if event.type == KEYDOWN:
            if event.unicode == "r":
                screen.fill((0,0,0))
                screen.blit(reset_text,(1,1))
            
        pygame.display.update()
            


