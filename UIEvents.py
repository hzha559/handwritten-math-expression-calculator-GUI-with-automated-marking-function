import os
from kivy.core.window import Window
import numpy as np
import cv2
import json
import Functions as fx
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.app import App
from model import network
from model import recognize
 
class UIEvents():
    #define all the functions of the GUI
    def __init__(self,ImageViewer,LabelReminder,textinput):
        self.ImageViewer = ImageViewer
         
        self.LabelReminder = LabelReminder  #interactive text
        self.text=textinput  #the text field
        self.model,self.transform=network()
        self.nextflag=0  #record if the "Next Page" button is pressed
        self.path='sliced one/'  #the location to store the handwriting
        self.CurrentDrawPointVector = []  #the position of each stroke
        self.BlankLabel = np.zeros((576,1021),np.uint8)  #the drawing interface        
        self.CurrentLabel = self.BlankLabel
        self.CurrentDicom = np.array(cv2.imread('cat.jpg'),np.uint8)  #the background
        self.DrawStatus = False
        self.EraserStatus = False
        self.DrawInProgress = False
        self.stringindex=0  #to deal with multiple pages
        self.index=0
        self.havedraw=0
        self.LoadContent()  

        # Accept mouse input
        Window.bind(on_touch_move=self.on_touch_move)
        Window.bind(on_touch_down=self.on_touch_down)
        Window.bind(on_touch_up=self.on_touch_up)
####################################################################################################
    def DrawClick(self,event):
        self.DrawStatus = True
        self.EraserStatus = False
        self.havedraw=0
        if self.nextflag==0:
            self.LabelReminder.text=''
        if self.nextflag==0:
            self.text.text=self.text.text
        if self.nextflag==2:
            self.nextflag=3
        self.LabelReminder.color=(1, .3, .3, 1)
        self.LabelReminder.text = 'Drawing on progress, Please click "Recognition" after drawing'
            
    def EraseClick(self,event):
        self.havedraw=0
        self.DrawStatus = False
        self.EraserStatus = True
        self.LabelReminder.color=(1, .3, .3, 1)
        self.LabelReminder.text = 'Erasing on progress'
        if self.nextflag==2:
            self.nextflag=3

    def NextPageClick(self,event):
        self.LabelReminder.color=(1, .3, .3, 1)
        if self.text.text!='' and self.havedraw==1:
            self.EditsMade = True

            self.CurrentLabel =np.zeros((576,1021),np.uint8)

            self.CurrentDisplayImage =fx.CreateDisplayImage(self.CurrentDicom,np.zeros((576,1021),np.uint8))
            self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)
            self.nextflag=1
            self.LabelReminder.text = 'This is page '+str(self.stringindex+1)
        else:
            self.LabelReminder.text = 'Please "Draw" and "Recognize" for the current page first'
    
    def Close(self,event):
        #App.get_running_app().stop()
        Window.close()

    def EraseAllClick(self,event):
        self.LabelReminder.color=(1, .3, .3, 1)
        self.EditsMade = True
        #self.CurrentLabel = self.BlankLabel.copy()
        self.CurrentLabel =np.zeros((576,1021),np.uint8)
        #self.CurrentDisplayImage = fx.CreateDisplayImage(self.CurrentDicom, self.CurrentLabel)#
        self.CurrentDisplayImage =fx.CreateDisplayImage(self.CurrentDicom,np.zeros((576,1021),np.uint8))
        self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)
        self.LabelReminder.text='All drawings, including the previous pages are deleted'
        self.nextflag=0
        self.havedraw=0
        self.text.text=''
        
    def Recognition(self,event):
        
        self.DrawStatus = False
        self.EraserStatus = False
        self.EditsMade = False
        #p = fx.SetPNGCompression(0)
        
        im=self.CurrentDisplayImage
        image=[[] for i in range(11)]
        for i in range(11):
            image[i]=im[217:312,12+90*i:98+90*i]
            if i<10:
                cv2.imwrite(self.path+str(i)+'.jpg',image[i])
            else:
                cv2.imwrite(self.path+'a'+'.jpg',image[i])
        try:
            if recognize(self.model,self.path,self.transform)!='':
                self.havedraw=1
            else:
                self.havedraw=0
            if self.nextflag==1:  # if the "Next Page" is clicked, the TextField will keep the recognized result for every page
                    self.stringindex+=1
                    #print('index',self.stringindex)
                    self.index=len(self.text.text)
                    self.text.text+=recognize(self.model,self.path,self.transform)
                    print('recognized',self.text.text) 
                    self.nextflag=2  # ensure that repeated symbols will not be loaded if the user clicks recognition for multiple times
                    
            elif self.nextflag==0:
                    self.text.text=recognize(self.model,self.path,self.transform)##################################
                    print('recognized',self.text.text)
                    # in normal mode the TextField will be updated with the lateset handwritten expression
                    
            elif self.nextflag==3:
                    self.text.text=self.text.text[0:self.index]+recognize(self.model,self.path,self.transform)
                    print('recognized',self.text.text)
            
        except:
            self.havedraw=0
            self.LabelReminder.color=(1, .3, .3, 1) # this occurs very rare that saved images of math expression are lost
            self.LabelReminder.text = 'All drawings lost unexpectedly, Please draw it again'
            
    
    def Calculate(self,event):
        resultlist=[]
        string=self.text.text  # the TextField expression
        calculate=True
        empty=True
        if string!= '':
            empty=False  # if not empty
            if '=' not in string:  # if it is not an expression
                try:
                    result=eval(string)
                    resultlist.append(result)
                    resultlist.append('') # if eval() works on the expression, it will return the result 
                    
                except:
                    calculate=False  # else do not proceed

            else:
                position=string.index('=') # find the position of the equal sign to segment the left and right hand side expressions
                try:
                    left=eval(string[0:position]) 
                    right=eval(string[position+1:]) #calculate left and right hand side expressions
                    #print(left,right)
                    if type(left)==int: # if left hand side expression generates an int result
                        if left==right:
                            resultlist.append(left)
                            resultlist.append('Correct') # require the right hand side expression to precisely equal to it to be regarded as correct
                        else:
                            resultlist.append(left)
                            resultlist.append('Wrong')
                            
                    else:  # if left hand side expression yields a float
                        if abs(np.round(left,2)-right)<abs(left*0.005):  # require the error between the left hand side and right hand side equation to be less than 0.5% to be regarded as correct
                            #print(left)
                            resultlist.append(left)
                            resultlist.append('Correct')
                            #print(list)
                        else:
                            resultlist.append(left)
                            resultlist.append('Wrong')
                            #print(list)

                except:
                    calculate=False # invalid expression

        else:
            calculate=False
            empty=True # the expression is empty
        ###################################################################
                
                
        if calculate==False and empty==False: # if the expression is invalid
            self.LabelReminder.color=(1, .3, .3, 1) # warning message in red
            self.LabelReminder.text = 'Wrong expression, please check it'
                    
        if resultlist!= []:
            if resultlist[1]=='Wrong':
                self.LabelReminder.color=(1, .3, .3, 1) # if corrected, showing 'True' and the correct result in green
            else:
                self.LabelReminder.color=(0, 1, 0, 1) # if wrong, showing 'False' and the correct result in red
            self.LabelReminder.text = 'The left-hand expression equals to '+str(resultlist[0])+'  '+str(resultlist[1])
            
        elif empty==True:
            self.LabelReminder.color=(1, .3, .3, 1) # if the expression is empty, show nothing
            self.LabelReminder.text = 'Cannot calculate because nothing is recognized'
        #self.LabelReminder.color=(1, .3, .3, 1)

    def LoadContent(self):
        self.CurrentDisplayImage = np.array(cv2.imread('cat.jpg'),np.uint8)
        self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage) # load the background which contains grids
        
    def on_touch_move(self, x, touch):
        self.mouse_x,self.mouse_y = self.ConvertToImageCoords(touch.x,touch.y)
        if self.mouse_x >= 0 and self.mouse_x < 1021 and self.mouse_y >= 0 and self.mouse_y < 576:
            # print "Mouse Up ( " + str(self.mouse_x) + " , " + str(self.mouse_y) + " )"
            if self.DrawInProgress and self.DrawStatus :
                self.EditsMade = True
                self.CurrentDrawPointVector.append([self.mouse_x,self.mouse_y])
                # Display user drawing outline
                if self.mouse_x < 1021 - 1 and self.mouse_x > 0 + 1 and self.mouse_y < 576 - 1 and self.mouse_y > 0 + 1:# 512 ->1024 dash line for entire image
                    #self.CurrentDisplayImage[self.mouse_y-1:self.mouse_y+1,self.mouse_x+5-1:self.mouse_x+5+1]=(250,100,100)#offest by 512 
                    self.CurrentDisplayImage[self.mouse_y-1:self.mouse_y+1,self.mouse_x-1:self.mouse_x+1]=(250,100,100)
                    self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)
            
    def on_touch_down(self, x, touch): # when the mouse is hold
        self.DrawInProgress = True

    def on_touch_up(self, x, touch): # when the mouse is released
        self.mouse_x,self.mouse_y = self.ConvertToImageCoords(touch.x,touch.y)
        if self.mouse_x >= 0 and self.mouse_x < 1021 and self.mouse_y >= 0 and self.mouse_y < 576:  #if the mouse is within the drawing zone
            if self.DrawInProgress and self.EraserStatus:
                self.EditsMade = True
                self.FloodFillBlob(self.mouse_x,self.mouse_y,0)  # allow erasing if erase is clicked
            if self.DrawInProgress and self.DrawStatus and len(self.CurrentDrawPointVector)>0:
                self.EditsMade = True
                if len(self.CurrentDrawPointVector) > 0:  # if more than 1 coordinates has the mouse draw, enable drawing the 8-connected line
                    self.DrawFilledPolygon()
        if self.DrawInProgress and self.DrawStatus: #???????????????????????
            del self.CurrentDrawPointVector[:]
        self.DrawInProgress = False

    def ConvertToImageCoords(self,x,y):
        c = x
        r = 800- y+20-152
        c = int(round(c))
        r = int(round(r))  #change the coordinates of the mouse to that of the image
        return c,r

    def DrawFilledPolygon(self):
        pointvector = np.array(self.CurrentDrawPointVector,np.int32)
        cv2.polylines(self.CurrentLabel, [pointvector], 0, (255,255,255),1)  # use 8-connected lines for drawing
        self.CurrentDisplayImage = fx.CreateDisplayImage(self.CurrentDicom, self.CurrentLabel)
        self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)

    def FloodFillBlob(self, x, y, i):
        #used to erase a single stroke
        if self.CurrentLabel[y-20:y+20,x-20:x+20].any()!= i:  # if any of the adjacent pixels are not white, erase them
            self.CurrentLabel[y-20:y+20,x-20:x+20]= i
            self.CurrentDisplayImage = fx.CreateDisplayImage(self.CurrentDicom, self.CurrentLabel)
            self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)








