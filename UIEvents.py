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
import struct
from kivy.uix.textinput import TextInput
#import recognize
from kivy.app import App
#start=0
from model import f1
from model import f2



class UIEvents():
    def __init__(self,ImageViewer,FilePath,TextReport,StatusMessage,LabelReminder,textinput):
        self.ImageViewer = ImageViewer
        self.FilePath = FilePath
        self.TextReport = TextReport
        self.StatusMessage = StatusMessage
        self.LabelReminder = LabelReminder
        self.text=textinput
        self.model=f1()
        self.nextflag=0
        self.stringindex=1
        


        self.curUnlabelAccIx = 0
        self.CurrentImgIndex = 0
        self.CurrentDrawPointVector = []
        self.BlankLabel = np.zeros((576,1021),np.uint8)#512,512,change here, now can draw on entire image
        #self.BlankLabel = np.zeros((1024,1024),np.uint8)#512,512,change here, now can draw on entire image
        self.CurrentLabel = self.BlankLabel
        self.CurrentDicom = np.array(cv2.imread('cat.jpg'),np.uint8)
        self.DrawStatus = False
        self.EraserStatus = False
        self.EditsMade = False
        self.NextFunction = ""
        self.FreezeKeyboard = False
        self.DrawInProgress = False
        self.DebugMode = False

        self.LoadContent()

        # Accept keyboard input
        '''
        self.Keyboard = Window.request_keyboard(
            self._keyboard_closed, self, 'text')##########################
        
        self.Keyboard.bind(on_key_down=self._on_keyboard_down)
        '''
        # Accept mouse input
        Window.bind(on_touch_move=self.on_touch_move)
        Window.bind(on_touch_down=self.on_touch_down)
        Window.bind(on_touch_up=self.on_touch_up)
####################################################################################################
    def DrawClick(self,event):
        self.DrawStatus = True
        self.EraserStatus = False
        self.LabelReminder.text=''
        if self.nextflag==0:
            self.text.text=''
        if self.nextflag==2:
            self.LabelReminder.text = 'Please click "Next page" first'
            
    def EraseClick(self,event):
        self.DrawStatus = False
        self.EraserStatus = True

    def SaveDrawingClick(self,event):
        self.EditsMade = True
        
        self.CurrentLabel =np.zeros((576,1021),np.uint8)
        
        self.CurrentDisplayImage =fx.CreateDisplayImage(self.CurrentDicom,np.zeros((576,1021),np.uint8))
        self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)
        self.nextflag=1
        self.LabelReminder.text = 'This is page '+str(self.stringindex)
    def DontSaveDrawingClick(self,event):
        self.DrawStatus = False
        self.EraserStatus = False
        self.EditsMade = False
    def Close(self,event):
        #App.get_running_app().stop()
        Window.close()

    def EraseAllClick(self,event):
        self.EditsMade = True
        #self.CurrentLabel = self.BlankLabel.copy()
        self.CurrentLabel =np.zeros((576,1021),np.uint8)
        #self.CurrentDisplayImage = fx.CreateDisplayImage(self.CurrentDicom, self.CurrentLabel)#
        self.CurrentDisplayImage =fx.CreateDisplayImage(self.CurrentDicom,np.zeros((576,1021),np.uint8))
        self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)
        self.text.text=''
        self.nextflag=0
        
    def Recognition(self,event):
        self.DrawStatus = False
        self.EraserStatus = False
        self.EditsMade = False
        p = fx.SetPNGCompression(0)
        
        im=self.CurrentDisplayImage
        path='sliced one/'
        im1=im[217:312,12:98]#86,4
        im2=im[217:312,102:188]
        im3=im[217:312,192:278]
        im4=im[217:312,282:368]
        im5=im[217:312,372:458]
        im6=im[217:312,462:548]
        im7=im[217:312,552:638]
        im8=im[217:312,642:728]
        im9=im[217:312,732:818]
        im10=im[217:312,822:908]
        im11=im[217:312,912:998]
         
        cv2.imwrite(path+'1.jpg',im1)
        cv2.imwrite(path+'2.jpg',im2)
        cv2.imwrite(path+'3.jpg',im3)
        cv2.imwrite(path+'4.jpg',im4)
        cv2.imwrite(path+'5.jpg',im5)
        cv2.imwrite(path+'6.jpg',im6)
        cv2.imwrite(path+'7.jpg',im7)
        cv2.imwrite(path+'8.jpg',im8)
        cv2.imwrite(path+'9.jpg',im9) 
        cv2.imwrite(path+'a.jpg',im10)
        cv2.imwrite(path+'b.jpg',im11)
        try:
            if self.nextflag==1:
                self.stringindex+=1
                #print('index',self.stringindex)
                self.text.text+=f2(self.model)##################################
                print('recognized',self.text.text)
                self.nextflag=2
            elif self.nextflag==0:
                self.text.text=f2(self.model)##################################
                print('recognized',self.text.text)
            
        except:
            self.LabelReminder.color=(1, .3, .3, 1)
            self.LabelReminder.text = 'All drawings lost unexpectedly, Please draw it again'
            
    
    def Calculate(self,event):
        list=[]
        string=self.text.text
        calculate=True
        empty=True
        if string!= '':
            empty=False
            if '=' not in string:#
                try:
                    result=eval(string)
                    list.append(result)
                    list.append('')
                except:
                    calculate=False 

            else:
                position=string.index('=')
                try:
                    left=eval(string[0:position])
                    right=eval(string[position+1:])
                    #print(left,right)
                    if type(left)==int:
                        if left==right:
                            list.append(left)
                            list.append(True)
                        else:
                            list.append(left)
                            list.append(False)
                            #print(list)
                    else:#double
                        if abs(np.round(left,2)-right)<abs(left*0.005):
                            #print(left)
                            list.append(left)
                            list.append(True)
                            #print(list)
                        else:
                            list.append(left)
                            list.append(False)
                            #print(list)
                    
                    
                    
                except:
                    calculate=False 
                    print('here')
                
                
            

        else:
            calculate=False
            empty=True
        ###################################################################
        #print(calculate)
                
                
        if calculate==False and empty==False:
            self.LabelReminder.color=(1, .3, .3, 1)
            self.LabelReminder.text = 'Wrong expression, please check it'
                    
        if list!= []:##########################can display result here
            if str(list[1])=='True':
                self.LabelReminder.color=(0, 1, 0, 1)
            else:
                self.LabelReminder.color=(1, .3, .3, 1)
            self.LabelReminder.text = 'Calculated result== '+str(list[0])+'  '+str(list[1])
            
        elif empty==True:
            self.LabelReminder.color=(1, .3, .3, 1)
            self.LabelReminder.text = ''

    def LoadContent(self):
        scanLabelText = 'NO LABEL'
        sliceLabelText = 'NO LABEL'

        self.CurrentDisplayImage = np.array(cv2.imread('cat.jpg'),np.uint8)
        self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)

        text = ''

        self.FilePath.text = text
        a=3*4
        
########################################
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        # print('The key', keycode[1], 'have been pressed')
        # print(' - text is %r' % text)
        # print(' - modifiers are %r' % modifiers)

        # Keycode is composed of an integer + a string
        # If we hit escape, release the keyboard
        if not self.FreezeKeyboard:
            print('keyboard input',keycode[1])
            return keycode[1]
            #textinput = TextInput(text='Hello world', multiline=False)
            '''
            if keycode[1] == 'escape':
                keyboard.release()
            elif keycode[1] == 'a':
                self.PrevSliceClick(event=None)
            elif keycode[1] == 'd':
                self.NextSliceClick(event=None)
            elif keycode[1] == 'z':
                self.PrevAccessionClick(event=None)
            elif keycode[1] == 'c':
                self.NextAccessionClick(event=None)
                '''
        # Return True to accept the key. Otherwise, it will be used by
        # the system.
        return True

    def _keyboard_closed(self):
        print('My keyboard have been closed!')
        self.Keyboard.unbind(on_key_down=self._on_keyboard_down)####################
        self.Keyboard = None############################################
        #self._keyboard.unbind(on_key_down=self._on_keyboard_down)####################
        #self._keyboard = None############################################

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
            
    def on_touch_down(self, x, touch):
        self.DrawInProgress = True

    def on_touch_up(self, x, touch):
        self.mouse_x,self.mouse_y = self.ConvertToImageCoords(touch.x,touch.y)
        if self.mouse_x >= 0 and self.mouse_x < 1021 and self.mouse_y >= 0 and self.mouse_y < 576:#was 512
            if self.DrawInProgress and self.EraserStatus:
                self.EditsMade = True
                self.FloodFillBlob(self.mouse_x,self.mouse_y,0)############
            if self.DrawInProgress and self.DrawStatus and len(self.CurrentDrawPointVector)>0:
                self.EditsMade = True
                if len(self.CurrentDrawPointVector) > 0:#was 10
                    self.DrawFilledPolygon()
                else:
                    self.CurrentDisplayImage = np.array(cv2.imread('cat.jpg'),np.uint8)
                    self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)
        if self.DrawInProgress and self.DrawStatus:
            del self.CurrentDrawPointVector[:]
        self.DrawInProgress = False

    def ConvertToImageCoords(self,x,y):
        c = x#
        #r = 512 + 40 + 200 - y
        #if x>400:
            #c = x-(510-x)*0.10
        
        #else:
            #c = x-(510-x)*0.10
        
        r = 800- y+20-152#keep it
        c = int(round(c))#nothing to change here
        r = int(round(r))
        return c,r

    def DrawFilledPolygon(self):
        pointvector = np.array(self.CurrentDrawPointVector,np.int32)
        #cv2.fillPoly(self.CurrentLabel, [pointvector], 255)
        #polylines(img, pts, isClosed, color, thickness)
        #cv2.line(self.CurrentLabel, [pointvector],(255,255,255),5)
        cv2.polylines(self.CurrentLabel, [pointvector], 0, (255,255,255),1)#0 thich
        self.CurrentDisplayImage = fx.CreateDisplayImage(self.CurrentDicom, self.CurrentLabel)
        self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)

    def FloodFillBlob(self, x, y, i):
        #print(np.sum(self.CurrentLabel))
        if self.CurrentLabel[y-20:y+20,x-20:x+20].any()!= i:
            #print('erase')
            self.CurrentLabel[y-20:y+20,x-20:x+20]= i
            #mask = np.zeros((576 + 2, 1021 + 2), np.uint8)
            #cv2.floodFill(self.CurrentLabel,mask,(x,y),255,0,0, cv2.FLOODFILL_FIXED_RANGE | 8)#i
            self.CurrentDisplayImage = fx.CreateDisplayImage(self.CurrentDicom, self.CurrentLabel)
            self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)








