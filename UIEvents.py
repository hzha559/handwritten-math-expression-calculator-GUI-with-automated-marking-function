from kivy.core.window import Window
import numpy as np
import cv2
import Functions as fx
from model import network #import the neural network in model.py 
from model import recognize #import the recognition function in model.py 
 
class UIEvents():
    #define all the functions of the GUI
    #linked with the windows object. ImageViewer is the drawing window, LabelReminder is the interactives texts and textinput is the TextField
    def __init__(self,ImageViewer,LabelReminder,textinput):
        self.ImageViewer = ImageViewer
        self.LabelReminder = LabelReminder  
        self.text=textinput  
        self.model,self.transform=network() #loading the network and image transform tehniques from model.py
        #self.nextflag=0  #record if the "Next Page" button is pressed
        self.path='sliced one/'  #the location to store the handwriting
        self.CurrentDrawPointVector = []  #the position of each stroke
        self.BlankLabel = np.zeros((576,1021),np.uint8)    
        self.CurrentLabel = self.BlankLabel
        self.CurrentDicom = np.array(cv2.imread('cat.jpg'),np.uint8)  #the background
        self.DrawStatus = False
        self.EraserStatus = False
        self.DrawInProgress = False
        #self.pageindex=0  #page number, to deal with multiple pages
        self.index=0 #store the position index of the recognized result in the TextField
        #self.havedraw=0 #determine whether the user has modified the drawing
        self.LoadContent()  
        self.previous=''

        # Accept mouse input
        Window.bind(on_touch_move=self.on_touch_move)
        Window.bind(on_touch_down=self.on_touch_down)
        Window.bind(on_touch_up=self.on_touch_up)
        
####################################################################################################
    def DrawClick(self,event):
        # this function is to used to draw a stroke by allowing comditions for the "on-touch-up" below
        self.DrawStatus = True
        self.EraserStatus = False
        #self.havedraw=0 #indicating haven't make any drawing
        
        self.LabelReminder.color=(1, .3, .3, 1)
        self.LabelReminder.text = 'Drawing in progress' # display in the interactive text in red
            
    def EraseClick(self,event):
        # this function is used to erase a stroke by allowing comditions for the "on-touch-up" below
        self.havedraw=0
        self.DrawStatus = False
        self.EraserStatus = True
        self.LabelReminder.color=(1, .3, .3, 1)
        self.LabelReminder.text = 'Erasing in progress' # display in the interactive text in red

    def EraseAllClick(self,event):
        # abandon anything on the GUI, including things in the TextField
        self.LabelReminder.color=(1, .3, .3, 1)
        self.CurrentLabel =np.zeros((576,1021),np.uint8)
        self.CurrentDisplayImage =fx.CreateDisplayImage(self.CurrentDicom,np.zeros((576,1021),np.uint8))
        self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage) # refresh the GUI
        self.LabelReminder.text='All drawings, including those on the previous pages are deleted'
        self.havedraw=0
        self.text.text='' # clear the TextField
        self.previous='' # clear the recongnition result of the previous cells
        
    def Recognition(self,event):
        # call the neural network and perform recogniton 
        # return the recognized results to the TextField
        self.DrawStatus = False
        self.EraserStatus = False      
        im=self.CurrentDisplayImage
        image=[[] for i in range(11)]
        
        for i in range(11):
            image[i]=im[217:312,12+90*i:98+90*i]
            if i<10:
                cv2.imwrite(self.path+str(i)+'.jpg',image[i])
            else:
                cv2.imwrite(self.path+'a'+'.jpg',image[i]) # save individual symbols as 11 images 
        #print(self.nextflag)
        try:
            if event=='w' or event=='e': # if handwriting not in the last cell, recognize the current 11 cells
                self.text.text=self.previous+recognize(self.model,self.path,self.transform,11)#self.text.text[0:self.index]+
                #print('recognized',self.text.text) 
                self.index=len(self.text.text) # the index of current iteration recognition 
                
            elif event=='11': # if handwriting in the last cell, save the 3 cells that will be shifted out soon to 'previous'
                self.previous+=recognize(self.model,self.path,self.transform,3)
                #print('recognized',self.text.text) 
        
        except: # this occurs very rare that saved images of math expression are lost
            self.havedraw=0
            self.LabelReminder.color=(1, .3, .3, 1) 
            self.LabelReminder.text = 'All drawings lost unexpectedly, Please draw it again'
            
        if event=='w': # continue writing or erasing after recognition
            self.DrawStatus = True
            self.EraserStatus = False
        elif event=='e':
            self.DrawStatus = False
            self.EraserStatus = True
            
    
    def Calculate(self,event):
        # perform calculation based on the expression in the TextField
        # display the result in the interactive texts
        resultlist=[]
        string=self.text.text  # the TextField expression
        print(string)
        calculate=True
        empty=True
        
        if string!= '':
            empty=False  # if not empty
            if '=' not in string:  # if it is not an equation
                try:
                    result=eval(string)
                    result=np.round(result,3)
                    resultlist.append(result)
                    resultlist.append('') # if eval() works on the expression, it will return the result 
                    
                except:
                    calculate=False  # else do not proceed

            else: # the input is an equation
                position=string.index('=') # find the position of the equal sign to segment the left and right hand side expressions
                try:
                    left=eval(string[0:position]) 
                    right=eval(string[position+1:]) #calculate left and right hand side expressions
                    #print(left,right)
                    if type(left)==int: # if left hand side expression generates an int result
                        if left==right:
                            resultlist.append(left)
                            resultlist.append('Correct') 
                            # require the right hand side expression to precisely equal to it to be regarded as correct
                        else:
                            resultlist.append(left)
                            resultlist.append('Wrong')
                            
                    else:  # if left hand side expression yields a float
                        if abs(np.round(left,2)-right)<abs(left*0.005):  
                        # require the error between the left hand side and right hand side equation to be less than 0.5% to be regarded as correct
                            #print(left)
                            left=np.round(left,3)
                            resultlist.append(left)
                            resultlist.append('Correct')
                            #print(list)
                        else:
                            left=np.round(left,3)
                            resultlist.append(left)
                            resultlist.append('Wrong')
                            #print(list)

                except: # invalid expression in this case
                    calculate=False 

        else: # the expression is empty
            calculate=False
            empty=True 
        ################################################################### 
        # below controls the display in the interactive texts
                
        if calculate==False and empty==False: # if the expression is invalid
            self.LabelReminder.color=(1, .3, .3, 1) # warning message in red
            self.LabelReminder.text = 'Wrong expression, please check it'
                    
        if resultlist!= []:
            if resultlist[1]=='Wrong':
                self.LabelReminder.color=(1, .3, .3, 1) # if the calculation result is wrong, showing 'Wrong' and the correct result in red
            else:
                self.LabelReminder.color=(0, 1, 0, 1) # if the calculation result is correct, showing 'Correct' and the correct result in green
            self.LabelReminder.text = 'The left-hand expression equals to '+str(resultlist[0])+'  '+str(resultlist[1])
            
        elif empty==True:
            self.LabelReminder.color=(1, .3, .3, 1) # if the expression is empty, give warning in red
            self.LabelReminder.text = 'Cannot calculate because nothing is recognized'
        else:
            self.LabelReminder.text = 'Wrong expression, please check it'


    def LoadContent(self):
        # load the background which contains grids
        self.CurrentDisplayImage = np.array(cv2.imread('cat.jpg'),np.uint8)
        self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage) 
        
    def on_touch_move(self, x, touch):
        # when the mouse is still drawing
        self.mouse_x,self.mouse_y = self.ConvertToImageCoords(touch.x,touch.y)
        if self.mouse_x >= 0 and self.mouse_x < 1021 and self.mouse_y >= 0 and self.mouse_y < 576: # within the drawing zone
            # print "Mouse Up ( " + str(self.mouse_x) + " , " + str(self.mouse_y) + " )"
            if self.DrawInProgress and self.DrawStatus :
                self.EditsMade = True
                self.CurrentDrawPointVector.append([self.mouse_x,self.mouse_y]) # Display user' current drawing in dotted lines
                if self.mouse_x < 1021 - 1 and self.mouse_x > 0 + 1 and self.mouse_y < 576 - 1 and self.mouse_y > 0 + 1:
                    self.CurrentDisplayImage[self.mouse_y-1:self.mouse_y+1,self.mouse_x-1:self.mouse_x+1]=(250,100,100)
                    self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)
            
    def on_touch_down(self, x, touch): # when the mouse is hold, allow drawing
        self.DrawInProgress = True

    def on_touch_up(self, x, touch): 
        # when the mouse is released, perpare to show the effect
        self.mouse_x,self.mouse_y = self.ConvertToImageCoords(touch.x,touch.y)
        if self.mouse_x >= 0 and self.mouse_x < 1021 and self.mouse_y >= 0 and self.mouse_y < 576: # if the mouse is within the drawing zone
            
            if self.DrawInProgress and self.EraserStatus:
                self.FloodFillBlob(self.mouse_x,self.mouse_y,0)  # allow erasing if erase is clicked
                
                self.Recognition(event='e')
            if self.DrawInProgress and self.DrawStatus and len(self.CurrentDrawPointVector)>0: # if more than 1 coordinates has the mouse draw, enable drawing the 8-connected lines
                 self.DrawFilledPolygon()
            del self.CurrentDrawPointVector[:]
        
        if self.DrawInProgress and self.DrawStatus: # to be ready to draw the next stroke
            del self.CurrentDrawPointVector[:] #claer the current stroke array

            if self.mouse_x<1000: # if the mouse is within the writing area
                self.Recognition(event='w')
            if self.mouse_x>912 and self.mouse_x<998 and self.CurrentLabel[:,912:998].any()!= 0 and self.mouse_y>217 and self.mouse_y< 312:
                # if something is written in the last cell, shift the equation left for 3 cells
                self.DrawStatus = False 
                self.Recognition(event='11')
                self.CurrentLabel=np.concatenate((self.CurrentLabel[:,270:],np.zeros((576,270),np.uint8)),axis=1)# shift by padding zeros on the right
                self.CurrentDisplayImage =fx.CreateDisplayImage(self.CurrentDicom,self.CurrentLabel)
                self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage) # update the display
                self.EraserStatus = False

        
        self.DrawInProgress = False

    def ConvertToImageCoords(self,x,y):
        #change the coordinates of the mouse since in the GUI the y axis is opposite to the convention
        c = x
        r = 800- y+20-152
        c = int(round(c))
        r = int(round(r))  
        return c,r

    def DrawFilledPolygon(self):
        # used to draw a single stroke
        pointvector = np.array(self.CurrentDrawPointVector,np.int32)
        cv2.polylines(self.CurrentLabel, [pointvector], 0, (255,255,255),1)  # use 8-connected lines for drawing
        self.CurrentDisplayImage = fx.CreateDisplayImage(self.CurrentDicom, self.CurrentLabel) # update the change immediately
        self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)

    def FloodFillBlob(self, x, y, i):
        # used to erase a single stroke
        if self.CurrentLabel[y-20:y+20,x-20:x+20].any()!= i:  # if any of the 20 adjacent pixels are not white, erase them
            self.CurrentLabel[y-20:y+20,x-20:x+20]= i
            self.CurrentDisplayImage = fx.CreateDisplayImage(self.CurrentDicom, self.CurrentLabel)
            self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)








