import os
from kivy.core.window import Window
import numpy as np
import cv2
import pydicom as dicom
import json
import Functions as fx
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.button import Button
import struct

class UIEvents():
    def __init__(self,ImageViewer,FilePath,TextReport,StatusMessage,LabelReminder):
        self.ImageViewer = ImageViewer
        self.FilePath = FilePath
        self.TextReport = TextReport
        self.StatusMessage = StatusMessage
        self.LabelReminder = LabelReminder
        # self.DatasetPath = "C:/Users/MIME Project/Dropbox/MIME/Alfred FCN/uitest/"
        with open('Settings.json','r') as f:
            self.settings = json.load(f)


        self.curUnlabelAccIx = 0
        self.CurrentImgIndex = 0
        self.CurrentDrawPointVector = []
        self.BlankLabel = np.zeros((512,512),np.uint8)
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
        self.Keyboard = Window.request_keyboard(
            self._keyboard_closed, self, 'text')
        self.Keyboard.bind(on_key_down=self._on_keyboard_down)

        # Accept mouse input
        Window.bind(on_touch_move=self.on_touch_move)
        Window.bind(on_touch_down=self.on_touch_down)
        Window.bind(on_touch_up=self.on_touch_up)
####################################################################################################
    def DrawClick(self,event):
        self.DrawStatus = True
        self.EraserStatus = False

    def EraseClick(self,event):
        self.DrawStatus = False
        self.EraserStatus = True

    def SaveDrawingClick(self,event):
        self.DrawStatus = False
        self.EraserStatus = False
        self.EditsMade = False
        p = fx.SetPNGCompression(0)
        cv2.imwrite(self.CurrentLabelPath,self.CurrentLabel, p)

    def DontSaveDrawingClick(self,event):
        self.DrawStatus = False
        self.EraserStatus = False
        self.EditsMade = False

    def EraseAllClick(self,event):
        self.EditsMade = True
        self.CurrentLabel = self.BlankLabel.copy()
        self.CurrentDisplayImage = fx.CreateDisplayImage(self.CurrentDicom, self.CurrentLabel)
        self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)

    def PrevSliceClick(self,event):
        self.NextFunction = "PrevSliceClick"
        self.CheckEditsMade()

    def NextSliceClick(self,event):
        self.NextFunction = "NextSliceClick"
        self.CheckEditsMade()

    def PrevAccessionClick(self,event):
        self.NextFunction = "PrevAccessionClick"
        self.CheckEditsMade()

    def NextAccessionClick(self,event):
        self.NextFunction = "NextAccessionClick"
        self.CheckEditsMade()

    def ScanFungalClick(self,event):
        fx.SaveScanLabel(self.settings,self.state['listUnlabelledAccessions'][self.curUnlabelAccIx],'fungalProb',1)
        self.LoadContent()

    def ScanNormalClick(self,event):
        fx.SaveScanLabel(self.settings,self.state['listUnlabelledAccessions'][self.curUnlabelAccIx],'fungalProb',0)
        self.LoadContent()

    def SliceSpecificClick(self,event):
        acc = self.state['listUnlabelledAccessions'][self.curUnlabelAccIx]
        fx.SaveSliceLabel(self.settings,acc,self.dictAllSlices[acc][self.CurrentImgIndex],'fungalProb',1)
        self.LoadContent()

    def SliceNormalClick(self,event):
        acc = self.state['listUnlabelledAccessions'][self.curUnlabelAccIx]
        fx.SaveSliceLabel(self.settings,acc,self.dictAllSlices[acc][self.CurrentImgIndex],'fungalProb',0)
        self.LoadContent()

    def AgreementClick(self,event):
        try:
            scans, slices = EvaluationMetrics.CalculateAllMetrics(self.settings, self.dictAllSlices)
            # iou = 'Overlap IOU = [ {:6.4f} , {:6.4f} , {:6.4f} ]'.format(pixels['IOU'][0],pixels['IOU'][1],pixels['IOU'][2])
            scanErrors = ''
            if sum(scans['errorCount']) > 0:
                scanErrors = '\t Scan labels INCOMPLETE: '+str(sum(scans['errorCount']))+' errors found! '+ str(scans['errorCount'])
            if sum(slices['errorCount']) > 0:
                scanErrors += '\t Slice labels INCOMPLETE: '+str(sum(slices['errorCount']))+' errors found! '+ str(slices['errorCount'])

            self.StatusMessage.text = 'Raters: 1 Samantha, 2 Anthony, 3 Dinesh \n'\
                                        + 'Scan level:'+'IRA='+str(scans['IRA'][1:])+', TP='+str(scans['scanTP'][1:])+', FP='+str(scans['scanFP'][1:])+', FN='+str(scans['scanFN'][1:])+', TN='+str(scans['scanTN'][1:]) +'\n'\
                                        + 'Slice level:'+'IRA='+str(slices['IRA'][1:])+', TP='+str(slices['sliceTP'][1:])+', FP='+str(slices['sliceFP'][1:])+', FN='+str(slices['sliceFN'][1:])+', TN='+str(slices['sliceTN'][1:]) +'\n'\
                                        + scanErrors + '\n'\


                                      # +'Blob FP = '+str(pixels['blobFP'][1:])+' , '\
                                      #   +'Blob FN = '+str(pixels['blobFN'][1:])+' \n '\
                                      #   + iou +' \n '\
                                      #'True fungal count = '+str(pixels['blobTrue'])+' , '\
        except Exception as e:
            self.StatusMessage.text = 'Error calculating metrics: '+str(e)

    def DebugModeClick(self,event):
        self.DebugMode = ~self.DebugMode

########################################

    def FetchSlice(self,i):
        # CurrentImgNum starts from 1
        self.CurrentImgIndex = self.CurrentImgIndex + i
        acc = self.state['listUnlabelledAccessions'][self.curUnlabelAccIx]
        if self.CurrentImgIndex < 0:
            self.CurrentImgIndex = len(self.dictAllSlices[acc])-1
        elif self.CurrentImgIndex >= len(self.dictAllSlices[acc]):
            self.CurrentImgIndex = 0

        self.LoadContent()

    def FetchAccession(self,i):
        self.curUnlabelAccIx = self.curUnlabelAccIx + i
        if self.curUnlabelAccIx < 0:
            self.curUnlabelAccIx = len(self.state['listUnlabelledAccessions'])-1
        if self.curUnlabelAccIx >= len(self.state['listUnlabelledAccessions']):
            self.curUnlabelAccIx = 0

        self.CurrentImgIndex = 0
        self.LoadContent()



    def LoadContent(self):
        scanLabelText = 'NO LABEL'
        sliceLabelText = 'NO LABEL'

        self.CurrentDisplayImage = np.array(cv2.imread('cat.jpg'),np.uint8)
        self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)




        text = 'test test test'



        self.FilePath.text = text

        if scanLabelText == 'NO LABEL':
            self.LabelReminder.text = 'Scan not labelled!'
        else:
            self.LabelReminder.text = ''

########################################
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        # print('The key', keycode[1], 'have been pressed')
        # print(' - text is %r' % text)
        # print(' - modifiers are %r' % modifiers)

        # Keycode is composed of an integer + a string
        # If we hit escape, release the keyboard
        if not self.FreezeKeyboard:
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
        # Return True to accept the key. Otherwise, it will be used by
        # the system.
        return True

    def _keyboard_closed(self):
        print('My keyboard have been closed!')
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def on_touch_move(self, x, touch):
        self.mouse_x,self.mouse_y = self.ConvertToImageCoords(touch.x,touch.y)
        if self.mouse_x >= 0 and self.mouse_x < 512 and self.mouse_y >= 0 and self.mouse_y < 512:
            # print "Mouse Up ( " + str(self.mouse_x) + " , " + str(self.mouse_y) + " )"
            if self.DrawInProgress and self.DrawStatus :
                self.EditsMade = True
                self.CurrentDrawPointVector.append([self.mouse_x,self.mouse_y])
                # Display user drawing outline
                if self.mouse_x < 512 - 1 and self.mouse_x > 0 + 1 and self.mouse_y < 512 - 1 and self.mouse_y > 0 + 1:
                    self.CurrentDisplayImage[self.mouse_y-1:self.mouse_y+1,self.mouse_x+512+5-1:self.mouse_x+512+5+1]=(250,100,100)
                    self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)
            
    def on_touch_down(self, x, touch):
        self.DrawInProgress = True

    def on_touch_up(self, x, touch):
        self.mouse_x,self.mouse_y = self.ConvertToImageCoords(touch.x,touch.y)
        if self.mouse_x >= 0 and self.mouse_x < 512 and self.mouse_y >= 0 and self.mouse_y < 512:
            if self.DrawInProgress and self.EraserStatus:
                self.EditsMade = True
                self.FloodFillBlob(self.mouse_x,self.mouse_y,0)
            if self.DrawInProgress and self.DrawStatus and len(self.CurrentDrawPointVector)>0:
                self.EditsMade = True
                if len(self.CurrentDrawPointVector) > 10:
                    self.DrawFilledPolygon()
                else:
                    self.CurrentDisplayImage = np.array(cv2.imread('cat.jpg'),np.uint8)
                    self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)
        if self.DrawInProgress and self.DrawStatus:
            del self.CurrentDrawPointVector[:]
        self.DrawInProgress = False

    def ConvertToImageCoords(self,x,y):
        c = x - 512 - 5
        r = 512 + 40 + 200 - y
        c = int(round(c))
        r = int(round(r))
        return c,r

    def DrawFilledPolygon(self):
        pointvector = np.array(self.CurrentDrawPointVector,np.int32)
        cv2.fillPoly(self.CurrentLabel, [pointvector], 255)
        self.CurrentDisplayImage = fx.CreateDisplayImage(self.CurrentDicom, self.CurrentLabel)
        self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)

    def FloodFillBlob(self, x, y, i):
        if self.CurrentLabel[y,x]!= i:
            mask = np.zeros((512 + 2, 512 + 2), np.uint8)
            cv2.floodFill(self.CurrentLabel,mask,(x,y),i,0,0, cv2.FLOODFILL_FIXED_RANGE | 8)
            self.CurrentDisplayImage = fx.CreateDisplayImage(self.CurrentDicom, self.CurrentLabel)
            self.ImageViewer.texture = fx.RenderDisplayImage(self.CurrentDisplayImage)



    def CheckEditsMade(self):
        if self.EditsMade:
            self.FreezeKeyboard = True
            self.SaveConfirmPopup = Popup(title='Do you want to save?',
                          content=ConfirmDialog(self),
                          size_hint=(None, None), size=(400, 200), auto_dismiss=False)
            self.SaveConfirmPopup.open()
        else:
            self.CheckEditsMadeComplete(False)


    def PopupDontSave(self,event):
        self.DontSaveDrawingClick(event)
        self.CheckEditsMadeComplete(True)

    def PopupSave(self,event):
        self.SaveDrawingClick(event)
        self.CheckEditsMadeComplete(True)

    def CheckEditsMadeComplete(self,popup):
        if popup:
            self.SaveConfirmPopup.dismiss()
            self.FreezeKeyboard = False
        if self.NextFunction == "PrevSliceClick":
            self.FetchSlice(-1)
        elif self.NextFunction == "NextSliceClick":
            self.FetchSlice(1)
        elif self.NextFunction == "PrevAccessionClick":
            self.CurrentImgNum = 1
            self.FetchAccession(-1)
        elif self.NextFunction == "NextAccessionClick":
            self.CurrentImgNum = 1
            self.FetchAccession(1)

        self.NextFunction = ""


#########################################################################################################

#########################################################################################################


class ConfirmDialog(RelativeLayout):

    def __init__(self, uievents, **kwargs):
        super(ConfirmDialog, self).__init__(**kwargs)
        self.uievents = uievents

        label = Label(text='Edits have been made.\nDo you want to save?')
        label.size_hint = (None, None)
        label.size = (400, 100)
        label.pos = (0, 50)

        btnDontSave = Button(text='Don\'t Save')
        btnDontSave.size_hint = (None, None)
        btnDontSave.size = (170, 50)
        btnDontSave.pos = (0, 0)
        btnDontSave.color = (1, 1, 1, 1)
        btnDontSave.bind(on_release=uievents.PopupDontSave)

        btnSave = Button(text='Save')
        btnSave.size_hint = (None, None)
        btnSave.size = (170, 50)
        btnSave.pos = (200, 0)
        btnSave.color = (1, 1, 1, 1)
        btnSave.bind(on_release=uievents.PopupSave)

        self.add_widget(label)
        self.add_widget(btnDontSave)
        self.add_widget(btnSave)







