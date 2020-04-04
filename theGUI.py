
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics','borderless',0)
Config.set('graphics','fullscreen',0)
Config.set('graphics', 'width', '1350')
Config.set('graphics', 'height', '600')#752
Config.set('graphics', 'resizable', 0)
Config.write()
from kivy.app import App
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from UIEvents import UIEvents
from UIEvents import f1
from kivy.uix.textinput import TextInput


class GUI:
    def __init__(self):
        self.root = RootWidget()
        self.root.run()


class RootWidget(App):
    def build(self):
        self.Main = MainWindow()
        return self.Main#

    def on_stop(self):
        self.Main.on_stop()

class WorkPanel(RelativeLayout):

    def __init__(self,uievents, **kwargs):
        super(WorkPanel, self).__init__(**kwargs)
        self.uievents = uievents

        Instructions = Label(font_size="16sp")
        Instructions.size_hint = (None, None)
        Instructions.size = (270, 200)
        Instructions.pos = (1045, 330-152)
        Instructions.color = (0, 1, 0, 1)
        Instructions.text = "Handwritten math expression calculator"
        self.add_widget(Instructions)

        btnDraw = Button(text="Draw")
        btnDraw.size_hint = (None, None)
        btnDraw.size = (130, 40)
        btnDraw.pos = (1040, 320-152)
        btnDraw.color = (1, 1, 1, 1)
        btnDraw.bind(on_release=self.uievents.DrawClick)
        self.add_widget(btnDraw)

        btnErase = Button(text="Erase")
        btnErase.size_hint = (None, None)
        btnErase.size = (130, 40)
        btnErase.pos = (1180, 320-152)
        btnErase.color = (1, 1, 1, 1)
        btnErase.bind(on_release=self.uievents.EraseClick)
        self.add_widget(btnErase)

        btnSaveDrawing = Button(text="Save")
        btnSaveDrawing.size_hint = (None, None)
        btnSaveDrawing.size = (130, 40)
        btnSaveDrawing.pos = (1040, 260-152)
        btnSaveDrawing.color = (1, 1, 1, 1)
        btnSaveDrawing.bind(on_release=self.uievents.SaveDrawingClick)
        self.add_widget(btnSaveDrawing)

        btnEraseAll = Button(text="Erase All")
        btnEraseAll.size_hint = (None, None)
        btnEraseAll.size = (130, 40)
        btnEraseAll.pos = (1180, 260-152)
        btnEraseAll.color = (1, 1, 1, 1)
        btnEraseAll.bind(on_release=self.uievents.EraseAllClick)
        self.add_widget(btnEraseAll)
        
        btnPrevSlice = Button(text="Recognition")
        btnPrevSlice.size_hint = (None, None)
        btnPrevSlice.size = (130, 40)
        btnPrevSlice.pos = (1040, 170-152)
        btnPrevSlice.color = (1, 1, 1, 1)
        btnPrevSlice.bind(on_release=self.uievents.Recognition)
        self.add_widget(btnPrevSlice)
        
        Close = Button(text="Close")
        Close.size_hint = (None, None)
        Close.size = (130, 40)
        Close.pos = (1040, 110-152)
        Close.color = (1, 1, 1, 1)
        Close.bind(on_release=self.uievents.Close)
        self.add_widget(Close)
        

        Calculate = Button(text="Calculate")
        Calculate.size_hint = (None, None)
        Calculate.size = (130, 40)
        Calculate.pos = (1180, 170-152)
        Calculate.color = (1, 1, 1, 1)
        Calculate.bind(on_release=self.uievents.Calculate)
        self.add_widget(Calculate)


class MainWindow(RelativeLayout):

    def __init__(self,**kwargs):
        super(MainWindow, self).__init__(**kwargs)#########chang

        ImageViewer = Image(source="")
        ImageViewer.size_hint = (None, None)
        #ImageViewer.size = (1029, 512)
        ImageViewer.size = (1021, 576)
        ImageViewer.pos = (0,240-152)
        self.add_widget(ImageViewer)
        
        FilePath = Label(font_size="14sp", text="", color=(1,1,1,1),halign='left',text_size=(1000,100))
        FilePath.size_hint = (None, None)
        FilePath.size = (1000, 100)
        FilePath.pos = (20, 140-152)
        self.add_widget(FilePath)
        
        TextReport = Label(font_size="14sp", text="", color=(1, 1, 1, 1), halign='left', text_size=(1300, 150))
        TextReport.size_hint = (None, None)
        TextReport.size = (1300, 150)
        TextReport.valign = 'top'
        TextReport.text = ""
        TextReport.pos = (20, 0-152)
        self.add_widget(TextReport)

        StatusMessage = Label(font_size="14sp", text="", color=(1, 1, 1, 1), halign='left', text_size=(1300, 150))
        StatusMessage.size_hint = (None, None)
        StatusMessage.size = (1300, 150)
        StatusMessage.valign = 'top'
        StatusMessage.text = ""
        StatusMessage.pos = (20, -70-152)
        self.add_widget(StatusMessage)

        LabelReminder = Label(font_size="15sp", text="", color=(1, .3, .3, 1), halign='left', text_size=(300, 150))
        LabelReminder.size_hint = (None, None)
        LabelReminder.size = (300, 150)
        LabelReminder.valign = 'top'
        LabelReminder.text = ""
        LabelReminder.pos = (1050, 185-152)
        self.add_widget(LabelReminder)
        
        textinput=TextInput(multiline=False)
        textinput.size_hint = (None, None)
        textinput.size = (300, 40)#130,40
        textinput.pos = (1040, 250-152)#1040,170
        #textinput.bind(text=self.uievents.Recognition)
        self.add_widget(textinput)
        
        
        self.uievents = UIEvents(ImageViewer,FilePath,TextReport,StatusMessage,LabelReminder,textinput)

        WorkingPanel = WorkPanel(self.uievents)
        WorkingPanel.pos = (0,240)
        self.add_widget(WorkingPanel)
        
    def on_stop(self):
        App.get_running_app().stop()


GUI()




