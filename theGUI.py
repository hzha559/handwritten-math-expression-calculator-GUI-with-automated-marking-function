
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics','borderless',0)
Config.set('graphics','fullscreen',0)
Config.set('graphics', 'width', '1350')
Config.set('graphics', 'height', '600')  #define the properties of the main window
Config.set('graphics', 'resizable', 0)
Config.write()
from kivy.app import App
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from UIEvents import UIEvents

from kivy.uix.textinput import TextInput


class GUI:
    def __init__(self):
        self.root = RootWidget()
        self.root.run()
# used to launch the entire GUI

class RootWidget(App):
    def build(self):
        self.Main = MainWindow()
        return self.Main

    def on_stop(self):
        self.Main.on_stop()

class WorkPanel(RelativeLayout): # the work panel contains all the buttons

    def __init__(self,uievents, **kwargs):
        super(WorkPanel, self).__init__(**kwargs)
        self.uievents = uievents

        Instructions = Label(font_size="16sp")  # the title of this program
        Instructions.size_hint = (None, None)
        Instructions.size = (270, 200)
        Instructions.pos = (1045, 330-152)
        Instructions.color = (0, 1, 0, 1)  # the text is written in green
        Instructions.text = "Handwritten math expression calculator"
        self.add_widget(Instructions)

        ########### defining buttons below ###################################################
        btnDraw = Button(text="Draw")  # defining the drawing button
        btnDraw.size_hint = (None, None)
        btnDraw.size = (130, 40)
        btnDraw.pos = (1040, 320-152) # define the button's location in the GUI
        btnDraw.color = (1, 1, 1, 1) # the text color is white
        btnDraw.bind(on_release=self.uievents.DrawClick) # bind the buttons with "DrawClick" function in UIEvent
        self.add_widget(btnDraw)

        btnErase = Button(text="Erase")
        btnErase.size_hint = (None, None)
        btnErase.size = (130, 40)
        btnErase.pos = (1180, 320-152)
        btnErase.color = (1, 1, 1, 1)
        btnErase.bind(on_release=self.uievents.EraseClick) # bind the buttons with "EraseClick" function in UIEvent
        self.add_widget(btnErase)
        '''
        btnNextPage = Button(text="Next Page")
        btnNextPage.size_hint = (None, None)
        btnNextPage.size = (130, 40)
        btnNextPage.pos = (1040, 260-152)
        btnNextPage.color = (1, 1, 1, 1)
        btnNextPage.bind(on_release=self.uievents.NextPageClick) # bind the buttons with "NextPageClick" function in UIEvent
        self.add_widget(btnNextPage)
        '''
        btnEraseAll = Button(text="Erase All")
        btnEraseAll.size_hint = (None, None)
        btnEraseAll.size = (130, 40)
        btnEraseAll.pos = (1180, 260-152)
        btnEraseAll.color = (1, 1, 1, 1)
        btnEraseAll.bind(on_release=self.uievents.EraseAllClick) # bind the buttons with "EraseAllClick" function in UIEvent
        self.add_widget(btnEraseAll)
        '''
        btnRecogntion = Button(text="Recognition")
        btnRecogntion.size_hint = (None, None)
        btnRecogntion.size = (130, 40)
        btnRecogntion.pos = (1040, 170-152)
        btnRecogntion.color = (1, 1, 1, 1)
        btnRecogntion.bind(on_release=self.uievents.Recognition) # bind the buttons with "Recognition" function in UIEvent
        self.add_widget(btnRecogntion)
        
        Close = Button(text="Close")
        Close.size_hint = (None, None)
        Close.size = (130, 40)
        Close.pos = (1040, 110-152)
        Close.color = (1, 1, 1, 1)
        Close.bind(on_release=self.uievents.Close) # bind the buttons with "Close" function in UIEvent
        self.add_widget(Close)
        '''
        Calculate = Button(text="Calculate")
        Calculate.size_hint = (None, None)
        Calculate.size = (130, 40)
        Calculate.pos = (1040, 260-152)
        Calculate.color = (1, 1, 1, 1)
        Calculate.bind(on_release=self.uievents.Calculate) # bind the buttons with "Calculate" function in UIEvent
        self.add_widget(Calculate)


class MainWindow(RelativeLayout): 
    # the main window which contains the interactive writing window,the interactive texts and the TextField

    def __init__(self,**kwargs):
        super(MainWindow, self).__init__(**kwargs)

        ImageViewer = Image(source="") # the interactive writing window
        ImageViewer.size_hint = (None, None)
        ImageViewer.size = (1021, 576)
        ImageViewer.pos = (0,240-152)
        self.add_widget(ImageViewer)
        
        # The reason to put the interactive texts and the TextField in the MainWindow is that MainWindow can change according to time
        
        LabelReminder = Label(font_size="15sp", text="", color=(1, .3, .3, 1), halign='left', text_size=(300, 150))
        LabelReminder.size_hint = (None, None)  # the interactive text, which is initially set to red
        LabelReminder.size = (300, 150)
        LabelReminder.valign = 'top'
        LabelReminder.text = ""
        LabelReminder.pos = (1050, 210-152)
        self.add_widget(LabelReminder)
        
        textinput=TextInput(multiline=False)  #the TextField
        textinput.size_hint = (None, None)
        textinput.size = (300, 40)
        textinput.pos = (1040, 250-152)
        self.add_widget(textinput)
        
        self.uievents = UIEvents(ImageViewer,LabelReminder,textinput)

        WorkingPanel = WorkPanel(self.uievents)
        WorkingPanel.pos = (0,240)
        self.add_widget(WorkingPanel)
        
    def on_stop(self):
        App.get_running_app().stop() #the function that shuts down the thread


GUI()




