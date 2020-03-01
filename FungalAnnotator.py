
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics','borderless',0)
Config.set('graphics','fullscreen',0)
Config.set('graphics', 'width', '1350')
Config.set('graphics', 'height', '752')
Config.set('graphics', 'resizable', 0)
Config.write()
from kivy.app import App
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from UIEvents import UIEvents
from kivy.uix.textinput import TextInput


class RadiologistGUI:
    def __init__(self):
        self.root = RootWidget()
        self.root.run()


class RootWidget(App):
    def build(self):
        self.Main = MainWindow()
        return self.Main

    def on_stop(self):
        self.Main.on_stop()

class WorkPanel(RelativeLayout):

    def __init__(self,uievents, **kwargs):
        super(WorkPanel, self).__init__(**kwargs)
        self.uievents = uievents

        Instructions = Label(font_size="14sp")
        Instructions.size_hint = (None, None)
        Instructions.size = (270, 200)
        Instructions.pos = (1045, 330)
        Instructions.color = (1, 1, 1, 1)
        Instructions.text = "Example text \n" \
                            "Example text\n" \
                            "\n" \
                            "Example text\n"
        self.add_widget(Instructions)

        btnDraw = Button(text="Draw")
        btnDraw.size_hint = (None, None)
        btnDraw.size = (130, 40)
        btnDraw.pos = (1040, 320)
        btnDraw.color = (1, 1, 1, 1)
        btnDraw.bind(on_release=self.uievents.DrawClick)
        self.add_widget(btnDraw)

        btnErase = Button(text="Erase")
        btnErase.size_hint = (None, None)
        btnErase.size = (130, 40)
        btnErase.pos = (1180, 320)
        btnErase.color = (1, 1, 1, 1)
        btnErase.bind(on_release=self.uievents.EraseClick)
        self.add_widget(btnErase)

        btnSaveDrawing = Button(text="Save")
        btnSaveDrawing.size_hint = (None, None)
        btnSaveDrawing.size = (130, 40)
        btnSaveDrawing.pos = (1040, 260)
        btnSaveDrawing.color = (1, 1, 1, 1)
        btnSaveDrawing.bind(on_release=self.uievents.SaveDrawingClick)
        self.add_widget(btnSaveDrawing)

        btnEraseAll = Button(text="Erase All")
        btnEraseAll.size_hint = (None, None)
        btnEraseAll.size = (130, 40)
        btnEraseAll.pos = (1180, 260)
        btnEraseAll.color = (1, 1, 1, 1)
        btnEraseAll.bind(on_release=self.uievents.EraseAllClick)
        self.add_widget(btnEraseAll)
        
        textinput=TextInput(multiline=False)
        textinput.size_hint = (None, None)
        textinput.size = (130, 40)
        textinput.pos = (1040, 170)
        #textinput.bind(text=self.uievents._on_keyboard_down)
        self.add_widget(textinput)
'''
        btnPrevSlice = Button(text="Prev. Slice (a)")
        btnPrevSlice.size_hint = (None, None)
        btnPrevSlice.size = (130, 40)
        btnPrevSlice.pos = (1040, 170)
        btnPrevSlice.color = (1, 1, 1, 1)
        btnPrevSlice.bind(on_release=self.uievents.PrevSliceClick)
        self.add_widget(btnPrevSlice)

        btnNextSlice = Button(text="Next Slice (d)")
        btnNextSlice.size_hint = (None, None)
        btnNextSlice.size = (130, 40)
        btnNextSlice.pos = (1180, 170)
        btnNextSlice.color = (1, 1, 1, 1)
        btnNextSlice.bind(on_release=self.uievents.NextSliceClick)
        self.add_widget(btnNextSlice)

        btnPrevAccession = Button(text="Prev. Access. (z)")
        btnPrevAccession.size_hint = (None, None)
        btnPrevAccession.size = (130, 40)
        btnPrevAccession.pos = (1040, 110)
        btnPrevAccession.color = (1, 1, 1, 1)
        btnPrevAccession.bind(on_release=self.uievents.PrevAccessionClick)
        self.add_widget(btnPrevAccession)

        btnNextAccession = Button(text="Next Access. (c)")
        btnNextAccession.size_hint = (None, None)
        btnNextAccession.size = (130, 40)
        btnNextAccession.pos = (1180, 110)
        btnNextAccession.color = (1, 1, 1, 1)
        btnNextAccession.bind(on_release=self.uievents.NextAccessionClick)
        self.add_widget(btnNextAccession)

        btnScanFungal = Button(text="Scan: Fungal")
        btnScanFungal.size_hint = (None, None)
        btnScanFungal.size = (110, 40)
        btnScanFungal.pos = (1050, 30)
        btnScanFungal.color = (1, 1, 1, 1)
        btnScanFungal.bind(on_release=self.uievents.ScanFungalClick)
        self.add_widget(btnScanFungal)

        btnScanNormal = Button(text="Scan: Normal")
        btnScanNormal.size_hint = (None, None)
        btnScanNormal.size = (110, 40)
        btnScanNormal.pos = (1050, -25)
        btnScanNormal.color = (1, 1, 1, 1)
        btnScanNormal.bind(on_release=self.uievents.ScanNormalClick)
        self.add_widget(btnScanNormal)

        btnSliceSpecific = Button(text="Slice: Specific")
        btnSliceSpecific.size_hint = (None, None)
        btnSliceSpecific.size = (110, 40)
        btnSliceSpecific.pos = (1190, 30)
        btnSliceSpecific.color = (1, 1, 1, 1)
        btnSliceSpecific.bind(on_release=self.uievents.SliceSpecificClick)
        self.add_widget(btnSliceSpecific)

        btnSliceNormal = Button(text="Slice: Normal")
        btnSliceNormal.size_hint = (None, None)
        btnSliceNormal.size = (110, 40)
        btnSliceNormal.pos = (1190, -25)
        btnSliceNormal.color = (1, 1, 1, 1)
        btnSliceNormal.bind(on_release=self.uievents.SliceNormalClick)
        self.add_widget(btnSliceNormal)

        btnAgreement = Button(text="Agreement")
        btnAgreement.size_hint = (None, None)
        btnAgreement.size = (100, 40)
        btnAgreement.pos = (1210, -110)
        btnAgreement.color = (1, 1, 1, 1)
        btnAgreement.bind(on_release=self.uievents.AgreementClick)
        self.add_widget(btnAgreement)

        btnDebugMode = Button(text="Debug")
        btnDebugMode.size_hint = (None, None)
        btnDebugMode.size = (60, 30)
        btnDebugMode.pos = (1250, -200)
        btnDebugMode.color = (1, 1, 1, 1)
        btnDebugMode.bind(on_release=self.uievents.DebugModeClick)
        self.add_widget(btnDebugMode)
'''

class MainWindow(RelativeLayout):

    def __init__(self,**kwargs):
        super(MainWindow, self).__init__(**kwargs)

        ImageViewer = Image(source="")
        ImageViewer.size_hint = (None, None)
        #ImageViewer.size = (1029, 512)
        ImageViewer.size = (1021, 576)
        ImageViewer.pos = (0,240)
        self.add_widget(ImageViewer)
        
        FilePath = Label(font_size="14sp", text="", color=(1,1,1,1),halign='left',text_size=(1000,100))
        FilePath.size_hint = (None, None)
        FilePath.size = (1000, 100)
        FilePath.pos = (20, 140)
        self.add_widget(FilePath)
        
        TextReport = Label(font_size="14sp", text="", color=(1, 1, 1, 1), halign='left', text_size=(1300, 150))
        TextReport.size_hint = (None, None)
        TextReport.size = (1300, 150)
        TextReport.valign = 'top'
        TextReport.text = ""
        TextReport.pos = (20, 0)
        self.add_widget(TextReport)

        StatusMessage = Label(font_size="14sp", text="", color=(1, 1, 1, 1), halign='left', text_size=(1300, 150))
        StatusMessage.size_hint = (None, None)
        StatusMessage.size = (1300, 150)
        StatusMessage.valign = 'top'
        StatusMessage.text = ""
        StatusMessage.pos = (20, -70)
        self.add_widget(StatusMessage)

        LabelReminder = Label(font_size="12sp", text="", color=(1, .3, .3, 1), halign='left', text_size=(300, 150))
        LabelReminder.size_hint = (None, None)
        LabelReminder.size = (300, 150)
        LabelReminder.valign = 'top'
        LabelReminder.text = ""
        LabelReminder.pos = (1050, 185)
        self.add_widget(LabelReminder)

        self.uievents = UIEvents(ImageViewer,FilePath,TextReport,StatusMessage,LabelReminder)

        WorkingPanel = WorkPanel(self.uievents)
        WorkingPanel.pos = (0,240)
        self.add_widget(WorkingPanel)
        
    def on_stop(self):
        self.uievents.CheckEditsMade()


RadiologistGUI()




