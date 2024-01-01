import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd

import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm
import train as tr
import resnettraining as tr1
import densenetraining as tr2

import test as pre



bgcolor="#032174"
bgcolor1="#0492C2"
fgcolor="#FFFFFF"
accuracy=[]


def Home():
        global window
        def clear():
            print("Clear1")
            txt.delete(0,'end')
            txt1.delete(0, 'end')    



        window = tk.Tk()
        window.title("Hybrid Cucumber Leaf Disease Detection Using Machine Learning")

 
        window.geometry('1280x720')
        window.configure(background=bgcolor)
        #window.attributes('-fullscreen', True)

        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        

        message1 = tk.Label(window, text="Hybrid Cucumber Leaf Disease Detection" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, )) 
        message1.place(x=20, y=20)

        lbl = tk.Label(window, text="Select Dataset Folder",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl.place(x=100, y=200)
        
        txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt.place(x=400, y=215)

        lbl1 = tk.Label(window, text="Select Image",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1.place(x=100, y=300)
        
        txt1 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1.place(x=400, y=315)

        def browse():
                path=filedialog.askdirectory()
                print(path)
                txt.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Train Dataset")      

        def browse1():
                path=filedialog.askopenfilename()
                print(path)
                txt1.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Train Dataset")
        def Trainprocess():
                sym=txt.get()
                if sym!="":
                        macc=tr.process(sym)
                        accuracy.append(macc)
                        tm.showinfo("Output", "Training finished successfully")
                else:
                        tm.showinfo("Input error", "Select Train Dataset")
        def Trainprocess1():
                sym=txt.get()
                if sym!="":
                        racc=tr1.process(sym)
                        accuracy.append(racc)
                        tm.showinfo("Output", "Training finished successfully")
                else:
                        tm.showinfo("Input error", "Select Train Dataset")
        def Trainprocess2():
                sym=txt.get()
                if sym!="":
                        dacc=tr2.process(sym)
                        accuracy.append(dacc)
                        tm.showinfo("Output", "Training finished successfully")
                else:
                        tm.showinfo("Input error", "Select Train Dataset")
        def Predictprocess():
                sym=txt1.get()
                if sym!="":
                        res,conf=pre.process(sym)
                        tm.showinfo("Output", "Predicted as "+str(res)+" With Confidance of "+str(conf))

                else:
                        tm.showinfo("Input error", "Select Input Image")


        
        

        browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=10  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse.place(x=650, y=200)

        browse1 = tk.Button(window, text="Browse", command=browse1  ,fg=fgcolor  ,bg=bgcolor1  ,width=10  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse1.place(x=650, y=300)

        clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=10  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
        clearButton.place(x=850, y=200)
         
        RFbutton = tk.Button(window, text="MobileNET", command=Trainprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        RFbutton.place(x=80, y=450)
        RFbutton1 = tk.Button(window, text="RESNET", command=Trainprocess1  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        RFbutton1.place(x=280, y=450)
        RFbutton2 = tk.Button(window, text="DENSENET", command=Trainprocess2  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        RFbutton2.place(x=480, y=450)

        DCbutton = tk.Button(window, text="PREDICT", command=Predictprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        DCbutton.place(x=680, y=450)

        quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow.place(x=880, y=450)

        window.mainloop()
Home()

