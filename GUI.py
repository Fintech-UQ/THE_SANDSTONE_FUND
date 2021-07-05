import tkinter as tk
from tkinter import filedialog, Text
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import os
import numpy as np

root = tk.Tk()
root.geometry("700x700")
fig = Figure(figsize=(5, 4), dpi=100)
canvas = tk.Canvas(root, height=700, width=700, bg="#171628")

frame = tk.Frame(root, bg="#171628")
frame.place(relwidth=1, relheight=1)

figure_canvas = FigureCanvasTkAgg(fig, master=frame)
figure_canvas.draw()
figure_canvas.get_tk_widget().pack(side='top', fill='x', pady=(0, 10))

run = tk.Button(frame, text="Run Model", relief='flat', fg="white", bg="#0a09a5")
run.pack()

data_history_label = tk.Label(frame, text="Historic Data Period", fg="white", bg="#171628")
data_history_label.place(x=20, y=450, anchor="w")

data_history = tk.Entry(frame, width=10)
data_history.place(x=150, y=450, anchor="w")

holding_period_label = tk.Label(frame, text="Holding Period", fg="white", bg="#171628")
holding_period_label.place(x=20, y=500, anchor="w")

holding_period = tk.Entry(frame, width=10)
holding_period.place(x=150, y=500, anchor="w")

waiting_period_label = tk.Label(frame, text="Waiting Period", fg="white", bg="#171628")
waiting_period_label.place(x=20, y=550, anchor="w")

waiting_period = tk.Entry(frame, width=10)
waiting_period.place(x=150, y=550, anchor="w")

ranking_system_label = tk.Label(frame, text="Ranking System", fg="white", bg="#171628")
ranking_system_label.place(x=20, y=600, anchor="w")

ranking_system = tk.Entry(frame, width=10)
ranking_system.place(x=150, y=600, anchor="w")

root.resizable(False, False)
root.mainloop()
