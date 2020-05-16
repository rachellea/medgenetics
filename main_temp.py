# main.py

import os
import datetime

#Custom imports
from src import visualization2_all, visualization3_all

if __name__=='__main__':
    date_dir = 'C:\\Users\\Rachel\\Documents\\CarinLab\\Project_Genetics_Landstrom\\2020\\Final_PACE_Results\\MergedCopiesFinal'
    #visualization2_all.MakePanelFigure_SensSpec(date_dir)
    visualization3_all.MakeFigure_MysteryViolin(date_dir)