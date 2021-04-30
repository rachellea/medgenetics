# main.py

import os
import datetime

#Custom imports
from src import visualization1_all, visualization2_all, visualization3_all

if __name__=='__main__':
    date_dir = 'C:\\Users\\Rachel\\Documents\\CarinLab\\Project_Genetics_Landstrom\\newmedgenetics\\results\\2021-02-10'
    visualization1_all.MakePanelFigure(date_dir)
    visualization2_all.MakePanelFigure_SensSpec(date_dir)
    visualization3_all.MakeFigure_MysteryViolin(date_dir)