# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 23:05:46 2020

@author: AsakoKagurazaka
"""
#......IMPORT .........
import argparse
def task1():
    print("draft code")

def task2():
    print("draft code")

def task3():
    print("draft code")

my_parser = argparse.ArgumentParser()
my_parser.add_argument('-o','--OP_folder', type=str,help='Output folder name', default = 'OUTPUT')
my_parser.add_argument('-m','--min_area', type=int,action='store', required = True, help='Minimum pixel area to be occupied, to be considered a whole rice kernel')
my_parser.add_argument('-f','--input_filename', type=str,action='store', required = True, help='Filename of image ')
# Execute parse_args()
args = my_parser.parse_args()

import cv2