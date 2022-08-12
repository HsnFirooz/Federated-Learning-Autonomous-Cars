import os
import sys
import argparse
from os import listdir
from os.path import isfile, join
import subprocess

def folder():
    ap = argparse.ArgumentParser()
    ap.add_argument("-fi", "--fi", default="folder_input", help="folder input")
    # ap.add_argument("-fo", "--fo", default="folder_output", help="folder output")
    args = vars(ap.parse_args())

    inputs = os.path.join(args["fi"])
    # output = os.path.join(args["fo"])

    for dirpath,_,filenames in os.walk(inputs):
        for f in filenames:
            # print(os.path.abspath(os.path.join(dirpath, f)))
            subprocess.call("python3 ts_real_time.py -f "+os.path.abspath(os.path.join(dirpath, f)), shell=True)

    # for image in input_files:


folder()