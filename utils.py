from tabulate import tabulate_formats, tabulate
import os
import re

def pdf(df):
    print(tabulate(df, headers='keys', tablefmt=tabulate_formats[2]))


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory {dir} created.')
    # else:
    #     print(f'Directory {dir} already exists.')

def strip_suffix(name):
    return re.sub(r'\s+(Jr\.|Sr\.|I{1,3})$', '', name)
