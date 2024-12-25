from tabulate import tabulate_formats, tabulate
import os


def pdf(df):
    print(tabulate(df, headers='keys', tablefmt=tabulate_formats[2]))


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory {dir} created.')
    # else:
    #     print(f'Directory {dir} already exists.')