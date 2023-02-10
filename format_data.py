
import os
import glob
import argparse
import shutil

def list_dir(root: str, prefix: bool = False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories

def list_files(root: str, suffix = None, prefix: bool = False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = [p for p in os.listdir(root) if os.path.isfile(os.path.join(root, p)) and (suffix is None or p.endswith(suffix))]
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files

parser = argparse.ArgumentParser()
parser.add_argument('root_dir')
parser.add_argument('target_dir')
parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'])

args = parser.parse_args()


if not os.path.exists(args.target_dir):
    os.makedirs(args.target_dir)

for split in args.splits:
    split_target=os.path.join(args.target_dir, split)
    split_source=os.path.join(args.root_dir, split)
    if not os.path.exists(split_target):
        os.makedirs(split_target)
    
    classes = list_dir(split_source)
    for classname in classes:
        classdir = os.path.join(split_source, classname)
        imgfiles = list_files(classdir)
        for imgfile in imgfiles:
            #tgt_name = classname + "."
            tgtname = imgfile   #change if needed
            shutil.copy(os.path.join(classdir, imgfile), os.path.join(split_target, tgtname))




