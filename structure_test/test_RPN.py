import os
import torch
import importlib


def file_parts(file_path):
    """
    Lists a files parts such as base_path, file name and extension

    Example
        base, name, ext = file_parts('path/to/file/dog.jpg')
        print(base, name, ext) --> ('path/to/file/', 'dog', '.jpg')
    """

    base_path, tail = os.path.split(file_path)
    name, ext = os.path.splitext(tail)

    return base_path, name, ext


def absolute_import(file_path):
    """
    Imports a python module / file given its ABSOLUTE path.

    Args:
         file_path (str): absolute path to a python file to attempt to import
    """

    # module name
    _, name, _ = file_parts(file_path)

    # load the spec and module
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


if __name__ == '__main__':
    network = absolute_import("../models/densenet121_3d_dilate_depth_aware.py")
    print(network)
