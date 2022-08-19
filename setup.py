#!/usr/bin/env python
"""
setup
"""

import os
import shlex
import shutil
import stat
import subprocess

from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info

version = '0.1.0'
cur_dir = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(cur_dir, 'build')


def clean():
    # pylint: disable=unused-argument
    def readonly_handler(func, path, execinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if os.path.exists(os.path.join(cur_dir, 'build')):
        shutil.rmtree(os.path.join(cur_dir, 'build'), onerror=readonly_handler)
    if os.path.exists(os.path.join(cur_dir, 'mindvision.egg-info')):
        shutil.rmtree(os.path.join(cur_dir, 'mindvision.egg-info'), onerror=readonly_handler)


def write_version(file):
    file.write("__version__ = '{}'\n".format(version))


def build_depends():
    """generate python file"""
    version_file = os.path.join(cur_dir, 'mindvision/', 'version.py')
    with open(version_file, 'w') as f:
        write_version(f)


clean()
build_depends()


def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD)


def get_description():
    """
    Get description.

    Returns:
        str, wheel package description.
    """
    cmd = "git log --format='[sha1]:%h, [branch]:%d' -1"
    process = subprocess.Popen(
        shlex.split(cmd),
        shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, _ = process.communicate()
    if not process.returncode:
        git_version = stdout.decode().strip()
        return "An open source computer vision research tool box. Git version: %s" % (git_version)
    return "An open source computer vision research tool box."


class EggInfo(egg_info):
    """Egg info."""

    def run(self):
        super().run()
        egg_info_dir = os.path.join(cur_dir, 'mindvision.egg-info')
        update_permissions(egg_info_dir)


class BuildPy(build_py):
    """BuildPy."""

    def run(self):
        super().run()
        mindarmour_dir = os.path.join(pkg_dir, 'lib', 'mindvision')
        update_permissions(mindarmour_dir)


setup(
    name="mindvision",
    version=version,
    author="MindVision Core Team",
    url="https://gitee.com/mindspore/vision/tree/master/",
    project_urls={
        'Sources': 'https://gitee.com/mindspore/vision',
        'Issue Tracker': 'https://gitee.com/mindspore/vision/issues',
    },
    description=get_description(),
    license='Apache 2.0',
    packages=find_packages(exclude=("example")),
    include_package_data=True,
    cmdclass={
        'egg_info': EggInfo,
        'build_py': BuildPy,
    },
    install_requires=[
        'scipy >= 1.5.2',
        'numpy >= 1.17.0',
        'matplotlib >= 3.2.1',
        'pillow >= 6.2.0',
        'pytest >= 4.3.1',
        'wheel >= 0.32.0',
        'setuptools >= 40.8.0',
        'scikit-learn >= 0.23.1',
        'easydict >= 1.9',
        'ml_collections',
        'opencv-python-headless',
        'opencv-contrib-python-headless',
        'tqdm'
    ],
    classifiers=[
        'License :: OSI Approved :: Apache Software License'
    ]
)
print(find_packages())
