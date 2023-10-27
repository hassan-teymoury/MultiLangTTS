import subprocess
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy


def parse_packages(packages_txt):

    with open(packages_txt, "r") as f:
        lines = f.readlines()
        f.close()

    packages_list = []
    for line in lines:
        packages_list.append(str(line.strip()))

    return packages_list


# cmd = "pip install -r requirements.txt"
# subprocess.check_output(cmd, shell=True)


# cmd = ";".join([
#       "cd vits/",
#       "cd monotonic_align/",
#       "python3 setup.py build_ext --inplace",
#       "cd ..",
# ])

# subprocess.check_output(cmd, shell=True)


setup(
    name='multitts',
    version='0.1.0',
    description='A Text To Speech package for multi-ligual applications',
    author='Hassan Teymouri',
    author_email='hassan.teymoury@gmail.com',
    url="",
    license='Apache License 2.0',
    packages=["multitts"],
    include_package_data=True,
    install_requires=parse_packages("requirements.txt"),
    ext_modules=[Extension(
            'multitts.vits.monotonic_align.core',
            library_dirs=['/usr/local/bin'],
            sources=['multitts/vits/monotonic_align/core.c'])],    
    
)


