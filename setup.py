import subprocess
from setuptools import setup, find_packages
from setuptools.extension import Extension


def parse_packages(packages_txt):

    with open(packages_txt, "r") as f:
        lines = f.readlines()
        f.close()

    packages_list = []
    for line in lines:
        packages_list.append(str(line.strip()))

    return packages_list



if __name__ == "__main__":
    

    setup(
        name='multitts',
        version='0.1.0',
        description='A Text To Speech package for multi-ligual applications',
        author='Hassan Teymouri',
        author_email='hassan.teymoury@gmail.com',
        url="https://github.com/hassan-teymoury/MultiLangTTS.git",
        license='Apache License 2.0',
        packages=["multitts.models", "multitts.vits",
            "multitts.vits.text", "multitts.vits.monotonic_align",
            "multitts.vits.configs", "multitts.vits.filelists"],
        include_package_data=True,
        install_requires=parse_packages("requirements.txt"),
        ext_modules=[Extension(
                'multitts.vits.monotonic_align.core',
                library_dirs=['/usr/local/bin'],
                sources=['multitts/vits/monotonic_align/core.c'])],    
        
    )
