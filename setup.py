import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="splcurve",
    version="0.0.1",
    author="Jan Niklas Franke",
    author_email="jan.franke@tu-drotmund.de",
    description="B-Spline Curves in 2D",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NklasF/splcurve",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
