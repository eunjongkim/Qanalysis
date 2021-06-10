import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Qanalysis-eunjongkim", # Replace with your own username
    version="0.1.2",
    author="Eunjong Kim",
    author_email="ekim7206@gmail.com",
    description="Qubit Measurement Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eunjongkim/Qanalysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)