import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ['numpy', 'pandas', 'sklearn', 'slackclient', 'statsmodels', 'matplotlib', 'seaborn', 'plotly']
# other stuff missing

setuptools.setup(
    name="utilsJ",
    version="0.0.1",
    author="Jordi Pastor",
    author_email="author@example.com",
    description="custom utils for common tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PastorJordi/custom_utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements
)
