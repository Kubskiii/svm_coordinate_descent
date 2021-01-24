import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.readlines()

setuptools.setup(
    name="scikit-learn-coordinate-descent-svc",
    version="0.0.1",
    author="Jakub Waszkiewicz, Michał Urawski",
    author_email="waszkiewiczj.dev@outlook.com",
    description="Coordinate Descent SVC implementation integrated with Scikit Learn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kubskiii/svm_coordinate_descent",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
)