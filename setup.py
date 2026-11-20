from setuptools import setup, find_packages

setup(
    name="micra-cmb",
    version="1.0.0",
    description=(
        "MICRA: Spacing-Conditioned and Physically Consistent Slice-Persistence "
        "Learning for Cerebral Microbleed Segmentation and Detection Across "
        "Heterogeneous T2*-Weighted MRI"
    ),
    author="T. Grace Berin, C. Helen Sulochana, R. Rajesh",
    author_email="graceberin8@gmail.com",
    url="https://github.com/graceberin8-commits/micra-cmb.git",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.2",
        "torchvision>=0.16.2",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "nibabel>=5.1.0",
        "matplotlib>=3.7.0",
    ],
)
