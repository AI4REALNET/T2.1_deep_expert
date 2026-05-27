import os
import setuptools
from setuptools import setup

__version__ = '0.1.0'

pkgs = {
    "required": [
        "cvxpy",
        "grid2op>=1.10.5",
        "l2rpn_baselines==0.8.0",
        "lightsim2grid>=0.10.3",
        "numpy>=1.24.3",
        "torch>=2.12.0",
        "tensorflow>=2.12.1",
        "stable-baselines3>=2.8.0",
        "imageio>=2.37.3",
        "numba>=0.65.1",
        "lxml",
    ],
}

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='ExpertAgent',
      version=__version__,
      description='ExpertAgent to solve power grid congestion and overload',
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.10',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English"
      ],
      keywords='AI4REALNET project',
      author='IRTSX',
      url="https://github.com/AI4REALNET/T2.1_deep_expert",
      license='MPL',
      packages=setuptools.find_packages(),
      include_package_data=True,
      package_data={
            # If any package contains *.txt or *.rst files, include them:
            "": ["*.ini", "*.zip", "*.npz"],
            },
      install_requires=pkgs["required"],
    #   extras_require=pkgs["extras"],
      zip_safe=False,
      entry_points={
          'console_scripts': []
     }
)