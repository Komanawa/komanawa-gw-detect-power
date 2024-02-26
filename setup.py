"""
created matt_dumont 
on: 24/03/22
"""
import os
from setuptools import setup, find_namespace_packages

BUILD_ID = os.environ.get("BUILD_BUILDID", "0")

setup(
    name="komanawa-gw_detect_power",
    version="v2.0.0",
    # Author details
    author="Matt Dumont",
    author_email="hansonmcoombs@gmail.com",
    packages=find_namespace_packages(where='src/', include=['komanawa.gw_detect_power']),
    package_dir={"": "src"},
    setup_requires=[],
    tests_require=[],
    extras_require={},
    install_requires=[
        "pandas>=2.0.3",
        "numpy>=1.25.2",
        "scipy>=1.11.2",
        "tables>=3.8.0",
        'psutil>=5.9.5',
    ],
    python_requires=">=3.11"
)
