import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

DEPENDENCIES = ['configargparse', "fenics-dolfin",
                "scipy", "numpy", "pyyaml"]
TEST_DEPENDENCIES = ['pytest']

VERSION = "2.4"
URL = "https://github.com/KVSlab/turtleFSI.git"

setuptools.setup(
    name="turtleFSI",
    version=VERSION,
    license="GPL",
    author="",
    author_email="",
    url=URL,
    project_urls={
        "Documentation": "https://turtlefsi.readthedocs.io/",
        "Source Code": URL,
    },
    description="turtleFSI - Fluid-structure interaction",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Dependencies
    install_requires=DEPENDENCIES,
    tests_require=TEST_DEPENDENCIES,

    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
    package_dir={"turtleFSI": "turtleFSI"},
    include_package_data=True,
    entry_points={'console_scripts': [
        'turtleFSI=turtleFSI.run_turtle:main',
        'turtleFSI-pod=turtleFSI.rom.pod:main',
        'turtleFSI-pod-galerkin=turtleFSI.rom.pod_galerkin:main',
    ]},

)
