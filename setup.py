import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

DEPENDENCIES = ['configargparse', "fenics-dolfin", "numpy",
                "scipy", "cppimport"]
TEST_DEPENDENCIES = ['pytest']

VERSION = "1.5"
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
    packages=["turtleFSI",
              "turtleFSI.modules",
              "turtleFSI.problems",
              "turtleFSI.utils"],
    package_dir={"turtleFSI": "turtleFSI"},
    package_data={"turtleFSI": ["utils/probe/*.h",
                                "utils/probe/*.cpp"]},
    include_package_data=True,
    entry_points={'console_scripts': ['turtleFSI=turtleFSI.run_turtle:main']},

)
