from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=[ 'scene_analyser', 'MaskPredictor', 'dom_octo_mapper' ],
    package_dir={'': 'src'}
)

setup(**setup_args)
