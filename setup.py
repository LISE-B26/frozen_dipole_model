from setuptools import setup, find_packages
from frozen_dipole_model import __version__ as current_version

# ========================================================================================================
# ========================================================================================================
# NOTES for updating this file:
# 1) for version update in the frozen_dipole_model.__init__
# 2) update the following comment_on_changes
comment_on_changes = 'initial commit'
# ========================================================================================================
# ========================================================================================================


setup(name='frozen_dipole_model',
      version=current_version,
      packages=find_packages(),
      description='Levitated Dipole modes and equilibrium positions according to frozen dipole model',
      url='https://github.com/JanGieseler/frozen_dipole_model',
      author='Jan Gieseler',
      author_email='jangie@pm.me',
      license='MIT',
      zip_safe=False,
      keywords = 'magnetic levitation, levitodynamics, nanolevitation, frozen dipole',
      long_description = comment_on_changes,
      classifiers = [
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: 2-Clause BSD License',
            'Development Status :: 4 - Beta',
            'Environment :: MacOS (Ubuntu)',
      ]
      )