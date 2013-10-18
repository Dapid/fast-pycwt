'''
Created on 9 Oct 2013

@author: david
'''
import os
import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

module = '_cwt'

setup(cmdclass={'build_ext': build_ext},
      name=module,
      ext_modules=[Extension(module,
                             [module + ".pyx"])],
      include_dirs=[numpy.get_include(),
                    os.path.join(numpy.get_include(), 'numpy')]
      )
