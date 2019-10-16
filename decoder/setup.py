from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

try:
    import numpy
except ImportError:
    numpy = None

class NumpyIncludePath(object):
    """Lazy import of numpy to get include path."""
    @staticmethod
    def __str__():
        import numpy
        return numpy.get_include()


if cythonize is not None and numpy is not None:
    EXTENSIONS = cythonize([Extension('functional',
                                      ['functional.pyx'],
                                      include_dirs=[numpy.get_include()]),
                           ],
                           annotate=True,
                           compiler_directives={'language_level': 3})
else:
    EXTENSIONS = [Extension('functional',
                            ['functional.pyx'],
                            include_dirs=[NumpyIncludePath()])]

setup(name='functional',
      ext_modules=EXTENSIONS)