"""
setup.py: Install nebterpolator.
"""
VERSION="1.0"
__author__ = "Robert McGibbon and Lee-Ping Wang"
__version__ = VERSION

from distutils.sysconfig import get_config_var
from distutils.core import setup,Extension
import os
import shutil
import numpy
import glob

# Comment left here as an example
# Copied from MSMBuilder 'contact' library for rapidly computing interatomic distances.
# CONTACT = Extension('forcebalance/_contact_wrap',
#                     sources = ["ext/contact/contact.c",
#                                "ext/contact/contact_wrap.c"],
#                     extra_compile_args=["-std=c99","-O3","-shared",
#                                         "-fopenmp", "-Wall"],
#                     extra_link_args=['-lgomp'],
#                     include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])

def buildKeywordDictionary():
    from distutils.core import Extension
    setupKeywords = {}
    setupKeywords["name"]              = "nebterpolator"
    setupKeywords["version"]           = VERSION
    setupKeywords["author"]            = __author__
    setupKeywords["author_email"]      = "leeping@stanford.edu"
    setupKeywords["license"]           = "GPL 3.0"
    setupKeywords["url"]               = "https://github.com/rmcgibbo/nebterpolator"
    setupKeywords["download_url"]      = "https://github.com/rmcgibbo/nebterpolator"
    setupKeywords["scripts"]           = glob.glob("bin/*.py") + glob.glob("bin/*.sh")
    setupKeywords["packages"]          = ["nebterpolator", "nebterpolator.io", "nebterpolator.core"]
    # setupKeywords["package_data"]      = {"nebterpolator" : ["data/*.sh","data/uffparms.in","data/oplsaa.ff/*"]}
    setupKeywords["data_files"]        = []
    setupKeywords["ext_modules"]       = []
    setupKeywords["platforms"]         = ["Linux"]
    setupKeywords["description"]       = "Internal coordinate smoothing."

    outputString=""
    firstTab     = 40
    secondTab    = 60
    for key in sorted( setupKeywords.iterkeys() ):
         value         = setupKeywords[key]
         outputString += key.rjust(firstTab) + str( value ).rjust(secondTab) + "\n"
    
    print "%s" % outputString

    return setupKeywords
    
def main():
    setupKeywords=buildKeywordDictionary()
    setup(**setupKeywords)

if __name__ == '__main__':
    main()

