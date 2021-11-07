
import subprocess
import os
from sys import stdout

build_cython = True
build_wheel  = True
test = True

indexmapping_version = "IndexMapping-1.0.2*"
py_versions = ['%python3_6%', '%python3_7%', '%python3_8%', '%python3_9%', '%python3_10%']

command1 = " setup_mapping.py build_ext --inplace --force"
command2 = " setup.py sdist bdist_wheel"
if test:
    command3 = "C:\\Users\\yoann\\AppData\\Roaming\\Python\\Python36\\Scripts\\twine upload " +\
        " --verbose --repository testpypi dist/IndexMapping-" + indexmapping_version
else:
    command3 = "C:\\Users\\yoann\\AppData\\Roaming\\Python\\Python36\\Scripts\\twine upload " + \
               " --verbose dist/IndexMapping-" + indexmapping_version


def runcmd(cmd):
    x = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    return x.communicate(stdout)


if build_cython:
    # BUILD CYTHON CODE
    for ver in py_versions:
        print("\n")
        os.system('echo %s ' % ver)
        os.system('%s %s' % (ver, command1))

if build_wheel:
    for ver in py_versions:
        print("\n")
        os.system('echo %s ' % ver)
        os.system('%s %s' % (ver, command2))
