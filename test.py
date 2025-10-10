"""
Run this script when troubleshooting TC-Python

It is important to run this script EXACTLY the same way as you run your TC-Python script
(In the same IDE, same project, same Python environment, same Jupyter notebook e.t.c)

"""

version = '2025b'

print('Testing TC-Python version: ' + version)
print('Please make sure that the variable "version" above, matches the release that you want to test, if not change it and re-run this script.')

# below this line, nothing needs to be manually updated.

import sys
print('')
print('Python version (needs to be at least Python 3.8, Python 2.x is not supported):')
print(sys.version)
if sys.version_info[0] < 3 or sys.version_info[1] < 8:
    print('Wrong version of Python !!!!!')

print('')
print('Python executable path: (gives a hint about the used virtual / conda environment, in case of Anaconda the corresponding \n'
      'environment name can be found by running `conda env list` on the Anaconda command prompt, '
      'TC-Python must be installed into \nEACH separate environment used!)')
print(sys.executable)

import os
print('')
print('Thermo-Calc ' + version + ' installation directory: (must be a valid path to a complete installation of ' + version + ')')
tc_env_variable = 'TC' + version[2:].upper() + '_HOME'
try:
    print(os.environ[tc_env_variable])
except:
    print('No Thermo-calc environment variable for ' + version + ' was found. (' + tc_env_variable + ')')

import tc_python
numerical_version = version[:-1]
if version[-1] == 'a':
    numerical_version += '.1.*'
elif version[-1] == 'b':
    numerical_version += '.2.*'
print('')
print('TC-Python version: (needs to be ' + numerical_version + ')')
print(tc_python.__version__)

user_based_license_var = os.environ.get('TC_LICENSE_SPRING', None)
user_based_license = False
if user_based_license_var is not None:
    user_based_license = user_based_license_var.upper() == 'Y'

if not user_based_license:
    print('Url of license server: (if license server is NO-NET, you need a local license file)')
    try:
        print(os.environ['LSHOST'])
    except:
        print('No Thermo-calc license server url was found. (LSHOST)')

    print('')
    print('Path to local license file: (only necessary if not using license server)')
    try:
        print(os.environ['LSERVRC'])
    except:
        print('No path to local license file was found. (LSERVRC)')
else:
    print('')
    print('User/password based licenses is enabled')
    print("License Information:")
    with tc_python.TCPython() as session:
        license_manager = session.get_license_manager()
        print(license_manager.get_info())



with tc_python.TCPython() as session:
    print('')
    print('Lists the databases (should be a complete list of the installed databases that you have license for or do not require license):')
    print(session.get_databases())