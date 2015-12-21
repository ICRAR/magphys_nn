# Magphys neural network preprocessor config file
# Date: 7/12/2015
# Author: Samuel Foster
# Organisation: ICRAR
#
# Configuration variables for the preprocessor
# Anything prefixed with a C_ comes from the config file.
#

# Set to True to override database entries when loading in new files.
# Set to False to skip files if they already exist in the database.
RELOAD = False

# Set to True to shuffle the directory listing whenever input files are searched for.
# Set to False to not shuffle.
SHUFFLE = False

# Set to True to allow NaN values from input files to be loaded in to the database (as nulls).
# Set to False to ignore files with NaN values.
ALLOW_NAN = False

# The login string for the database. Must be a valid SQL Alchemy database string!
DB_LOGIN = 'sqlite:///./preprocess/Database.db'
