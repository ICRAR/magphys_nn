# Magphys neural network preprocessor
# Date: 7/12/2015
# Author: Samuel Foster
# Organisation: ICRAR
#
# For getting data from pogs_analysis
#

import os, sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

from common.pogs_analysis_tables import *
from sqlalchemy.engine import create_engine
from sqlalchemy import func
from sqlalchemy.sql.expression import func as ffunc

pogs_connection = create_engine('mysql://pogs:pogs@munro.icrar.org/pogs_analysis').connect()


def import_from_pogs():

    # result = select galaxy_id, redshift from galaxy where run_id = 1
    input_runs = {}
    # for row in result:
    #   input_result = select filter, x, y, value, sigma from original_value where galaxy_id = result.galaxy_id order by x asc, y asc, filter_number asc;
    #       for row in input_result:
    #           id = '{0}_{1}_{2}'.format(galaxy_id, x, y'
    #           if not id in runs:
    #               runs[id] = [] # There have been no runs added for this yet.
    #
    #
    #           runs[id].append( (filter_number, value, sigma) )
    #



