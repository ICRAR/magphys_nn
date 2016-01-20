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
from common.database import db_init
from sqlalchemy.engine import create_engine
from sqlalchemy import select, exists
from sqlalchemy import func
from sqlalchemy.sql.expression import func as ffunc

pogs_connection = create_engine('mysql://pogs:pogs@munro.icrar.org/pogs_analysis').connect()
nn_connection = db_init('sqlite:///Database_run01.db')

filter_map = {123: 'fuv',
              124: 'nuv',
              229: 'u',
              230: 'g',
              323: 'g',
              324: 'r',
              231: 'r',
              325: 'i',
              232: 'i',
              326: 'z',
              233: 'z',
              327: 'y',
              280: 'WISEW1',
              281: 'WISEW2',
              282: 'WISEW3',
              283: 'WISEW4',
              172: 'SPIRE250',
              173: 'SPIRE350',
              174: 'SPIRE500',
              115: 'u',
              116: 'g',
              117: 'r',
              118: 'i',
              119: 'z'}


def import_from_pogs():

    train_ins = {}
    train_outs = {}
    galaxy_table = pogs_connection.execute(select([GALAXY.c.galaxy_id, GALAXY.c.redshift]).where(GALAXY.c.run_id == 1))
    for galaxy in galaxy_table:

        print 'Galaxy {0}, redshift {1}'.format(galaxy['galaxy_id'], galaxy['redshift'])

        input_result = pogs_connection.execute(select(
            [ORIGINAL_VALUE])
             .where(ORIGINAL_VALUE.c.galaxy_id == galaxy['galaxy_id'])
             .order_by(ORIGINAL_VALUE.c.x)
             .order_by(ORIGINAL_VALUE.c.y)
             .order_by(ORIGINAL_VALUE.c.filter_number))

        i = 50
        for row in input_result:
            i -= 1

            if i == 0:
                for k, v in train_ins.iteritems():
                    print '{0}      {1}'.format(k, v)
                return

            print ' x:{0} y:{1} filter:{2} value:{3} sigma{4}'.format(row['x'], row['y'], row['filter_number'], row['value'], row['sigma'])

            run_key = '{0}_{1}_{2}'.format(galaxy['galaxy_id'], row['x'], row['y'])
            if run_key in train_ins:
                train_ins[run_key].append(row['value'])
                train_ins[run_key].append(row['sigma'])
            else:
                train_ins[run_key] = []
                train_ins[run_key].append(galaxy['redshift'])
                train_ins[run_key].append(row['value'])
                train_ins[run_key].append(row['sigma'])


            """
            for row in input_result:
                id = '{0}_{1}_{2}'.format(galaxy_id, x, y'
                if not id in runs:
                    runs[id] = [] # There have been no runs added for this yet.
                runs[id].append( (filter_number, value, sigma) )
            """

if __name__ == '__main__':
    import_from_pogs()