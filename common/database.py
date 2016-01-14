# Magphys neural network preprocessor
# Date: 7/12/2015
# Author: Samuel Foster
# Organisation: ICRAR
#
# SQLAlchemy related functions for the preprocessor
#

import os, sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

from sqlalchemy import func
from sqlalchemy.sql.expression import func as ffunc

from database_definition import *
from logger import config_logger
from database_output_formatting import *
import os

LOG = config_logger(__name__)

connection = None


def db_init(db_string):
    """
    Initialises the database to the given string.
    Should be an SQLite database, but others will work too.
    """
    engine = create_engine(db_string)

    global connection
    connection = engine.connect()


def add_process_data_to_db(galaxy, run_id, sh_filename):

    transaction = connection.begin()

    try:
        connection.execute(NN_TRAIN.update().where(NN_TRAIN.c.galaxy_id == galaxy['galaxy_number'])
                                            .values(run_id=run_id,
                                                    redshift=galaxy['redshift'],
                                                    galaxy_id=galaxy['galaxy_number'],
                                                    run_filename=sh_filename
                                                    ))
    except:
        connection.execute(NN_TRAIN.insert().values(run_id=run_id,
                                                    redshift=galaxy['redshift'],
                                                    galaxy_id=galaxy['galaxy_number'],
                                                    run_filename=sh_filename
                                                    ))

    connection.execute(INPUT_JY.insert().values(galaxy_id=galaxy['galaxy_number'],
                       fuv=galaxy['fuv'],
                       nuv=galaxy['nuv'],
                       u=galaxy['u'],
                       g=galaxy['g'],
                       r=galaxy['r'],
                       i=galaxy['i'],
                       z=galaxy['z'],
                       Z_=galaxy['Z'],
                       Y=galaxy['Y'],
                       J=galaxy['J'],
                       H=galaxy['H'],
                       K=galaxy['K'],
                       WISEW1=galaxy['WISEW1'],
                       WISEW2=galaxy['WISEW2'],
                       WISEW3=galaxy['WISEW3'],
                       WISEW4=galaxy['WISEW4'],
                       PACS100=galaxy['PACS100'],
                       PACS160=galaxy['PACS160'],
                       SPIRE250=galaxy['SPIRE250'],
                       SPIRE350=galaxy['SPIRE350'],
                       SPIRE500=galaxy['SPIRE500'],
                       fuv_snr=galaxy['fuv_snr'],
                       nuv_snr=galaxy['nuv_snr'],
                       u_snr=galaxy['u_snr'],
                       g_snr=galaxy['g_snr'],
                       r_snr=galaxy['r_snr'],
                       i_snr=galaxy['i_snr'],
                       z_snr=galaxy['z_snr'],
                       Z__snr=galaxy['Z_snr'],
                       Y_snr=galaxy['Y_snr'],
                       J_snr=galaxy['J_snr'],
                       H_snr=galaxy['H_snr'],
                       K_snr=galaxy['K_snr'],
                       WISEW1_snr=galaxy['WISEW1_snr'],
                       WISEW2_snr=galaxy['WISEW2_snr'],
                       WISEW3_snr=galaxy['WISEW3_snr'],
                       WISEW4_snr=galaxy['WISEW4_snr'],
                       PACS100_snr=galaxy['PACS100_snr'],
                       PACS160_snr=galaxy['PACS160_snr'],
                       SPIRE250_snr=galaxy['SPIRE250_snr'],
                       SPIRE350_snr=galaxy['SPIRE350_snr'],
                       SPIRE500_snr=galaxy['SPIRE500_snr'],
                       ))
    transaction.commit()


def add_to_db(input, input_snr, output, best_fit_output, best_fit_model, best_fit_output_input, details):

    transaction = connection.begin()

    head, tail = os.path.split(details['filename'])
    gal_id = int(tail.split('.fit')[0])

    # If we read the .sh file first, then we can update.
    try:
        connection.execute(NN_TRAIN.update().where(NN_TRAIN.c.galaxy_id == gal_id)
                           .values(fit_filename=details['filename'])
                           )
    except: # If not, then we need to insert.
        connection.execute(NN_TRAIN.insert().values(fit_filename=details['filename'],
                                                    galaxy_id=gal_id)
                           )

    connection.execute(INPUT.insert().values(galaxy_id=gal_id,
                       fuv=input['fuv'],
                       nuv=input['nuv'],
                       u=input['u'],
                       g=input['g'],
                       r=input['r'],
                       i=input['i'],
                       z=input['z'],
                       Z_=input['Z'],
                       Y=input['Y'],
                       J=input['J'],
                       H=input['H'],
                       K=input['K'],
                       WISEW1=input['WISEW1'],
                       WISEW2=input['WISEW2'],
                       WISEW3=input['WISEW3'],
                       WISEW4=input['WISEW4'],
                       PACS100=input['PACS100'],
                       PACS160=input['PACS160'],
                       SPIRE250=input['SPIRE250'],
                       SPIRE350=input['SPIRE350'],
                       SPIRE500=input['SPIRE500'],
                       fuv_snr=input_snr['fuv'],
                       nuv_snr=input_snr['nuv'],
                       u_snr=input_snr['u'],
                       g_snr=input_snr['g'],
                       r_snr=input_snr['r'],
                       i_snr=input_snr['i'],
                       z_snr=input_snr['z'],
                       Z__snr=input_snr['Z'],
                       Y_snr=input_snr['Y'],
                       J_snr=input_snr['J'],
                       H_snr=input_snr['H'],
                       K_snr=input_snr['K'],
                       WISEW1_snr=input_snr['WISEW1'],
                       WISEW2_snr=input_snr['WISEW2'],
                       WISEW3_snr=input_snr['WISEW3'],
                       WISEW4_snr=input_snr['WISEW4'],
                       PACS100_snr=input_snr['PACS100'],
                       PACS160_snr=input_snr['PACS160'],
                       SPIRE250_snr=input_snr['SPIRE250'],
                       SPIRE350_snr=input_snr['SPIRE350'],
                       SPIRE500_snr=input_snr['SPIRE500']
                       ))

    connection.execute(MEDIAN_OUTPUT.insert().values(galaxy_id=gal_id,
                       xi_Wtot=output['xi_W^tot'],
                       ager=output['ager'],
                       tau_V=output['tau_V'],
                       agem=output['agem'],
                       tlastb=output['tlastb'],
                       Mstars=output['M(stars)'],
                       sfr29=output['sfr29'],
                       xi_PAHtot=output['xi_PAH^tot'],
                       f_muSFH=output['f_mu (SFH)'],
                       fb17=output['fb17'],
                       fb16=output['fb16'],
                       T_CISM=output['T_C^ISM'],
                       Ldust=output['Ldust'],
                       mu_parameter=output['mu parameter'],
                       xi_Ctot=output['xi_C^tot'],
                       f_muIR=output['f_mu (IR)'],
                       fb18=output['fb18'],
                       fb19=output['fb19'],
                       T_WBC=output['T_W^BC'],
                       SFR_0_1Gyr=output['SFR_0.1Gyr'],
                       fb29=output['fb29'],
                       sfr17=output['sfr17'],
                       sfr16=output['sfr16'],
                       sfr19=output['sfr19'],
                       sfr18=output['sfr18'],
                       tau_VISM=output['tau_V^ISM'],
                       sSFR_0_1Gyr=output['sSFR_0.1Gyr'],
                       metalicity_Z_Z0=output['metalicity Z/Zo'],
                       Mdust=output['M(dust)'],
                       xi_MIRtot=output['xi_MIR^tot'],
                       tform=output['tform'],
                       gamma=output['gamma']
                       ))

    connection.execute(BEST_FIT_OUTPUT.insert().values(galaxy_id=gal_id,
                       fmuSFH=best_fit_output['fmu(SFH)'],
                       fmuIR=best_fit_output['fmu(IR)'],
                       mu=best_fit_output['mu'],
                       tauv=best_fit_output['tauv'],
                       sSFR=best_fit_output['sSFR'],
                       m_asterisk=best_fit_output['M*'],
                       Ldust=best_fit_output['Ldust'],
                       T_WBC=best_fit_output['T_W^BC'],
                       T_CISM=best_fit_output['T_C^ISM'],
                       xi_Ctot=best_fit_output['xi_C^tot'],
                       xi_PAHtot=best_fit_output['xi_PAH^tot'],
                       xi_MIRtot=best_fit_output['xi_MIR^tot'],
                       xi_Wtot=best_fit_output['xi_MIR^tot'],
                       tvism=best_fit_output['tvism'],
                       Mdust=best_fit_output['Mdust'],
                       SFR=best_fit_output['SFR'],
                       ))

    connection.execute(BEST_FIT_HISTOGRAM.insert().values(galaxy_id=gal_id,
                                        fuv=best_fit_output_input['fuv'],
                                        nuv=best_fit_output_input['nuv'],
                                        u=best_fit_output_input['u'],
                                        g=best_fit_output_input['g'],
                                        r=best_fit_output_input['r'],
                                        i=best_fit_output_input['i'],
                                        z=best_fit_output_input['z'],
                                        Z_=best_fit_output_input['Z'],
                                        Y=best_fit_output_input['Y'],
                                        J=best_fit_output_input['J'],
                                        H=best_fit_output_input['H'],
                                        K=best_fit_output_input['K'],
                                        WISEW1=best_fit_output_input['WISEW1'],
                                        WISEW2=best_fit_output_input['WISEW2'],
                                        WISEW3=best_fit_output_input['WISEW3'],
                                        WISEW4=best_fit_output_input['WISEW4'],
                                        PACS100=best_fit_output_input['PACS100'],
                                        PACS160=best_fit_output_input['PACS160'],
                                        SPIRE250=best_fit_output_input['SPIRE250'],
                                        SPIRE350=best_fit_output_input['SPIRE350'],
                                        SPIRE500=best_fit_output_input['SPIRE500']
                                        ))

    connection.execute(BEST_FIT_MODEL.insert().values(galaxy_id=gal_id,
                                        i_sfh=best_fit_model['i_sfh'],
                                        i_ir=best_fit_model['i_ir'],
                                        chi2=best_fit_model['chi2'],
                                        redshift=best_fit_model['redshift']
                                        ))

    transaction.commit()


def exists_in_db(filename):
    count = connection.execute(select([func.count(NN_TRAIN)]).where(NN_TRAIN.c.filename == filename)).first()[0]
    if count == 0:
        return False

    return True


def row2dict(row):
    d = {}
    for column in row:
        d[column.name] = float(getattr(row, column.name))

    return d


def check_valid_row(row):
    """
    Ensures an input row from the database is valid
    :param row:
    :return:
    """
    for v in row:
        if v == -999 or v == 0:
            return False

    return True


def get_train_test_data(num_test, num_train,
                        input_type='normal', output_type='median',
                        include_sigma=True,
                        repeat_redshift=1,
                        input_filter_types=None,
                        unknown_input_handler=None):

    print 'Getting from DB'
    total_to_get = num_train+num_test
    count = connection.execute(select([func.count(NN_TRAIN)]).where(NN_TRAIN.c.fit_filename != None)
                               .order_by(ffunc.random()).limit(total_to_get + total_to_get*0.05)).first()[0]

    print '{0} entries available'.format(count)
    if count < total_to_get:
        total_to_get = count
        print 'Could not find {0}, using {1} instead'.format(total_to_get, count)

    print 'Getting indexes for {0} entries'.format(count)
    result = connection.execute(select([NN_TRAIN]).where(NN_TRAIN.c.fit_filename != None)
                                .order_by(ffunc.random())).fetchall()

    # Train and test sets are now separated, need to get the actual data now.

    all_in = []
    all_out = []
    galaxy_ids = []
    redshifts = []

    added_count = 0

    print 'Getting input and output data for {0}'.format(output_type)
    for row in result:

        if output_type == 'median':
            output_row = connection.execute(select([MEDIAN_OUTPUT]).where(MEDIAN_OUTPUT.c.galaxy_id == row['galaxy_id'])).first()
            if not output_row:
                continue
            row_outputs = map_outputrow2list(output_row)

        elif output_type == 'best_fit':
            output_row = connection.execute(select([BEST_FIT_OUTPUT]).where(BEST_FIT_OUTPUT.c.galaxy_id == row['galaxy_id'])).first()
            if not output_row:
                continue
            row_outputs = map_best_fit_output2list(output_row)

        elif output_type == 'best_fit_model':
            output_row = connection.execute(select([BEST_FIT_MODEL]).where(BEST_FIT_MODEL.c.galaxy_id == row['galaxy_id'])).first()
            if not output_row:
                continue
            row_outputs = map_best_fit_model_output2list(output_row)

        elif output_type == 'best_fit_inputs':
            output_row = connection.execute(select([BEST_FIT_HISTOGRAM]).where(BEST_FIT_HISTOGRAM.c.galaxy_id == row['galaxy_id'])).first()
            if not output_row:
                continue
            row_outputs = map_best_fit_output_inputs2list(output_row)
        else:
            raise Exception('Invalid output type')

        if input_type == 'normal':
            input_row = connection.execute(select([INPUT]).where(INPUT.c.galaxy_id == row['galaxy_id'])).first()

            if not input_row:
                continue

            if include_sigma:
                row_inputs = map_inputrow2list(input_row, input_filter_types)
            else:
                row_inputs = map_inputrow2list_nosigma(input_row, input_filter_types)

        elif input_type == 'Jy':
            # For these, the SNR and normal readings are already interleaved

            input_row = connection.execute(select([INPUT_JY]).where(INPUT_JY.c.galaxy_id == row['galaxy_id'])).first()

            if not input_row:
                continue

            if include_sigma:
                row_inputs = map_inputrow2list_Jy(input_row, input_filter_types)
            else:
                row_inputs = map_inputrow2list_Jy_nosigma(input_row, input_filter_types)

        # Add the row even if it's not a value, if we have an unknown input handler
        if not unknown_input_handler and not check_valid_row(row_inputs):
            # This row contains some invalid data (such as -999s or 0s)
            continue

        # Get redshift
        for i in range(0, repeat_redshift):
            row_inputs.insert(0, row['redshift'])

        redshifts.append(row['redshift'])

        galaxy_ids.append(row['galaxy_id'])

        all_in.append(row_inputs)
        all_out.append(row_outputs)

        added_count += 1

        sys.stdout.write('\rProgress: {0}/{1} {2:4.1f}%'.format(added_count, total_to_get, added_count/float(total_to_get)*100))
        sys.stdout.flush()

        if added_count == total_to_get:
            break

    if unknown_input_handler:
        all_in = unknown_input_handler(all_in, 0)  # Ignore the first one as this is redshift and CAN be 0

    if added_count < total_to_get:
        print "\nCould not get {0} good readings. Got {1} instead.".format(total_to_get, added_count)
    # test in, test out, train in, train out
    return all_in, all_out, redshifts, galaxy_ids

if __name__ == '__main__':

    LOG.info('Performing database clense.')
    transaction = connection.begin()
    for table in reversed(MAGPHYS_NN_METADATA.sorted_tables):
        connection.execute(table.delete())
    transaction.commit()
    LOG.info('Done')


