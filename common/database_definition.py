import os, sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

from sqlalchemy import MetaData, Table, Column, Integer, String, Float, TIMESTAMP, ForeignKey, BigInteger, DateTime
from sqlalchemy.engine import create_engine
from sqlalchemy.sql.expression import select
from Jy_to_luminosity import convert_to_luminosity

MAGPHYS_NN_METADATA_old = MetaData()
MAGPHYS_NN_METADATA = MetaData()

# FIT file names -> database names
median_output_parameter_name_map = {
    'ager': 'ager',
    'tau_V': 'tau_V',
    'agem': 'agem',
    'tlastb': 'tlastb',
    'M(stars)': 'Mstars',
    'sfr29': 'sfr29',
    'xi_PAH^tot': 'xi_PAHtot',
    'f_mu(SFH)': 'f_muSFH',
    'fb17': 'fb17',
    'fb16': 'fb16',
    'T_C^ISM': 'T_CISM',
    'Ldust': 'Ldust',
    'mu_parameter': 'mu_parameter',
    'xi_C^tot': 'xi_Ctot',
    'xi_W^tot': 'xi_Wtot',
    'f_mu(IR)': 'f_muIR',
    'fb18': 'fb18',
    'fb19': 'fb19',
    'T_W^BC': 'T_WBC',
    'SFR_0.1Gyr': 'SFR_0_1Gyr',
    'fb29': 'fb29',
    'sfr17': 'sfr17',
    'sfr16': 'sfr16',
    'sfr19': 'sfr19',
    'sfr18': 'sfr18',
    'tau_V^ISM': 'tau_VISM',
    'sSFR_0.1Gyr': 'sSFR_0_1Gyr',
    'metalicity Z/Zo': 'metalicity_Z_Z0',
    'M(dust)': 'Mdust',
    'xi_MIR^tot': 'xi_MIRtot',
    'tform': 'tform',
    'gamma': 'gamma'
}

# FIT file names -> database names
best_fit_output_parameter_name_map = {
    'fmu(SFH)': 'fmuSFH',
    'fmu(IR)': 'fmuIR',
    'mu': 'mu',
    'tauv': 'tauv',
    'sSFR': 'sSFR',
    'M*': 'M_asterisk',
    'Ldust': 'Ldust',
    'T_W^BC': 'T_WBC',
    'T_C^ISM': 'T_CISM',
    'xi_C^tot': 'xi_Ctot',
    'xi_PAH^tot': 'xi_PAHtot',
    'xi_MIR^tot': 'xi_MIRtot',
    'xi_W^tot': 'xi_Wtot',
    'tvism': 'tvism',
    'Mdust': 'Mdust',
    'SFR': 'SFR'
}

NN_TRAIN_old = Table('nn_train',
                     MAGPHYS_NN_METADATA_old,
                 Column('train_id', Integer, primary_key=True, autoincrement=True),
                 Column('run_id', String(30)),
                 Column('filename', String(200)),
                 Column('last_updated', TIMESTAMP),
                 Column('redshift', Float),
                 Column('galaxy_number', Integer),
                 Column('input', Integer, ForeignKey('input.input_id')),
                 Column('input_snr', Integer, ForeignKey('input.input_id')),

                 Column('input_Jy', Integer, ForeignKey('input_Jy.input_Jy_id')),
                 Column('input_Jy_snr', Integer, ForeignKey('input_Jy.input_Jy_id')),

                 Column('output_median', Integer, ForeignKey('median_output.median_output_id')),
                 Column('output_best_fit', Integer, ForeignKey('best_fit_output.best_fit_output_id')),
                 Column('output_best_fit_model', Integer, ForeignKey('best_fit_model_output.best_fit_model_id')),
                 Column('output_best_fit_inputs', Integer, ForeignKey('best_fit_output_input.best_fit_output_input_id')),
                 )

INPUT_old = Table('input',
                  MAGPHYS_NN_METADATA_old,
              Column('input_id', Integer, primary_key=True, autoincrement=True),
              Column('type', String(20)),
              Column('fuv', Float),
              Column('nuv', Float),
              Column('u', Float),
              Column('g', Float),
              Column('r', Float),
              Column('z', Float),
              Column('Z_', Float),
              Column('Y', Float),
              Column('J', Float),
              Column('H', Float),
              Column('K', Float),
              Column('WISEW1', Float),
              Column('WISEW2', Float),
              Column('WISEW3', Float),
              Column('WISEW4', Float),
              Column('PACS100', Float),
              Column('PACS160', Float),
              Column('SPIRE250', Float),
              Column('SPIRE350', Float),
              Column('SPIRE500', Float)
              )

INPUT_JY_old = Table('input_Jy',
                     MAGPHYS_NN_METADATA_old,
                 Column('input_Jy_id', Integer, primary_key=True, autoincrement=True),
                 Column('type', String(30)),
                 Column('fuv', Float),
                 Column('nuv', Float),
                 Column('u', Float),
                 Column('g', Float),
                 Column('r', Float),
                 Column('Unknown', Float),
                 Column('z', Float),
                 Column('Z_', Float),
                 Column('Y', Float),
                 Column('J', Float),
                 Column('H', Float),
                 Column('K', Float),
                 Column('WISEW1', Float),
                 Column('WISEW2', Float),
                 Column('WISEW3', Float),
                 Column('WISEW4', Float),
                 Column('PACS100', Float),
                 Column('PACS160', Float),
                 Column('SPIRE250', Float),
                 Column('SPIRE350', Float),
                 Column('SPIRE500', Float),
                 )

MEDIAN_OUTPUT_old = Table('median_output',
                          MAGPHYS_NN_METADATA_old,
                      Column('median_output_id', Integer, primary_key=True, autoincrement=True),
                      Column('ager', Float),
                      Column('tau_V', Float),
                      Column('agem', Float),
                      Column('tlastb', Float),
                      Column('Mstars', Float),
                      Column('sfr29', Float),
                      Column('xi_PAHtot', Float),
                      Column('f_muSFH', Float),
                      Column('fb17', Float),
                      Column('fb16', Float),
                      Column('T_CISM', Float),
                      Column('Ldust', Float),
                      Column('mu_parameter', Float),
                      Column('xi_Ctot', Float),
                      Column('xi_Wtot', Float),
                      Column('f_muIR', Float),
                      Column('fb18', Float),
                      Column('fb19', Float),
                      Column('T_WBC', Float),
                      Column('SFR_0_1Gyr', Float),
                      Column('fb29', Float),
                      Column('sfr17', Float),
                      Column('sfr16', Float),
                      Column('sfr19', Float),
                      Column('sfr18', Float),
                      Column('tau_VISM', Float),
                      Column('sSFR_0_1Gyr', Float),
                      Column('metalicity_Z_Z0', Float),
                      Column('Mdust', Float),
                      Column('xi_MIRtot', Float),
                      Column('tform', Float),
                      Column('gamma', Float)
                      )

BEST_FIT_OUTPUT_old = Table('best_fit_output',
                            MAGPHYS_NN_METADATA_old,
                        Column('best_fit_output_id', Integer, primary_key=True, autoincrement=True),
                        Column('fmuSFH', Float),
                        Column('fmuIR', Float),
                        Column('mu', Float),
                        Column('tauv', Float),
                        Column('sSFR', Float),
                        Column('m_asterisk', Float),
                        Column('Ldust', Float),
                        Column('T_WBC', Float),
                        Column('T_CISM', Float),
                        Column('xi_Ctot', Float),
                        Column('xi_PAHtot', Float),
                        Column('xi_MIRtot', Float),
                        Column('xi_Wtot', Float),
                        Column('tvism', Float),
                        Column('Mdust', Float),
                        Column('SFR', Float),
                        )


BEST_FIT_OUTPUT_INPUT_old = Table('best_fit_output_input',
                                  MAGPHYS_NN_METADATA_old,
                              Column('best_fit_output_input_id', Integer, primary_key=True, autoincrement=True),
                              Column('fuv', Float),
                              Column('nuv', Float),
                              Column('u', Float),
                              Column('g', Float),
                              Column('r', Float),
                              Column('z', Float),
                              Column('Z_', Float),
                              Column('Y', Float),
                              Column('J', Float),
                              Column('H', Float),
                              Column('K', Float),
                              Column('WISEW1', Float),
                              Column('WISEW2', Float),
                              Column('WISEW3', Float),
                              Column('WISEW4', Float),
                              Column('PACS100', Float),
                              Column('PACS160', Float),
                              Column('SPIRE250', Float),
                              Column('SPIRE350', Float),
                              Column('SPIRE500', Float)
                              )

BEST_FIT_MODEL_old = Table('best_fit_model_output',
                           MAGPHYS_NN_METADATA_old,
                       Column('best_fit_model_id', Integer, primary_key=True, autoincrement=True),
                       Column('i_sfh', Float),
                       Column('i_ir', Float),
                       Column('chi2', Float),
                       Column('redshift', Float)
                       )

# --------------------------------------- NEW DATABASE DEFINITIONS-----------------------------------------
NN_TRAIN = Table('nn_train',
                 MAGPHYS_NN_METADATA,
                 Column('galaxy_id', Integer, primary_key=True, autoincrement=True),
                 Column('redshift', Float),
                 Column('run_id', String(30)),
                 Column('fit_filename', String(200)),
                 Column('run_filename', String(200)),
                 )

INPUT = Table('input',
              MAGPHYS_NN_METADATA,
              Column('galaxy_id', Integer, ForeignKey('nn_train.galaxy_id'), primary_key=True, autoincrement=True),
              Column('fuv', Float),
              Column('nuv', Float),
              Column('u', Float),
              Column('g', Float),
              Column('r', Float),
              Column('i', Float),
              Column('z', Float),
              Column('Z_', Float),
              Column('Y', Float),
              Column('J', Float),
              Column('H', Float),
              Column('K', Float),
              Column('WISEW1', Float),
              Column('WISEW2', Float),
              Column('WISEW3', Float),
              Column('WISEW4', Float),
              Column('PACS100', Float),
              Column('PACS160', Float),
              Column('SPIRE250', Float),
              Column('SPIRE350', Float),
              Column('SPIRE500', Float),
              Column('fuv_snr', Float),
              Column('nuv_snr', Float),
              Column('u_snr', Float),
              Column('g_snr', Float),
              Column('r_snr', Float),
              Column('i_snr', Float),
              Column('z_snr', Float),
              Column('Z__snr', Float),
              Column('Y_snr', Float),
              Column('J_snr', Float),
              Column('H_snr', Float),
              Column('K_snr', Float),
              Column('WISEW1_snr', Float),
              Column('WISEW2_snr', Float),
              Column('WISEW3_snr', Float),
              Column('WISEW4_snr', Float),
              Column('PACS100_snr', Float),
              Column('PACS160_snr', Float),
              Column('SPIRE250_snr', Float),
              Column('SPIRE350_snr', Float),
              Column('SPIRE500_snr', Float),
              )

INPUT_JY = Table('input_Jy',
                 MAGPHYS_NN_METADATA,
                 Column('galaxy_id', Integer, ForeignKey('nn_train.galaxy_id'), primary_key=True, autoincrement=True),
                 Column('fuv', Float),
                 Column('nuv', Float),
                 Column('u', Float),
                 Column('g', Float),
                 Column('r', Float),
                 Column('i', Float),
                 Column('z', Float),
                 Column('Z_', Float),
                 Column('Y', Float),
                 Column('J', Float),
                 Column('H', Float),
                 Column('K', Float),
                 Column('WISEW1', Float),
                 Column('WISEW2', Float),
                 Column('WISEW3', Float),
                 Column('WISEW4', Float),
                 Column('PACS100', Float),
                 Column('PACS160', Float),
                 Column('SPIRE250', Float),
                 Column('SPIRE350', Float),
                 Column('SPIRE500', Float),
                 Column('fuv_snr', Float),
                 Column('nuv_snr', Float),
                 Column('u_snr', Float),
                 Column('g_snr', Float),
                 Column('r_snr', Float),
                 Column('i_snr', Float),
                 Column('z_snr', Float),
                 Column('Z__snr', Float),
                 Column('Y_snr', Float),
                 Column('J_snr', Float),
                 Column('H_snr', Float),
                 Column('K_snr', Float),
                 Column('WISEW1_snr', Float),
                 Column('WISEW2_snr', Float),
                 Column('WISEW3_snr', Float),
                 Column('WISEW4_snr', Float),
                 Column('PACS100_snr', Float),
                 Column('PACS160_snr', Float),
                 Column('SPIRE250_snr', Float),
                 Column('SPIRE350_snr', Float),
                 Column('SPIRE500_snr', Float),
                 )

MEDIAN_OUTPUT = Table('median_output',
                      MAGPHYS_NN_METADATA,
                      Column('galaxy_id', Integer, ForeignKey('nn_train.galaxy_id'), primary_key=True, autoincrement=True),
                      Column('ager', Float),
                      Column('tau_V', Float),
                      Column('agem', Float),
                      Column('tlastb', Float),
                      Column('Mstars', Float),
                      Column('sfr29', Float),
                      Column('xi_PAHtot', Float),
                      Column('f_muSFH', Float),
                      Column('fb17', Float),
                      Column('fb16', Float),
                      Column('T_CISM', Float),
                      Column('Ldust', Float),
                      Column('mu_parameter', Float),
                      Column('xi_Ctot', Float),
                      Column('xi_Wtot', Float),
                      Column('f_muIR', Float),
                      Column('fb18', Float),
                      Column('fb19', Float),
                      Column('T_WBC', Float),
                      Column('SFR_0_1Gyr', Float),
                      Column('fb29', Float),
                      Column('sfr17', Float),
                      Column('sfr16', Float),
                      Column('sfr19', Float),
                      Column('sfr18', Float),
                      Column('tau_VISM', Float),
                      Column('sSFR_0_1Gyr', Float),
                      Column('metalicity_Z_Z0', Float),
                      Column('Mdust', Float),
                      Column('xi_MIRtot', Float),
                      Column('tform', Float),
                      Column('gamma', Float)
                      )

BEST_FIT_OUTPUT = Table('best_fit_output',
                        MAGPHYS_NN_METADATA,
                        Column('galaxy_id', Integer, ForeignKey('nn_train.galaxy_id'), primary_key=True, autoincrement=True),
                        Column('fmuSFH', Float),
                        Column('fmuIR', Float),
                        Column('mu', Float),
                        Column('tauv', Float),
                        Column('sSFR', Float),
                        Column('m_asterisk', Float),
                        Column('Ldust', Float),
                        Column('T_WBC', Float),
                        Column('T_CISM', Float),
                        Column('xi_Ctot', Float),
                        Column('xi_PAHtot', Float),
                        Column('xi_MIRtot', Float),
                        Column('xi_Wtot', Float),
                        Column('tvism', Float),
                        Column('Mdust', Float),
                        Column('SFR', Float),
                        )


BEST_FIT_HISTOGRAM = Table('best_fit_histogram',
                           MAGPHYS_NN_METADATA,
                           Column('galaxy_id', Integer, ForeignKey('nn_train.galaxy_id'), primary_key=True, autoincrement=True),
                           Column('fuv', Float),
                           Column('nuv', Float),
                           Column('u', Float),
                           Column('g', Float),
                           Column('r', Float),
                           Column('i', Float),
                           Column('z', Float),
                           Column('Z_', Float),
                           Column('Y', Float),
                           Column('J', Float),
                           Column('H', Float),
                           Column('K', Float),
                           Column('WISEW1', Float),
                           Column('WISEW2', Float),
                           Column('WISEW3', Float),
                           Column('WISEW4', Float),
                           Column('PACS100', Float),
                           Column('PACS160', Float),
                           Column('SPIRE250', Float),
                           Column('SPIRE350', Float),
                           Column('SPIRE500', Float)
                           )

BEST_FIT_MODEL = Table('best_fit_model',
                       MAGPHYS_NN_METADATA,
                       Column('galaxy_id', Integer, ForeignKey('nn_train.galaxy_id'), primary_key=True, autoincrement=True),
                       Column('i_sfh', Float),
                       Column('i_ir', Float),
                       Column('chi2', Float),
                       Column('redshift', Float)
                       )


def convert_from_jy(input, input_snr, redshift):
    out_list = [None] * 21
    out_list_snr = [None] * 21
    
    out_list[0] = input['fuv']
    out_list[1] = input['nuv']
    out_list[2] = input['u']
    out_list[3] = input['g']
    out_list[4] = input['r']
    out_list[5] = input['z']
    out_list[6] = input['Z_']
    out_list[7] = input['Y']
    out_list[8] = input['J']
    out_list[9] = input['H']
    out_list[10] = input['K']
    out_list[11] = input['WISEW1']
    out_list[12] = input['WISEW2']
    out_list[13] = input['WISEW3']
    out_list[14] = input['WISEW4']
    out_list[15] = input['PACS100']
    out_list[16] = input['PACS160']
    out_list[17] = input['SPIRE250']
    out_list[18] = input['SPIRE350']
    out_list[19] = input['SPIRE500']
    out_list[20] = input['Unknown']

    out_list_snr[0] = input_snr['fuv']
    out_list_snr[1] = input_snr['nuv']
    out_list_snr[2] = input_snr['u']
    out_list_snr[3] = input_snr['g']
    out_list_snr[4] = input_snr['r']
    out_list_snr[5] = input_snr['z']
    out_list_snr[6] = input_snr['Z_']
    out_list_snr[7] = input_snr['Y']
    out_list_snr[8] = input_snr['J']
    out_list_snr[9] = input_snr['H']
    out_list_snr[10] = input_snr['K']
    out_list_snr[11] = input_snr['WISEW1']
    out_list_snr[12] = input_snr['WISEW2']
    out_list_snr[13] = input_snr['WISEW3']
    out_list_snr[14] = input_snr['WISEW4']
    out_list_snr[15] = input_snr['PACS100']
    out_list_snr[16] = input_snr['PACS160']
    out_list_snr[17] = input_snr['SPIRE250']
    out_list_snr[18] = input_snr['SPIRE350']
    out_list_snr[19] = input_snr['SPIRE500']
    out_list_snr[20] = input_snr['Unknown']
    
    return convert_to_luminosity(out_list, out_list_snr, redshift)
    
    
def database_migrate(old_db, new_db):
    old_db_connection = create_engine(old_db).connect()
    new_db_connection = create_engine(new_db).connect()

    nn_train_old = old_db_connection.execute(select([NN_TRAIN_old]).where(NN_TRAIN_old.c.input != None))

    print 'Starting...'
    count = 0
    invalids = 0
    for result in nn_train_old:
        sys.stdout.write('\r Done: {0} ({1} invalids)'.format(count, invalids))
        sys.stdout.flush()

        # Build something to hold the data
        input_jy = old_db_connection.execute(select([INPUT_JY_old]).where(INPUT_JY_old.c.input_Jy_id == result['input_Jy'])).first()
        input_jy_snr = old_db_connection.execute(select([INPUT_JY_old]).where(INPUT_JY_old.c.input_Jy_id == result['input_Jy_snr'])).first()
        median_output = old_db_connection.execute(select([MEDIAN_OUTPUT_old]).where(MEDIAN_OUTPUT_old.c.median_output_id == result['output_median'])).first()

        if median_output is None:
            invalids += 1
            continue
        count += 1
        # Commit the data in the new format
        transaction = new_db_connection.begin()

        new_db_connection.execute(NN_TRAIN.insert().values(galaxy_id=result['galaxy_number'],
                                                           redshift=result['redshift'],
                                                           run_id=result['run_id'],
                                                           fit_filename=result['filename']
                                                           ))

        new_db_connection.execute(INPUT_JY.insert().values(galaxy_id=result['galaxy_number'],
                                                           fuv=input_jy['fuv'],
                                                           nuv=input_jy['nuv'],
                                                           u=input_jy['u'],
                                                           g=input_jy['g'],
                                                           r=input_jy['r'],
                                                           i=input_jy['z'],
                                                           z=input_jy['Z_'],
                                                           Z_=input_jy['Y'],
                                                           Y=input_jy['J'],
                                                           J=input_jy['H'],
                                                           H=input_jy['K'],
                                                           K=input_jy['WISEW1'],
                                                           WISEW1=input_jy['WISEW2'],
                                                           WISEW2=input_jy['WISEW3'],
                                                           WISEW3=input_jy['WISEW4'],
                                                           WISEW4=input_jy['PACS100'],
                                                           PACS100=input_jy['PACS160'],
                                                           PACS160=input_jy['SPIRE250'],
                                                           SPIRE250=input_jy['SPIRE350'],
                                                           SPIRE350=input_jy['SPIRE500'],
                                                           SPIRE500=input_jy['Unknown'],

                                                           fuv_snr=input_jy_snr['fuv'],
                                                           nuv_snr=input_jy_snr['nuv'],
                                                           u_snr=input_jy_snr['u'],
                                                           g_snr=input_jy_snr['g'],
                                                           r_snr=input_jy_snr['r'],
                                                           i_snr=input_jy_snr['z'],
                                                           z_snr=input_jy_snr['Z_'],
                                                           Z__snr=input_jy_snr['Y'],
                                                           Y_snr=input_jy_snr['J'],
                                                           J_snr=input_jy_snr['H'],
                                                           H_snr=input_jy_snr['K'],
                                                           K_snr=input_jy_snr['WISEW1'],
                                                           WISEW1_snr=input_jy_snr['WISEW2'],
                                                           WISEW2_snr=input_jy_snr['WISEW3'],
                                                           WISEW3_snr=input_jy_snr['WISEW4'],
                                                           WISEW4_snr=input_jy_snr['PACS100'],
                                                           PACS100_snr=input_jy_snr['PACS160'],
                                                           PACS160_snr=input_jy_snr['SPIRE250'],
                                                           SPIRE250_snr=input_jy_snr['SPIRE350'],
                                                           SPIRE350_snr=input_jy_snr['SPIRE500'],
                                                           SPIRE500_snr=input_jy_snr['Unknown'],
                                                           ))
        
        new_db_connection.execute(MEDIAN_OUTPUT.insert().values(galaxy_id=result['galaxy_number'],
                                                                xi_Wtot=median_output['xi_Wtot'],
                                                                ager=median_output['ager'],
                                                                tau_V=median_output['tau_V'],
                                                                agem=median_output['agem'],
                                                                tlastb=median_output['tlastb'],
                                                                Mstars=median_output['Mstars'],
                                                                sfr29=median_output['sfr29'],
                                                                xi_PAHtot=median_output['xi_PAHtot'],
                                                                f_muSFH=median_output['f_muSFH'],
                                                                fb17=median_output['fb17'],
                                                                fb16=median_output['fb16'],
                                                                T_CISM=median_output['T_CISM'],
                                                                Ldust=median_output['Ldust'],
                                                                mu_parameter=median_output['mu_parameter'],
                                                                xi_Ctot=median_output['xi_Ctot'],
                                                                f_muIR=median_output['f_muIR'],
                                                                fb18=median_output['fb18'],
                                                                fb19=median_output['fb19'],
                                                                T_WBC=median_output['T_WBC'],
                                                                SFR_0_1Gyr=median_output['SFR_0_1Gyr'],
                                                                fb29=median_output['fb29'],
                                                                sfr17=median_output['sfr17'],
                                                                sfr16=median_output['sfr16'],
                                                                sfr19=median_output['sfr19'],
                                                                sfr18=median_output['sfr18'],
                                                                tau_VISM=median_output['tau_VISM'],
                                                                sSFR_0_1Gyr=median_output['sSFR_0_1Gyr'],
                                                                metalicity_Z_Z0=median_output['metalicity_Z_Z0'],
                                                                Mdust=median_output['Mdust'],
                                                                xi_MIRtot=median_output['xi_MIRtot'],
                                                                tform=median_output['tform'],
                                                                gamma=median_output['gamma']))

        converted1, converted2 = convert_from_jy(input_jy, input_jy_snr, result['redshift'])

        new_db_connection.execute(INPUT.insert().values(galaxy_id=result['galaxy_number'],
                                                        fuv=converted1[0],
                                                        nuv=converted1[1],
                                                        u=converted1[2],
                                                        g=converted1[3],
                                                        r=converted1[4],
                                                        i=converted1[5],
                                                        z=converted1[6],
                                                        Z_=converted1[7],
                                                        Y=converted1[8],
                                                        J=converted1[9],
                                                        H=converted1[10],
                                                        K=converted1[11],
                                                        WISEW1=converted1[12],
                                                        WISEW2=converted1[13],
                                                        WISEW3=converted1[14],
                                                        WISEW4=converted1[15],
                                                        PACS100=converted1[16],
                                                        PACS160=converted1[17],
                                                        SPIRE250=converted1[18],
                                                        SPIRE350=converted1[19],
                                                        SPIRE500=converted1[20],

                                                        fuv_snr=converted2[0],
                                                        nuv_snr=converted2[1],
                                                        u_snr=converted2[2],
                                                        g_snr=converted2[3],
                                                        r_snr=converted2[4],
                                                        i_snr=converted2[5],
                                                        z_snr=converted2[6],
                                                        Z__snr=converted2[7],
                                                        Y_snr=converted2[8],
                                                        J_snr=converted2[9],
                                                        H_snr=converted2[10],
                                                        K_snr=converted2[11],
                                                        WISEW1_snr=converted2[12],
                                                        WISEW2_snr=converted2[13],
                                                        WISEW3_snr=converted2[14],
                                                        WISEW4_snr=converted2[15],
                                                        PACS100_snr=converted2[16],
                                                        PACS160_snr=converted2[17],
                                                        SPIRE250_snr=converted2[18],
                                                        SPIRE350_snr=converted2[19],
                                                        SPIRE500_snr=converted2[20],
                                                        ))
        transaction.commit()

if __name__ == '__main__':
    database_migrate('sqlite:///old.db', 'sqlite:///new.db')