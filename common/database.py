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

from sqlalchemy.engine import create_engine
from sqlalchemy import MetaData, Table, Column, Integer, String, Float, TIMESTAMP, ForeignKey, BigInteger, DateTime
from sqlalchemy import func
from sqlalchemy.sql.expression import select, func as ffunc

from common import config
from logger import config_logger
import numpy as np
import os


LOG = config_logger(__name__)

engine = create_engine(config.DB_LOGIN)
connection = engine.connect()

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

NN_TRAIN = Table('nn_train',
                 MAGPHYS_NN_METADATA,
                 Column('train_id', Integer, primary_key=True, autoincrement=True),
                 Column('run_id', String(30)),
                 Column('filename', String(200)),
                 Column('last_updated', TIMESTAMP),
                 Column('redshift', Float),
                 Column('galaxy_number', Integer),
                 Column('input', Integer, ForeignKey('input.input_id')),
                 Column('input_snr', Integer, ForeignKey('input.input_id')),
                 
                 # Be cheeky and use the pre-existing Input table. This is seriously bad design ;)
                 # Things are in the same format though, so whatever.
                 Column('input_process_data', Integer, ForeignKey('input.input_id')),
                 Column('input_process_data_snr', Integer, ForeignKey('input.input_id')),
                 
                 Column('output_median', Integer, ForeignKey('median_output.median_output_id')),
                 Column('output_best_fit', Integer, ForeignKey('best_fit_output.best_fit_output_id')),
                 Column('output_best_fit_model', Integer, ForeignKey('best_fit_model_output.best_fit_model_id')),
                 Column('output_best_fit_inputs', Integer, ForeignKey('best_fit_output_input.best_fit_output_input_id')),
                 )

INPUT = Table('input',
              MAGPHYS_NN_METADATA,
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

MEDIAN_OUTPUT = Table('median_output',
                      MAGPHYS_NN_METADATA,
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

BEST_FIT_OUTPUT = Table('best_fit_output',
                        MAGPHYS_NN_METADATA,
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


BEST_FIT_OUTPUT_INPUT = Table('best_fit_output_input',
                              MAGPHYS_NN_METADATA,
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

BEST_FIT_MODEL = Table('best_fit_model_output',
                       MAGPHYS_NN_METADATA,
                       Column('best_fit_model_id', Integer, primary_key=True, autoincrement=True),
                       Column('i_sfh', Float),
                       Column('i_ir', Float),
                       Column('chi2', Float),
                       Column('redshift', Float)
                       )


def add_process_data_to_db(galaxy, run_id):

    input_key = connection.execute(INPUT.insert().values(type='process',
                                    fuv=galaxy['fuv'],
                                    nuv=galaxy['nuv'],
                                    u=galaxy['u'],
                                    g=galaxy['g'],
                                    r=galaxy['r'],
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
                                    SPIRE500=galaxy['SPIRE500']
                                    )).inserted_primary_key[0]

    input_snr_key = connection.execute(INPUT.insert().values(type='process_snr',
                                    fuv=galaxy['fuv_snr'],
                                    nuv=galaxy['nuv_snr'],
                                    u=galaxy['u_snr'],
                                    g=galaxy['g_snr'],
                                    r=galaxy['r_snr'],
                                    z=galaxy['z_snr'],
                                    Z_=galaxy['Z_snr'],
                                    Y=galaxy['Y_snr'],
                                    J=galaxy['J_snr'],
                                    H=galaxy['H_snr'],
                                    K=galaxy['K_snr'],
                                    WISEW1=galaxy['WISEW1_snr'],
                                    WISEW2=galaxy['WISEW2_snr'],
                                    WISEW3=galaxy['WISEW3_snr'],
                                    WISEW4=galaxy['WISEW4_snr'],
                                    PACS100=galaxy['PACS100_snr'],
                                    PACS160=galaxy['PACS160_snr'],
                                    SPIRE250=galaxy['SPIRE250_snr'],
                                    SPIRE350=galaxy['SPIRE350_snr'],
                                    SPIRE500=galaxy['SPIRE500_snr']
                                    )).inserted_primary_key[0]
    
    connection.execute(NN_TRAIN.insert().values(run_id=run_id,
                                             last_updated=func.now(),
                                             redshift=galaxy['redshift'],
                                             galaxy_number=galaxy['galaxy_number'],
                                             input_process_data=input_key,
                                             input_process_data_snr=input_snr_key
                                             ))


def add_to_db(input, input_snr, output, best_fit_output, best_fit_model, best_fit_output_input, details):

    transaction = connection.begin()

    input_key = connection.execute(INPUT.insert().values(type='fit',
                    fuv=input['fuv'],
                    nuv=input['nuv'],
                    u=input['u'],
                    g=input['g'],
                    r=input['r'],
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
                    SPIRE500=input['SPIRE500']
                    )).inserted_primary_key[0]
    
    input_snr_key = connection.execute(INPUT.insert().values(type='fit_snr',
                        fuv=input_snr['fuv'],
                        nuv=input_snr['nuv'],
                        u=input_snr['u'],
                        g=input_snr['g'],
                        r=input_snr['r'],
                        z=input_snr['z'],
                        Z_=input_snr['Z'],
                        Y=input_snr['Y'],
                        J=input_snr['J'],
                        H=input_snr['H'],
                        K=input_snr['K'],
                        WISEW1=input_snr['WISEW1'],
                        WISEW2=input_snr['WISEW2'],
                        WISEW3=input_snr['WISEW3'],
                        WISEW4=input_snr['WISEW4'],
                        PACS100=input_snr['PACS100'],
                        PACS160=input_snr['PACS160'],
                        SPIRE250=input_snr['SPIRE250'],
                        SPIRE350=input_snr['SPIRE350'],
                        SPIRE500=input_snr['SPIRE500']
                        )).inserted_primary_key[0]
    
    output_key = connection.execute(MEDIAN_OUTPUT.insert().values(xi_Wtot=output['xi_W^tot'],
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
                                        )).inserted_primary_key[0]

    best_fit_output_key = connection.execute(BEST_FIT_OUTPUT.insert().values(fmuSFH=best_fit_output['fmu(SFH)'],
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
                                        )).inserted_primary_key[0]

    best_fit_output_inputs_key = connection.execute(BEST_FIT_OUTPUT_INPUT.insert().values(fuv=best_fit_output_input['fuv'],
                                        nuv=best_fit_output_input['nuv'],
                                        u=best_fit_output_input['u'],
                                        g=best_fit_output_input['g'],
                                        r=best_fit_output_input['r'],
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
                                        )).inserted_primary_key[0]

    best_fit_output_model_key = connection.execute(BEST_FIT_MODEL.insert().values(i_sfh=best_fit_model['i_sfh'],
                                        i_ir=best_fit_model['i_ir'],
                                        chi2=best_fit_model['chi2'],
                                        redshift=best_fit_model['redshift']
                                        )).inserted_primary_key[0]

    head, tail = os.path.split(details['filename'])

    connection.execute(NN_TRAIN.update().where(NN_TRAIN.c.galaxy_number == int(tail.split('.fit')[0]))
                             .values(run_id=details['run_id'],
                             filename=details['filename'],
                             last_updated=func.now(),
                             input=input_key,
                             input_snr=input_snr_key,
                             output_median=output_key,
                             output_best_fit=best_fit_output_key,
                             output_best_fit_inputs=best_fit_output_inputs_key,
                             output_best_fit_model=best_fit_output_model_key
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


def map_inputrow2list(row, row_snr):
    out_list = [None] * 40
    #out_list = [None] * 20
    #"""
    # Normal values, suitable for NN input
    """
    out_list[0] = row['fuv']
    out_list[1] = row['nuv']
    out_list[2] = row['u']
    out_list[3] = row['g']
    out_list[4] = row['r']
    out_list[5] = row['z']
    out_list[6] = row['Z_']
    out_list[7] = row['Y']
    out_list[8] = row['J']
    out_list[9] = row['H']
    out_list[10] = row['K']
    out_list[11] = row['WISEW1']
    out_list[12] = row['WISEW2']
    out_list[13] = row['WISEW3']
    out_list[14] = row['WISEW4']
    out_list[15] = row['PACS100']
    out_list[16] = row['PACS160']
    out_list[17] = row['SPIRE250']
    out_list[18] = row['SPIRE350']
    out_list[19] = row['SPIRE500']

    out_list[20] = row_snr['fuv']
    out_list[21] = row_snr['nuv']
    out_list[22] = row_snr['u']
    out_list[23] = row_snr['g']
    out_list[24] = row_snr['r']
    out_list[25] = row_snr['z']
    out_list[26] = row_snr['Z_']
    out_list[27] = row_snr['Y']
    out_list[28] = row_snr['J']
    out_list[29] = row_snr['H']
    out_list[30] = row_snr['K']
    out_list[31] = row_snr['WISEW1']
    out_list[32] = row_snr['WISEW2']
    out_list[33] = row_snr['WISEW3']
    out_list[34] = row_snr['WISEW4']
    out_list[35] = row_snr['PACS100']
    out_list[36] = row_snr['PACS160']
    out_list[37] = row_snr['SPIRE250']
    out_list[38] = row_snr['SPIRE350']
    out_list[39] = row_snr['SPIRE500']
    """
    out_list[0] = row['fuv']
    out_list[2] = row['nuv']
    out_list[4] = row['u']
    out_list[6] = row['g']
    out_list[8] = row['r']
    out_list[10] = row['z']
    out_list[12] = row['Z_']
    out_list[14] = row['Y']
    out_list[16] = row['J']
    out_list[18] = row['H']
    out_list[20] = row['K']
    out_list[22] = row['WISEW1']
    out_list[24] = row['WISEW2']
    out_list[26] = row['WISEW3']
    out_list[28] = row['WISEW4']
    out_list[30] = row['PACS100']
    out_list[32] = row['PACS160']
    out_list[34] = row['SPIRE250']
    out_list[36] = row['SPIRE350']
    out_list[38] = row['SPIRE500']


    # SNR values, suitable for NN input
    out_list[1] = row_snr['fuv']
    out_list[3] = row_snr['nuv']
    out_list[5] = row_snr['u']
    out_list[7] = row_snr['g']
    out_list[9] = row_snr['r']
    out_list[11] = row_snr['z']
    out_list[13] = row_snr['Z_']
    out_list[15] = row_snr['Y']
    out_list[17] = row_snr['J']
    out_list[19] = row_snr['H']
    out_list[21] = row_snr['K']
    out_list[23] = row_snr['WISEW1']
    out_list[25] = row_snr['WISEW2']
    out_list[27] = row_snr['WISEW3']
    out_list[29] = row_snr['WISEW4']
    out_list[31] = row_snr['PACS100']
    out_list[33] = row_snr['PACS160']
    out_list[35] = row_snr['SPIRE250']
    out_list[37] = row_snr['SPIRE350']
    out_list[39] = row_snr['SPIRE500']

    return out_list


def map_outputrow2list(row):
    out_list = [None] * 32

    # Median output values
    out_list[0] = row['ager']
    out_list[1] = row['tau_V']
    out_list[2] = row['agem']
    out_list[3] = row['tlastb']
    out_list[4] = row['Mstars']
    out_list[5] = row['xi_Wtot']
    out_list[6] = row['sfr29']
    out_list[7] = row['xi_PAHtot']
    out_list[8] = row['f_muSFH']
    out_list[9] = row['fb17']
    out_list[10] = row['fb16']
    out_list[11] = row['T_CISM']
    out_list[12] = row['Ldust']
    out_list[13] = row['mu_parameter']
    out_list[14] = row['xi_Ctot']
    out_list[15] = row['f_muIR']
    out_list[16] = row['fb18']
    out_list[17] = row['fb19']
    out_list[18] = row['T_WBC']
    out_list[19] = row['SFR_0_1Gyr']
    out_list[20] = row['fb29']
    out_list[21] = row['sfr17']
    out_list[22] = row['sfr16']
    out_list[23] = row['sfr19']
    out_list[24] = row['sfr18']
    out_list[25] = row['tau_VISM']
    out_list[26] = row['sSFR_0_1Gyr']
    out_list[27] = row['metalicity_Z_Z0']
    out_list[28] = row['Mdust']
    out_list[29] = row['xi_MIRtot']
    out_list[30] = row['tform']
    out_list[31] = row['gamma']

    return out_list


def map_best_fit_output2list(row):
    out_list = [None] * 16

    out_list[0] = row['fmuSFH']
    out_list[1] = row['fmuIR']
    out_list[2] = row['mu']
    out_list[3] = row['tauv']
    out_list[4] = row['sSFR']
    out_list[5] = row['m_asterisk']
    out_list[6] = row['Ldust']
    out_list[7] = row['T_WBC']
    out_list[8] = row['T_CISM']
    out_list[9] = row['xi_Ctot']
    out_list[10] = row['xi_PAHtot']
    out_list[11] = row['xi_MIRtot']
    out_list[12] = row['xi_Wtot']
    out_list[13] = row['tvism']
    out_list[14] = row['Mdust']
    out_list[15] = row['SFR']

    return out_list


def map_best_fit_output_inputs2list(row):
    out_list = [None] * 20

    out_list[0] = row['fuv']
    out_list[1] = row['nuv']
    out_list[2] = row['u']
    out_list[3] = row['g']
    out_list[4] = row['r']
    out_list[5] = row['z']
    out_list[6] = row['Z_']
    out_list[7] = row['Y']
    out_list[8] = row['J']
    out_list[9] = row['H']
    out_list[10] = row['K']
    out_list[11] = row['WISEW1']
    out_list[12] = row['WISEW2']
    out_list[13] = row['WISEW3']
    out_list[14] = row['WISEW4']
    out_list[15] = row['PACS100']
    out_list[16] = row['PACS160']
    out_list[17] = row['SPIRE250']
    out_list[18] = row['SPIRE350']
    out_list[19] = row['SPIRE500']

    return out_list


def map_best_fit_model_output2list(row):
    out_list = [None] * 4

    out_list[0] = row['i_sfh']
    out_list[1] = row['i_ir']
    out_list[2] = row['chi2']
    out_list[3] = row['redshift']

    return out_list


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


def get_train_test_data(num_test, num_train, run_folder, single_value=None, output_type='median', repeat_redshift=1):

    total_to_get = num_train+num_test
    count = connection.execute(select([func.count(NN_TRAIN)]).where(NN_TRAIN.c.run_id.endswith(run_folder)).order_by(ffunc.random()).limit(total_to_get + total_to_get*0.05)).first()[0]

    print '{0} entries available'.format(count)
    if count < total_to_get:
        print "Error {0}".format(count)
        return None, None, None, None, None

    print 'Getting indexes for {0} entries'.format(count)
    result = connection.execute(select([NN_TRAIN]).where(NN_TRAIN.c.run_id.endswith(run_folder)).order_by(ffunc.random())).fetchall()

    # Train and test sets are now separated, need to get the actual data now.

    all_in = []
    all_out = []
    galaxy_ids = []

    added_count = 0

    print 'Getting input and output data for {0}'.format(output_type)
    for row in result:
        # Make new dict for this row

        # Pull out ID for input, input SNR and output
        input_id = row['input']
        input_snr_id = row['input_snr']

        if output_type == 'median':
            output_id = row['output_median']
            output_row = connection.execute(select([MEDIAN_OUTPUT]).where(MEDIAN_OUTPUT.c.median_output_id == output_id)).first()
            if not output_row:
                continue
            row_outputs = map_outputrow2list(output_row)

        elif output_type == 'best_fit':
            output_id = row['output_best_fit']
            output_row = connection.execute(select([BEST_FIT_OUTPUT]).where(BEST_FIT_OUTPUT.c.best_fit_output_id == output_id)).first()
            if not output_row:
                continue
            row_outputs = map_best_fit_output2list(output_row)

        elif output_type == 'best_fit_model':
            output_id = row['output_best_fit_model']
            output_row = connection.execute(select([BEST_FIT_MODEL]).where(BEST_FIT_MODEL.c.best_fit_model_id == output_id)).first()
            if not output_row:
                continue
            row_outputs = map_best_fit_model_output2list(output_row)

        elif output_type == 'best_fit_inputs':
            output_id = row['output_best_fit_inputs']
            output_row = connection.execute(select([BEST_FIT_OUTPUT_INPUT]).where(BEST_FIT_OUTPUT_INPUT.c.best_fit_output_input_id == output_id)).first()
            if not output_row:
                continue
            row_outputs = map_best_fit_output_inputs2list(output_row)
        else:
            raise Exception('Invalid output type')

        input_row = connection.execute(select([INPUT]).where(INPUT.c.input_id == input_id)).first()

        input_snr_row = connection.execute(select([INPUT]).where(INPUT.c.input_id == input_snr_id)).first()

        if not input_row or not input_snr_row:
            continue

        # Interleave the rows. Done manually (ugh).
        # Need this to ensure everything is in the right order for the NN to process
        row_inputs = map_inputrow2list(input_row, input_snr_row)

        if not check_valid_row(input_row):
            # This row contains some invalid data (such as -999s or 0s)
            continue

        if single_value is not None:  # If we only want to use a single value
            row_outputs = [row_outputs[single_value]]

        # Get redshift
        for i in range(0, repeat_redshift):
            row_inputs.insert(0, row['redshift'])

        galaxy_ids.append(row['galaxy_number'])

        all_in.append(row_inputs)
        all_out.append(row_outputs)

        added_count += 1

        sys.stdout.write('\rProgress: {0}/{1} {2:4.1f}%'.format(added_count, total_to_get, added_count/float(total_to_get)*100))
        sys.stdout.flush()

        if added_count == total_to_get:
            break

    if added_count < total_to_get:
        print "Could not get {0} good readings. Got {1} instead.".format(total_to_get, added_count)
    # test in, test out, train in, train out
    return np.array(all_in[:num_test]), np.array(all_out[:num_test]), np.array(all_in[num_test:]), np.array(all_out[num_test:]), galaxy_ids

if __name__ == '__main__':

    LOG.info('Performing database clense.')
    transaction = connection.begin()
    for table in reversed(MAGPHYS_NN_METADATA.sorted_tables):
        connection.execute(table.delete())
    transaction.commit()
    LOG.info('Done')


