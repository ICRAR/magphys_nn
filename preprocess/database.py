# Magphys neural network preprocessor
# Date: 7/12/2015
# Author: Samuel Foster
# Organisation: ICRAR
#
# SQLAlchemy related functions for the preprocessor
#
import config
from sqlalchemy.engine import create_engine

from sqlalchemy import MetaData, Table, Column, Integer, String, Float, TIMESTAMP, ForeignKey, BigInteger, Numeric
from sqlalchemy import select
from sqlalchemy import func
from logger import config_logger

LOG = config_logger(__name__)

engine = create_engine(config.DB_LOGIN)
connection = engine.connect()

MAGPHYS_NN_METADATA = MetaData()

NN_TRAIN = Table('nn_train',
                 MAGPHYS_NN_METADATA,
                 Column('train_id', Integer, primary_key=True, autoincrement=True),
                 Column('run_id', BigInteger),
                 Column('filename', String),
                 Column('last_updated', TIMESTAMP),
                 Column('type', String),  # Median or Best Fit
                 Column('input', Integer, ForeignKey('input.input_id')),
                 Column('input_snr', Integer, ForeignKey('input_snr.input_snr_id')),
                 Column('output', Integer, ForeignKey('median_output.median_output_id')),
                 )

INPUT = Table('input',
              MAGPHYS_NN_METADATA,
              Column('input_id', Integer, primary_key=True, autoincrement=True),
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


def add_to_db(input, input_snr, output, details):
    transaction = connection.begin()

    input_key = connection.execute(INPUT.insert().values(fuv=input['fuv'],
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
    
    input_snr_key = connection.execute(INPUT.insert().values(fuv=input_snr['fuv'],
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

    connection.execute(NN_TRAIN.insert().values(run_id=details['run_id'],
                             filename=details['filename'],
                             last_updated=details['last_updated'],
                             type=details['type'],
                             input=input_key,
                             input_snr=input_snr_key,
                             output=output_key
                             ))
    transaction.commit()


def exists_in_db(filename):
    count = connection.execute(select([func.count(NN_TRAIN)]).where(NN_TRAIN.c.filename == filename)).first()[0]
    if count == 0:
        return False

    return True

if __name__ == '__main__':

    LOG.info('Performing database clense.')
    transaction = connection.begin()
    for table in reversed(MAGPHYS_NN_METADATA.sorted_tables):
        connection.execute(table.delete())
    transaction.commit()
    LOG.info('Done')


