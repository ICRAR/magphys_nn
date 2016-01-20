#
#    (c) UWA, The University of Western Australia
#    M468/35 Stirling Hwy
#    Perth WA 6009
#    Australia
#
#    Copyright by UWA, 2012-2015
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#    MA 02111-1307  USA
#
"""
The tables used
"""
from sqlalchemy import MetaData, Table, Column, BigInteger, ForeignKey, Integer, String, select, Float, Numeric

POGS_METADATA = MetaData()

PARAMETER = Table(
    'parameter',
    POGS_METADATA,
    Column('parameter_id', Integer, primary_key=True, autoincrement=False),
    Column('description', String(250)),
)

STATISTIC = Table(
    'statistic',
    POGS_METADATA,
    Column('statistic_id', Integer, primary_key=True, autoincrement=False),
    Column('description', String(250)),
)

GALAXY = Table(
    'galaxy',
    POGS_METADATA,
    Column('galaxy_id', BigInteger, primary_key=True, autoincrement=False),
    Column('run_id', BigInteger),
    Column('name', String(128)),
    Column('dimension_x', Integer),
    Column('dimension_y', Integer),
    Column('dimension_z', Integer),
    Column('redshift', Numeric(7, 5)),
    Column('galaxy_type', String(10)),
    Column('ra_cent', Float),
    Column('dec_cent', Float),
    Column('pixel_count', Integer),
    Column('stored_count', Integer),
)

PARAMETER_STATISTIC = Table(
    'parameter_statistic',
    POGS_METADATA,
    Column('parameter_statistic_id', BigInteger, primary_key=True, autoincrement=True),
    Column('parameter_id', Integer, ForeignKey('parameter.parameter_id')),
    Column('statistic_id', Integer, ForeignKey('statistic.statistic_id')),
)

GALAXY_DETAIL = Table(
    'galaxy_detail',
    POGS_METADATA,
    Column('galaxy_detail_id', BigInteger, primary_key=True, autoincrement=True),
    Column('galaxy_id', BigInteger, nullable=False),
    Column('count', BigInteger, nullable=False),
    Column('min_value', Float, nullable=False),
    Column('max_value', Float, nullable=False),
    Column('percentile_10', Float, nullable=False),
    Column('percentile_20', Float, nullable=False),
    Column('percentile_25', Float, nullable=False),
    Column('percentile_30', Float, nullable=False),
    Column('percentile_40', Float, nullable=False),
    Column('percentile_50', Float, nullable=False),
    Column('percentile_60', Float, nullable=False),
    Column('percentile_70', Float, nullable=False),
    Column('percentile_75', Float, nullable=False),
    Column('percentile_80', Float, nullable=False),
    Column('percentile_90', Float, nullable=False),
    Column('parameter_id', Integer, ForeignKey('parameter.parameter_id'), nullable=False),
    Column('statistic_id', Integer, ForeignKey('statistic.statistic_id'), nullable=False),
)

ORIGINAL_VALUE = Table(
    'original_value',
    POGS_METADATA,
    Column('original_value_id', BigInteger, primary_key=True, autoincrement=True),
    Column('galaxy_id', BigInteger, ForeignKey('galaxy.galaxy_id'), nullable=False),
    Column('filter_number', Integer, nullable=False),
    Column('x', Integer, nullable=False),
    Column('y', Integer, nullable=False),
    Column('value', Float, nullable=False),
    Column('sigma', Float, nullable=False)
)

FILTER = Table(
    'filter',
    POGS_METADATA,
    Column('filter_id', Integer, primary_key=True),
    Column('name', String(30)),
    Column('eff_lambda', Numeric(10, 4)),
    Column('filter_number', Integer),
    Column('ultraviolet', Integer),
    Column('optical', Integer),
    Column('infrared', Integer),
    Column('label', String(20))
)

MASK = Table(
    'mask',
    POGS_METADATA,
    Column('mask_id', BigInteger, primary_key=True, autoincrement=True),
    Column('galaxy_id', BigInteger, ForeignKey('galaxy.galaxy_id'), nullable=False),
    Column('min_x', Integer, nullable=False),
    Column('max_x', Integer, nullable=False),
    Column('min_y', Integer, nullable=False),
    Column('max_y', Integer, nullable=False),
)

MASK_POINT = Table(
    'mask_point',
    POGS_METADATA,
    Column('mask_point_id', BigInteger, primary_key=True, autoincrement=True),
    Column('galaxy_id', BigInteger, ForeignKey('galaxy.galaxy_id'), nullable=False),
    Column('x', Integer, nullable=False),
    Column('y', Integer, nullable=False),
)

STEP_DONE = Table(
    'step_done',
    POGS_METADATA,
    Column('step_done_id', BigInteger, primary_key=True, autoincrement=True),
    Column('galaxy_id', BigInteger, ForeignKey('galaxy.galaxy_id'), nullable=False),
    Column('step', String(1000), nullable=False),
)

MAGPHYS_NN_METADATA = MetaData()

NN_TRAIN = Table('nn_train',
                 MAGPHYS_NN_METADATA,
                 Column('galaxy_id', Integer, primary_key=True),
                 Column('pix_x', Integer, primary_key=True),
                 Column('pix_y', Integer, primary_key=True),
                 Column('redshift', Float),
                 Column('run_id', String(30)),
                 Column('fit_filename', String(200)),
                 Column('run_filename', String(200)),
                 )

INPUT = Table('input',
              MAGPHYS_NN_METADATA,
              Column('galaxy_id', Integer, ForeignKey('nn_train.galaxy_id'), primary_key=True),
              Column('pix_x', Integer, ForeignKey('nn_train.pix_x'), primary_key=True),
              Column('pix_y', Integer, ForeignKey('nn_train.pix_y'), primary_key=True),
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
                 Column('galaxy_id', Integer, ForeignKey('nn_train.galaxy_id'), primary_key=True),
                 Column('pix_x', Integer, ForeignKey('nn_train.pix_x'), primary_key=True),
                 Column('pix_y', Integer, ForeignKey('nn_train.pix_y'), primary_key=True),
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
                      Column('galaxy_id', Integer, ForeignKey('nn_train.galaxy_id'), primary_key=True),
                      Column('pix_x', Integer, ForeignKey('nn_train.pix_x'), primary_key=True),
                      Column('pix_y', Integer, ForeignKey('nn_train.pix_y'), primary_key=True),
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