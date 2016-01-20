# Magphys neural network preprocessor
# Date: 7/12/2015
# Author: Samuel Foster
# Organisation: ICRAR
#
# Determines how data should be formatted for input to the NN.
#


def map_inputrow2list_Jy_nosigma(row, input_filter_types):
    if input_filter_types is None:
        out_list = [None] * 21

        out_list[0] = row['fuv']
        out_list[1] = row['nuv']
        out_list[2] = row['u']
        out_list[3] = row['g']
        out_list[4] = row['r']
        out_list[5] = row['i']
        out_list[6] = row['z']
        out_list[7] = row['Z_']
        out_list[8] = row['Y']
        out_list[9] = row['J']
        out_list[10] = row['H']
        out_list[11] = row['K']
        out_list[12] = row['WISEW1']
        out_list[13] = row['WISEW2']
        out_list[14] = row['WISEW3']
        out_list[15] = row['WISEW4']
        out_list[16] = row['PACS100']
        out_list[17] = row['PACS160']
        out_list[18] = row['SPIRE250']
        out_list[19] = row['SPIRE350']
        out_list[20] = row['SPIRE500']

    else:
        out_list = []
        if 'optical' in input_filter_types:
            out_list.append(row['u'])
            out_list.append(row['g'])
            out_list.append(row['r'])
            out_list.append(row['i'])
            out_list.append(row['z'])
            out_list.append(row['Z_'])
            out_list.append(row['Y'])
            out_list.append(row['J'])
            out_list.append(row['H'])
            out_list.append(row['K'])

        if 'ir' in input_filter_types:
            out_list.append(row['WISEW1'])
            out_list.append(row['WISEW2'])
            out_list.append(row['WISEW3'])
            out_list.append(row['WISEW4'])
            out_list.append(row['PACS100'])
            out_list.append(row['PACS160'])
            out_list.append(row['SPIRE250'])
            out_list.append(row['SPIRE350'])
            out_list.append(row['SPIRE500'])

        if 'uv' in input_filter_types:
            out_list.append(row['fuv'])
            out_list.append(row['nuv'])

    return out_list


def map_inputrow2list_nosigma(row, input_filter_types):
    if input_filter_types is None:
        out_list = [None] * 21
        # Normal values, suitable for NN input
        out_list[0] = row['fuv']
        out_list[1] = row['nuv']
        out_list[2] = row['u']
        out_list[3] = row['g']
        out_list[4] = row['r']
        out_list[5] = row['i']
        out_list[6] = row['z']
        out_list[7] = row['Z_']
        out_list[8] = row['Y']
        out_list[9] = row['J']
        out_list[10] = row['H']
        out_list[11] = row['K']
        out_list[12] = row['WISEW1']
        out_list[13] = row['WISEW2']
        out_list[14] = row['WISEW3']
        out_list[15] = row['WISEW4']
        out_list[16] = row['PACS100']
        out_list[17] = row['PACS160']
        out_list[18] = row['SPIRE250']
        out_list[19] = row['SPIRE350']
        out_list[20] = row['SPIRE500']

    else:
        out_list = []
        if 'optical' in input_filter_types:
            out_list.append(row['u'])
            out_list.append(row['g'])
            out_list.append(row['r'])
            out_list.append(row['i'])
            out_list.append(row['z'])
            out_list.append(row['Z_'])
            out_list.append(row['Y'])
            out_list.append(row['J'])
            out_list.append(row['H'])
            out_list.append(row['K'])

        if 'ir' in input_filter_types:
            out_list.append(row['WISEW1'])
            out_list.append(row['WISEW2'])
            out_list.append(row['WISEW3'])
            out_list.append(row['WISEW4'])
            out_list.append(row['PACS100'])
            out_list.append(row['PACS160'])
            out_list.append(row['SPIRE250'])
            out_list.append(row['SPIRE350'])
            out_list.append(row['SPIRE500'])

        if 'uv' in input_filter_types:
            out_list.append(row['fuv'])
            out_list.append(row['nuv'])

    return out_list


def map_inputrow2list_Jy(row, input_filter_types):
    if input_filter_types is None:
        out_list = [None] * 42

        out_list[0] = row['fuv']
        out_list[2] = row['nuv']
        out_list[4] = row['u']
        out_list[6] = row['g']
        out_list[8] = row['r']
        out_list[10] = row['i']
        out_list[12] = row['z']
        out_list[14] = row['Z_']
        out_list[16] = row['Y']
        out_list[18] = row['J']
        out_list[20] = row['H']
        out_list[22] = row['K']
        out_list[24] = row['WISEW1']
        out_list[26] = row['WISEW2']
        out_list[28] = row['WISEW3']
        out_list[30] = row['WISEW4']
        out_list[32] = row['PACS100']
        out_list[34] = row['PACS160']
        out_list[36] = row['SPIRE250']
        out_list[38] = row['SPIRE350']
        out_list[40] = row['SPIRE500']

        # SNR values, suitable for NN input
        out_list[1] = row['fuv_snr']
        out_list[3] = row['nuv_snr']
        out_list[5] = row['u_snr']
        out_list[7] = row['g_snr']
        out_list[9] = row['r_snr']
        out_list[11] = row['i_snr']
        out_list[13] = row['z_snr']
        out_list[15] = row['Z__snr']
        out_list[17] = row['Y_snr']
        out_list[19] = row['J_snr']
        out_list[21] = row['H_snr']
        out_list[23] = row['K_snr']
        out_list[25] = row['WISEW1_snr']
        out_list[27] = row['WISEW2_snr']
        out_list[29] = row['WISEW3_snr']
        out_list[31] = row['WISEW4_snr']
        out_list[33] = row['PACS100_snr']
        out_list[35] = row['PACS160_snr']
        out_list[37] = row['SPIRE250_snr']
        out_list[39] = row['SPIRE350_snr']
        out_list[41] = row['SPIRE500_snr']

    else:
        out_list = []
        if 'optical' in input_filter_types:
            out_list.append(row['u'])
            out_list.append(row['u_snr'])
            out_list.append(row['g'])
            out_list.append(row['g_snr'])
            out_list.append(row['r'])
            out_list.append(row['r_snr'])
            out_list.append(row['i'])
            out_list.append(row['i_snr'])
            out_list.append(row['z'])
            out_list.append(row['z_snr'])
            out_list.append(row['Z_'])
            out_list.append(row['Z__snr'])
            out_list.append(row['Y'])
            out_list.append(row['Y_snr'])
            out_list.append(row['J'])
            out_list.append(row['J_snr'])
            out_list.append(row['H'])
            out_list.append(row['H_snr'])
            out_list.append(row['K'])
            out_list.append(row['K_snr'])

        if 'ir' in input_filter_types:
            out_list.append(row['WISEW1'])
            out_list.append(row['WISEW1_snr'])
            out_list.append(row['WISEW2'])
            out_list.append(row['WISEW2_snr'])
            out_list.append(row['WISEW3'])
            out_list.append(row['WISEW3_snr'])
            out_list.append(row['WISEW4'])
            out_list.append(row['WISEW4_snr'])
            out_list.append(row['PACS100'])
            out_list.append(row['PACS100_snr'])
            out_list.append(row['PACS160'])
            out_list.append(row['PACS160_snr'])
            out_list.append(row['SPIRE250'])
            out_list.append(row['SPIRE250_snr'])
            out_list.append(row['SPIRE350'])
            out_list.append(row['SPIRE350_snr'])
            out_list.append(row['SPIRE500'])
            out_list.append(row['SPIRE500_snr'])

        if 'uv' in input_filter_types:
            out_list.append(row['fuv'])
            out_list.append(row['fuv_snr'])
            out_list.append(row['nuv'])
            out_list.append(row['nuv_snr'])

    return out_list


def map_inputrow2list(row, input_filter_types):
    if input_filter_types is None:
        out_list = [None] * 42
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
        out_list[10] = row['i']
        out_list[12] = row['z']
        out_list[14] = row['Z_']
        out_list[16] = row['Y']
        out_list[18] = row['J']
        out_list[20] = row['H']
        out_list[22] = row['K']
        out_list[24] = row['WISEW1']
        out_list[26] = row['WISEW2']
        out_list[28] = row['WISEW3']
        out_list[30] = row['WISEW4']
        out_list[32] = row['PACS100']
        out_list[34] = row['PACS160']
        out_list[36] = row['SPIRE250']
        out_list[38] = row['SPIRE350']
        out_list[40] = row['SPIRE500']

        # SNR values, suitable for NN input
        out_list[1] = row['fuv_snr']
        out_list[3] = row['nuv_snr']
        out_list[5] = row['u_snr']
        out_list[7] = row['g_snr']
        out_list[9] = row['r_snr']
        out_list[11] = row['i_snr']
        out_list[13] = row['z_snr']
        out_list[15] = row['Z__snr']
        out_list[17] = row['Y_snr']
        out_list[19] = row['J_snr']
        out_list[21] = row['H_snr']
        out_list[23] = row['K_snr']
        out_list[25] = row['WISEW1_snr']
        out_list[27] = row['WISEW2_snr']
        out_list[29] = row['WISEW3_snr']
        out_list[31] = row['WISEW4_snr']
        out_list[33] = row['PACS100_snr']
        out_list[35] = row['PACS160_snr']
        out_list[37] = row['SPIRE250_snr']
        out_list[39] = row['SPIRE350_snr']
        out_list[41] = row['SPIRE500_snr']

    else:
        out_list = []
        if 'optical' in input_filter_types:
            out_list.append(row['u'])
            out_list.append(row['u_snr'])
            out_list.append(row['g'])
            out_list.append(row['g_snr'])
            out_list.append(row['r'])
            out_list.append(row['r_snr'])
            out_list.append(row['i'])
            out_list.append(row['i_snr'])
            out_list.append(row['z'])
            out_list.append(row['z_snr'])
            out_list.append(row['Z_'])
            out_list.append(row['Z__snr'])
            out_list.append(row['Y'])
            out_list.append(row['Y_snr'])
            out_list.append(row['J'])
            out_list.append(row['J_snr'])
            out_list.append(row['H'])
            out_list.append(row['H_snr'])
            out_list.append(row['K'])
            out_list.append(row['K_snr'])

        if 'ir' in input_filter_types:
            out_list.append(row['WISEW1'])
            out_list.append(row['WISEW1_snr'])
            out_list.append(row['WISEW2'])
            out_list.append(row['WISEW2_snr'])
            out_list.append(row['WISEW3'])
            out_list.append(row['WISEW3_snr'])
            out_list.append(row['WISEW4'])
            out_list.append(row['WISEW4_snr'])
            out_list.append(row['PACS100'])
            out_list.append(row['PACS100_snr'])
            out_list.append(row['PACS160'])
            out_list.append(row['PACS160_snr'])
            out_list.append(row['SPIRE250'])
            out_list.append(row['SPIRE250_snr'])
            out_list.append(row['SPIRE350'])
            out_list.append(row['SPIRE350_snr'])
            out_list.append(row['SPIRE500'])
            out_list.append(row['SPIRE500_snr'])

        if 'uv' in input_filter_types:
            out_list.append(row['fuv'])
            out_list.append(row['fuv_snr'])
            out_list.append(row['nuv'])
            out_list.append(row['nuv_snr'])

    return out_list


def map_outputrow2list(row):
    out_list = [None] * 32

    """
    out_list[0] = row['tau_V']
    out_list[1] = row['Mstars']
    out_list[2] = row['xi_Wtot']
    out_list[3] = row['xi_PAHtot']
    out_list[4] = row['f_muSFH']
    out_list[5] = row['T_CISM']
    out_list[6] = row['Ldust']
    out_list[7] = row['mu_parameter']
    out_list[8] = row['xi_Ctot']
    out_list[9] = row['f_muIR']
    out_list[10] = row['T_WBC']
    out_list[11] = row['tau_VISM']
    out_list[12] = row['sSFR_0_1Gyr']
    out_list[13] = row['Mdust']
    out_list[14] = row['xi_MIRtot']

    # Median output values
    """
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
    """
    """

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
    out_list = [None] * 21

    out_list[0] = row['fuv']
    out_list[1] = row['nuv']
    out_list[2] = row['u']
    out_list[3] = row['g']
    out_list[4] = row['r']
    out_list[5] = row['i']
    out_list[6] = row['z']
    out_list[7] = row['Z_']
    out_list[8] = row['Y']
    out_list[9] = row['J']
    out_list[10] = row['H']
    out_list[11] = row['K']
    out_list[12] = row['WISEW1']
    out_list[13] = row['WISEW2']
    out_list[14] = row['WISEW3']
    out_list[15] = row['WISEW4']
    out_list[16] = row['PACS100']
    out_list[17] = row['PACS160']
    out_list[18] = row['SPIRE250']
    out_list[19] = row['SPIRE350']
    out_list[20] = row['SPIRE500']

    return out_list


def map_best_fit_model_output2list(row):
    out_list = [None] * 4

    out_list[0] = row['i_sfh']
    out_list[1] = row['i_ir']
    out_list[2] = row['chi2']
    out_list[3] = row['redshift']

    return out_list
