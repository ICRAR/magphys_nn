Begin Transaction;
Create  TABLE MAIN.[Temp_951081800](
   [input_id] bigint PRIMARY KEY UNIQUE NOT NULL
  ,[fuv] float
  ,[nuv] float
  ,[u] float
  ,[g] float
  ,[r] float
  ,[y] float
  ,[z] float
  ,[Z_] float
  ,[Y_] float
  ,[J] float
  ,[H] float
  ,[K] float
  ,[WISEW1] float
  ,[WISEW2] float
  ,[WISEW3] float
  ,[WISEW4] float
  ,[PACS100] float
  ,[PACS160] float
  ,[SPIRE250] float
  ,[SPIRE350] float
  ,[SPIRE500] float

) ;
Insert Into MAIN.[Temp_951081800] ([input_id],[fuv],[nuv],[u],[g],[r],[y],[z],[Z_],[Y_],[J],[H],[K],[WISEW1],[WISEW2],[WISEW3],[WISEW4],[PACS100],[PACS160],[SPIRE250],[SPIRE350],[SPIRE500])
  Select [input_id],[fuv],[nuv],[u],[g],[r],[y],[z],[Z_],[Y_],[J],[H],[K],[WISEW1],[WISEW2],[WISEW3],[WISEW4],[PACS100],[PACS160],[SPIRE250],[SPIRE350],[SPIRE500] From MAIN.[input];
Drop Table MAIN.[input];
Alter Table MAIN.[Temp_951081800] Rename To [input];

Commit Transaction;

Begin Transaction;
Create  TABLE MAIN.[Temp_586654629](
   [median_output_id] integer
  ,[ager]
  ,[tau_V]
  ,[agem]
  ,[tlastb]
  ,[Mstars]
  ,[sfr29]
  ,[xi_PAHtot]
  ,[f_muSFH]
  ,[fb17]
  ,[fb16]
  ,[T_CISM]
  ,[Ldust]
  ,[mu_parameter]
  ,[xi_Ctot]
  ,[f_muIR]
  ,[fb18]
  ,[fb19]
  ,[T_WBC]
  ,[SFR_0_1Gyr]
  ,[fb29]
  ,[sfr17]
  ,[sfr16]
  ,[sfr19]
  ,[sfr18]
  ,[tau_VISM]
  ,[sSFR_0_1Gyr]
  ,[metalicity_Z_Z0]
  ,[Mdust]
  ,[xi_MIRtot]
  ,[tform]
  ,[gamma]
  , Primary Key(median_output_id)
) ;
Insert Into MAIN.[Temp_586654629] ([median_output_id],[ager],[tau_V],[agem],[tlastb],[Mstars],[sfr29],[xi_PAHtot],[f_muSFH],[fb17],[fb16],[T_CISM],[Ldust],[mu_parameter],[xi_Ctot],[f_muIR],[fb18],[fb19],[T_WBC],[SFR_0_1Gyr],[fb29],[sfr17],[sfr16],[sfr19],[sfr18],[tau_VISM],[sSFR_0_1Gyr],[metalicity_Z_Z0],[Mdust],[xi_MIRtot],[tform],[gamma])
  Select [median_output_id],[ager],[tau_V],[agem],[tlastb],[Mstars],[sfr29],[xi_PAHtot],[f_muSFH],[fb17],[fb16],[T_CISM],[Ldust],[mu_parameter],[xi_Ctot],[f_muIR],[fb18],[fb19],[T_WBC],[SFR_0_1Gyr],[fb29],[sfr17],[sfr16],[sfr19],[sfr18],[tau_VISM],[sSFR_0_1Gyr],[metalicity_Z_Z0],[Mdust],[xi_MIRtot],[tform],[gamma] From MAIN.[median_output];
Drop Table MAIN.[median_output];
Alter Table MAIN.[Temp_586654629] Rename To [median_output];

Commit Transaction;

Begin Transaction;
Create  TABLE MAIN.[nn_train](
   [train_id] integer NOT NULL
  ,[run_id] varchar(30) NOT NULL
  ,[filename] varchar(200) NOT NULL
  ,[last_updated] timestamp NOT NULL
  ,[type] varchar(10) NOT NULL
  ,[input] integer NOT NULL REFERENCES [input] ([input_id])
  ,[input_snr] integer NOT NULL REFERENCES [input] ([input_id])
  ,[output] integer NOT NULL REFERENCES [median_output] ([median_output_id])
  , Primary Key(train_id)
) ;


Commit Transaction;