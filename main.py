import pandas as pd
import numpy as np

import wbgapi as wb
import requests

print(wb.series.info(q='MIL'))
vars_dict = {'code': ['NY.GDP.MKTP.KD.ZG', 'NY.GDP.MKTP.KD','NY.GDP.PCAP.KD',
                      'NE.CON.GOVT.ZS', 'NE.CON.PRVT.ZS', 'GC.NFN.TOTL.GD.ZS','NY.GDP.DEFL.KD.ZG', 'EG.FEC.RNEW.ZS',
                      'SL.UEM.TOTL.NE.ZS','SL.EMP.TOTL.SP.NE.ZS','HD.HCI.OVRL','SL.AGR.EMPL.ZS','SL.EMP.VULN.ZS',
                      # environment:
                      'EN.POP.SLUM.UR.ZS','ER.H2O.INTR.PC',
                      'NE.IMP.GNFS.ZS','NE.EXP.GNFS.ZS','BX.KLT.DINV.CD.WD','GC.DOD.TOTL.GD.ZS','FR.INR.RINR',
                      'SI.POV.DDAY','SI.POV.LMIC','SI.POV.UMIC','SI.POV.MDIM','SI.POV.MDIM.XQ',
                      'SI.POV.GINI','BN.CAB.XOKA.GD.ZS','SP.POP.TOTL','EN.POP.DNST','SP.POP.GROW',
                      'SP.DYN.LE00.IN','SP.DYN.CDRT.IN', 'SH.DYN.MORT','SH.DTH.COMM.ZS',
                      # agric:
                      'EG.ELC.RNEW.ZS','AG.LND.AGRI.ZS','NV.AGR.TOTL.ZS','AG.LND.ARBL.ZS','SP.RUR.TOTL.ZS',
                      # climate change:
                      'EG.ELC.ACCS.ZS','EG.USE.ELEC.KH.PC',
                      # energy & mining:
                      'TX.VAL.FUEL.ZS.UN','TX.VAL.MMTL.ZS.UN',
                      'EN.ATM.CO2E.PC','EN.CLC.GHGR.MT.CE','NV.IND.TOTL.ZS','NV.SRV.EMPL.KD',
                      'SE.ENR.PRSC.FM.ZS','SE.ADT.LITR.ZS','SE.TER.CUAT.BA.ZS','VC.IHR.PSRC.P5','SP.POP.SCIE.RD.P6',
                      #millitary:
                      'MS.MIL.XPND.CD','MS.MIL.XPND.CN','MS.MIL.XPND.GD.ZS', 'MS.MIL.XPND.ZS'],
            'label': ['gdp_real_gwt', 'gdp_real_us_fixed','gdp_per_capita',
                      'gdp_pp_govt', 'gdp_pp_private', 'investment', 'inflation_pp', 'renew_energy',
                      'unemployment','employment','HCI','agric_employment','vulnerable_employment',
                      # environment:
                      'pop_slums','renew_freshwater',
                      'imports','exports','foreign_inv','govt_debt','real_interest_rate',
                      'poverty_1.90','poverty_3.20','poverty_5.50','poverty_multidim','poverty_mult_index',
                      'gini_index','cab','population','pop_density','pop_growth',
                      'life_expectancy','death_rate', 'child_mortality','cause_of_death',
                      # agric:
                      'renewable_energy_output','agric_land','agff_gdp','arable_land','rural_pop',
                      # climate change:
                      'electricity_access','power_consumption',
                      # energy & mining:
                      'fuel_exports','metal_exports',
                      'co2_emissions','ghg_emissions','industry_gdp','service_value_added',
                      'school_enroll','literacy','bachelor','homicide','research',
                      'military_expenditure_usd','military_expenditure_lcu','military_expenditure_gdp', 'military_expenditure_general']}

vars_df = pd.DataFrame(vars_dict)
vars_df = vars_df.assign(definition='')
for i in range(0,len(vars_df)):
    vars_df.iloc[i,2] = wb.series.get(id=vars_df.iloc[i,0])['value']
print(vars_df)


def vert_df(gdp_df, name):
    gdp_df.columns = gdp_df.columns.str.replace('YR','')
    gdp_df = gdp_df.reset_index()
    gdp_df.drop('Country', inplace=True, axis=1)
    year_drop = list(range(1960, 1999))
    year_drop = [*map(str,year_drop)]
    gdp_df.drop(year_drop, inplace=True, axis=1)
    gdp_df = gdp_df.melt(id_vars = ['economy'], var_name = 'Year', value_name = name)
    gdp_df.Year = pd.to_numeric(gdp_df.Year)
    return gdp_df


gdp_df = vert_df(wb.data.DataFrame(vars_df.iloc[0,0], labels=True), vars_df.iloc[0,1])
for i in range(1, len(vars_df)):
    wbcode, wblabel = vars_df.iloc[i,0], vars_df.iloc[i,1]
    new_data = vert_df(wb.data.DataFrame(wbcode, labels=True), wblabel)
    gdp_df = pd.merge(gdp_df, new_data, how='left', on=['economy', 'Year'])
gdp_df.rename(columns = {'economy':'ISO3'}, inplace = True)
gdp_df.to_csv('info.csv', index=False)