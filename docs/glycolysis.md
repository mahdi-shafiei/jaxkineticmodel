## Custom ODE models
Not all kinetic model can be drafted only from stoichiometry. There are many empirical laws that might influence reactions in the system of ODEs. Below, we present a reimplemented version **[1]** of a glycolysis model in Jax with a manually set up of ODEs. 


### Glycolysis model
The model consists of 29 metabolites (ODEs), 38 reactions, with in total 141 parameters. The model can also be found in [Github/models/manual_implementations](https://github.com/AbeelLab/jaxkineticmodel/tree/main/models/manual_implementations/glycolysis) and loaded from there.

#### Rate laws
Start by loading the required kinetic mechanisms

```python 
from jaxkineticmodel.kinetic_mechanisms.JaxKineticMechanisms import  *
from jaxkineticmodel.kinetic_mechanisms.JaxKineticMechanismsCustom import *
from jaxkineticmodel.kinetic_mechanisms.JaxKineticModifiers import *
```
Now, we define all reactions in the model. Note that names in substrates and products should match actual species names in the initial conditions. First, lower glycolysis.

```python
#upper glycolysis (GLT, HXK, PGI, PFK,ALD,TPI)
v_GLT=Jax_Facilitated_Diffusion(substrate_extracellular='ECglucose',product_intracellular='ICglucose',vmax="p_GLT_VmGLT",km_internal='p_GLT_KmGLTGLCi',km_external='p_GLT_KmGLTGLCo')

v_HXK=Jax_MM_Rev_BiBi_w_Inhibition(substrate1='ICATP',substrate2="ICglucose",product1="ICADP",product2="ICATP",modifier="ICT6P",vmax="p_HXK_Vmax", k_equilibrium="p_HXK1_Keq", km_substrate1="p_HXK1_Katp", 
                                           km_substrate2="p_HXK1_Kglc",
                                           km_product1="p_HXK1_Kadp", km_product2="p_HXK1_Kg6p", ki_inhibitor="p_HXK1_Kt6p")
v_PGI=Jax_MM_Rev_UniUni(substrate='ICG6P',product='ICF6P',vmax='p_PGI1_Vmax',k_equilibrium='p_PGI1_Keq',km_substrate='p_PGI1_Kg6p',km_product='p_PGI1_Kf6p')


v_PFK=Jax_PFK(substrate1="ICF6P",substrate2="ICATP",product1="ICFBP",modifier="ICAMP",vmax="p_PFK_Vmax",kr_F6P="p_PFK_Kf6p", kr_ATP="p_PFK_Katp", gr="p_PFK_gR",c_ATP="p_PFK_Catp", ci_ATP="p_PFK_Ciatp", ci_AMP="p_PFK_Camp", 
                           ci_F26BP="p_PFK_Cf26bp", ci_F16BP="p_PFK_Cf16bp", l="p_PFK_L", 
                           kATP="p_PFK_Kiatp", kAMP="p_PFK_Kamp", F26BP ="p_PFK_F26BP",
                           kF26BP = "p_PFK_Kf26bp", kF16BP = "p_PFK_Kf16bp")
v_ALD=Jax_MM_Rev_UniBi(substrate='ICFBP',product1='ICGAP',product2='ICDHAP',vmax="p_FBA1_Vmax", k_equilibrium="p_FBA1_Keq", km_substrate="p_FBA1_Kf16bp",
                                    km_product1="p_FBA1_Kglyceral3p", km_product2="p_FBA1_Kdhap" )
v_TPI1=Jax_MM_Rev_UniUni(substrate="ICDHAP",product="ICGAP",vmax="p_TPI1_Vmax",k_equilibrium="p_TPI1_Keq", km_substrate="p_TPI1_Kdhap",
                                km_product="p_TPI1_Kglyceral3p")
```
Trehalose cycle
```python
#trehalose cycle (NTH1, PGM1, TPS2, TPS1)
v_NTH1=Jax_MM_Irrev_Uni(substrate='ICtreh',vmax='p_NTH1_Vmax',km_substrate='p_NTH1_Ktre')


v_PGM1=Jax_MM_Rev_UniUni(substrate='ICG1P',product='ICG6P',vmax='p_PGM1_Vmax',k_equilibrium='p_PGM1_Keq',km_substrate='p_PGM1_Kg1p',km_product='p_PGM1_Kg6p') #to do v_TPS1 for 2nd rate law

# inhibitor_TPS1=SimpleInhibitor(k_I='p_TPS1_Kpi')
activator_TPS1=SimpleActivator(k_A="p_TPS1_KmF6P")
v_TPS1=Jax_MM_Irrev_Bi_w_Modifiers(substrate1="ICG6P",substrate2="ICG1P",modifiers_list=['ICF6P'],vmax="p_TPS1_Vmax",km_substrate1="p_TPS1_Kg6p",
                                  km_substrate2="p_TPS1_Kudp_glc",modifiers=[activator_TPS1])

# v_TPS1=Jax_MM_Irrev_Bi(substrate1="ICG6P",substrate2="ICG1P",vmax="p_TPS1_Vmax",km_substrate1="p_TPS1_Kg6p",km_substrate2="p_TPS1_Kudp_glc")
v_TPS2=Jax_MM_Irrev_Bi_w_Inhibition(substrate="ICT6P",product="ICPHOS",vmax="p_TPS2_Vmax", km_substrate1="p_TPS2_Kt6p", ki="p_TPS2_Kpi")

```
Lower glycolysis
```python 
# Lower glycolysis (GAPDH, PGK,PGM,ENO, PDC, ADH)
v_GAPDH=Jax_MM_Ordered_Bi_Tri(substrate1="ICGAP",substrate2="ICNAD",substrate3="ICPHOS",product1="ICBPG",product2="ICNADH",
                              vmax="p_GAPDH_Vmax", k_equilibrium="p_TDH1_Keq", km_substrate1="p_TDH1_Kglyceral3p",
                                      km_substrate2="p_TDH1_Knad", ki="p_TDH1_Kpi", 
                                      km_product1="p_TDH1_Kglycerate13bp", km_product2="p_TDH1_Knadh") #might exchange this mechanism by a BiBi mechanism, since modeling Phos is a bit too much
v_PGK=Jax_MM_Rev_BiBi(substrate1="ICBPG",substrate2="ICADP",product1="IC3PG",product2="ICATP",vmax="p_PGK_VmPGK",k_equilibrium="p_PGK_KeqPGK", 
                               km_substrate1="p_PGK_KmPGKBPG", km_substrate2="p_PGK_KmPGKADP", 
                               km_product1="p_PGK_KmPGKP3G", km_product2="p_PGK_KmPGKATP")
v_PGM=Jax_MM_Rev_UniUni(substrate="IC3PG",product="IC2PG",vmax="p_PGM_Vm", k_equilibrium="p_PGM_Keq",
                                 km_substrate="p_PGM_K3pg", km_product="p_PGM_K2pg")
v_ENO=Jax_MM_Rev_UniUni(substrate="IC2PG",product="ICPEP",vmax="p_ENO1_Vm",k_equilibrium="p_ENO1_Keq",
                                km_substrate="p_ENO1_K2pg", km_product="p_ENO1_Kpep") 
v_PYK1=Jax_Hill_Irreversible_Bi_Activation(substrate1="ICPEP",substrate2="ICADP",activator="ICFBP",product="ICATP",vmax="p_PYK1_Vm", hill="p_PYK1_hill",
                                                  k_substrate1="p_PYK1_Kpep", k_substrate2="p_PYK1_Kadp",
                                                  k_product="p_PYK1_Katp", k_activator="p_PYK1_Kf16bp", l="p_PYK1_L")                                                                                  
v_PDC=Jax_Hill_Irreversible_Inhibition(substrate="ICPYR",inhibitor="ICPHOS",vmax="p_PDC1_Vmax",k_half_substrate="p_PDC1_Kpyr",
                                               hill="p_PDC1_hill", ki="p_PDC1_Kpi")
v_ADH = Jax_ADH(NAD="ICNAD",ETOH="ICETOH",NADH="ICNADH",ACE="ICACE",
    vmax='p_ADH_VmADH',
    k_equilibrium='p_ADH_KeqADH',km_substrate1='p_ADH_KiADHNAD',
    km_substrate2='p_ADH_KmADHETOH',km_product1='p_ADH_KmADHACE',
    km_product2='p_ADH_KmADHNADH',ki_substrate1='p_ADH_KiADHNAD',
    ki_substrate2='p_ADH_KiADHETOH',ki_product1='p_ADH_KiADHACE',
    ki_product2='p_ADH_KiADHNADH',exprs_cor="p_ADH_ExprsCor")

v_EtohT = Jax_Diffusion(substrate="ICETOH",enzyme="f_ETOH_e", transport_coef="p_kETOHtransport")

```
Glycerophospholipid pathway

``` python
# Glycerophospholipid (G3PDH, HOR2, GlycTransport)
HOR2_inhibition_Pi=SimpleInhibitor(k_I="p_HOR2_Kpi")
v_HOR2=Jax_MM_Irrev_Uni_w_Modifiers(substrate="ICG3P",vmax="p_HOR2_Vmax",km_substrate="p_HOR2_Kglyc3p",modifiers_list=["ICPHOS"],modifiers=[HOR2_inhibition_Pi])
v_GlycT=Jax_Diffusion(substrate="ICglyc",enzyme="f_GLYCEROL_e",transport_coef="p_GlycerolTransport")

v_G3PDH=Jax_MM_Rev_BiBi_w_Activation(substrate1="ICDHAP",substrate2="ICNADH",product1="ICG3P",product2="ICNAD",modifiers=['ICFBP', 'ICATP', 'ICADP'],vmax="p_GPD1_Vmax", k_equilibrium="p_GPD1_Keq", 
                                             km_substrate1="p_GPD1_Kdhap", km_substrate2="p_GPD1_Knadh",
                                             km_product1="p_GPD1_Kglyc3p", km_product2="p_GPD1_Knad",
                                               ka1="p_GPD1_Kf16bp", ka2="p_GPD1_Katp", ka3="p_GPD1_Kadp")
                                               
                                               
                                             
```
Cofactor metabolism
```python

v_mitoNADH=Jax_MM(substrate="ICNADH",vmax="p_mitoNADHVmax",km="p_mitoNADHKm") #I think this can be replaced by Jax_MM_Irrev_Uni
v_ATPmito=Jax_MM_Irrev_Bi("ICADP","ICPHOS",vmax="p_mitoVmax",km_substrate1="p_mitoADPKm",
                                  km_substrate2="p_mitoPiKm")
v_ATPase=Jax_ATPase("ICATP","ICADP",ATPase_ratio="p_ATPase_ratio")
v_ADK=Jax_MA_Rev_Bi(substrate1="ICADP",substrate2="ICADP",product1="ICATP",product2="ICAMP",k_equilibrium="p_ADK1_Keq",k_fwd="p_ADK1_k")
v_VacPi=Jax_MA_Rev(substrate="ICPHOS",k="p_vacuolePi_k",steady_state_substrate="p_vacuolePi_steadyStatePi")
v_AMD1=Jax_Amd1(substrate="ICAMP",product="ICATP",modifier="ICPHOS",vmax="p_Amd1_Vmax", k50="p_Amd1_K50", ki="p_Amd1_Kpi", k_atp="p_Amd1_Katp")
v_ADE1312=Jax_MA_Irrev(substrate="ICIMP",k_fwd="p_Ade13_Ade12_k")

v_ISN1=Jax_MA_Irrev(substrate="ICIMP",k_fwd="p_Isn1_k")
v_PNP1=Jax_MA_Irrev(substrate="ICINO",k_fwd="p_Pnp1_k")
v_HPT1=Jax_MA_Irrev(substrate="ICHYP",k_fwd="p_Hpt1_k")
```

Sink reactions are required due to the fact metabolism is only partially modeled.
```python 
v_sinkG6P=Jax_MM_Sink(substrate='ICG6P',v_sink='poly_sinkG6P',km_sink='km_sinkG6P')
v_sinkF6P=Jax_MM_Sink(substrate='ICF6P',v_sink='poly_sinkF6P',km_sink='km_sinkF6P')
v_sinkGAP=Jax_MM_Sink(substrate="ICGAP",v_sink="poly_sinkGAP",km_sink="km_sinkGAP")
vsink3PGA=Jax_MM_Sink(substrate='IC3PG',v_sink='poly_sinkP3G',km_sink='km_sinkP3G')
vsinkACE=Jax_MM_Sink(substrate="ICACE",v_sink="poly_sinkACE",km_sink="km_sinkACE")
vsinkPYR=Jax_MM_Sink(substrate="ICPYR",v_sink="poly_sinkPYR", km_sink="km_sinkPYR")
vsinkPEP = Jax_MM_Sink(substrate="ICPEP",v_sink="poly_sinkPEP", km_sink="km_sinkPEP") #reverse sign in stoichiometry
```

#### Parameters
After defining all reactions, we load the parameter set that fitted the data in **[2]**. The trained parameters are here given below, but can also be found in the results directory.

TODO: make an online results directory that can be viewed.



#### Custom ODE system
We can now set up the ODE system, but note that it is not an ODE purely based on stoichiometry. The points in the ODEs where it is not purely stoichiometric are denoted with a `##!!`

```python
class glycolysis():
    def __init__(self,
                  interpolate_dict:dict,

                  met_names:list,
                  dilution_rate:float):
       self.interpolate_dict=interpolate_dict
       self.met_names=met_names
       self.dilution_rate=dilution_rate
       self.ECbiomass=self.interpolate_dict['ECbiomass'].evaluate(dilution_rate)


    def __call__(self,t,y,args):
        params=args
        y=dict(zip(self.met_names,y))
        
        ##!!
        D=self.dilution_rate #dilution rate. In steady state D=mu        
        y['ECglucose']=self.interpolate_dict['ECglucose'].evaluate(t)
        ##!!
        eval_dict={**y,**params}



        #poly_sink parameters are defined based on a dilution rate dependency, which is an argument to the model
        
        eval_dict['poly_sinkG6P']=jnp.abs(3.6854 * D**3 -   1.4119 * D**2 -  0.6312* D    - 0.0043) 
        eval_dict['poly_sinkF6P']=jnp.abs(519.3740 * D**6 - 447.7990 * D**5 + 97.2843 * D**4 + 8.0698 * D**3 - 4.4005 * D**2 + 0.6254 * D - 0.0078)
        eval_dict['poly_sinkGAP']=jnp.abs(170.8447 * D**6 - 113.2975 * D**5 + 2.6494 * D**4 + 10.2461 * D**3 - 1.8002 * D**2 + 0.1988 * D + 0.0012)
        eval_dict['poly_sinkP3G']=jnp.abs(-0.2381* D**2 - 0.0210 * D -0.0034)
        eval_dict['poly_sinkPEP']=jnp.abs(- 0.0637 * D**2 -   0.0617 * D   -  0.0008)
        eval_dict['poly_sinkPYR']=jnp.abs(-8.4853e+03 * D**6 + 9.4027e+03 * D**5 - 3.8027e+03 * D**4 + 700.5 * D**3 - 60.26 * D**2 + 0.711 * D - 0.0356)
        eval_dict['poly_sinkACE']=jnp.abs(118.8562 * D**6 - 352.3943 * D**5 + 245.6092 * D**4 - 75.2550 * D**3 + 11.1153 * D**2 - 1.0379 * D + 0.0119)


        ## calculate the expression given the dilution rate and update parameters

        rate_vGLT=v_GLT(eval_dict)
        rate_vHXK=v_HXK(eval_dict)
        rate_vNTH1=v_NTH1(eval_dict)
        rate_vPGI=v_PGI(eval_dict)
        rate_vsinkG6P=v_sinkG6P(eval_dict)
        
        rate_vsinkF6P=v_sinkF6P(eval_dict)
        rate_vPGM1=v_PGM1(eval_dict)
        rate_vTPS1=v_TPS1(eval_dict)
        rate_vTPS2=v_TPS2(eval_dict)
        rate_vPFK=v_PFK(eval_dict)
        
        rate_vALD=v_ALD(eval_dict)
        rate_TPI1=v_TPI1(eval_dict)
        rate_GP3DH=v_G3PDH(eval_dict)
        rate_PGK=v_PGK(eval_dict)
        rate_vsinkGAP=v_sinkGAP(eval_dict)
        
        rate_GAPDH=v_GAPDH(eval_dict)
        rate_vsink3PGA=vsink3PGA(eval_dict)
        rate_HOR2=v_HOR2(eval_dict)
        rate_vGLycT=v_GlycT(eval_dict)
        rate_PGM=v_PGM(eval_dict)
        
        rate_ENO=v_ENO(eval_dict)
        rate_vsinkPEP=vsinkPEP(eval_dict)
        rate_PYK1=v_PYK1(eval_dict)
        rate_vsinkPYR=vsinkPYR(eval_dict)
        rate_vPDC=v_PDC(eval_dict)
        
        rate_ADH=v_ADH(eval_dict)
        rate_vsinkACE=vsinkACE(eval_dict)
        rate_ETOH_transport=v_EtohT(eval_dict)
        rate_vmitoNADH=v_mitoNADH(eval_dict)
        rate_ATPmito=v_ATPmito(eval_dict)
        
        rate_ATPase=v_ATPase(eval_dict)
        rate_ADK1=v_ADK(eval_dict)
        rate_VacPi=v_VacPi(eval_dict)
        rate_AMD1=v_AMD1(eval_dict)
        rate_ADE1312=v_ADE1312(eval_dict)
        
        rate_ISN1=v_ISN1(eval_dict)
        rate_PNP1=v_PNP1(eval_dict)
        rate_HPT1=v_HPT1(eval_dict)


	
        dG1P=-rate_vPGM1-params['flux_ugp']
        dT6P=+rate_vTPS1 -rate_vTPS2
        dICTRE=+rate_vTPS2 -rate_vNTH1


        dICglci=+rate_vGLT - rate_vHXK +2*rate_vNTH1

        dICG6P=+rate_vHXK-rate_vPGI -rate_vsinkG6P +rate_vPGM1-rate_vTPS1
        dICF6P=+rate_vPGI +rate_vsinkF6P-rate_vPFK
        dICFBP=+rate_vPFK -rate_vALD

        dICDHAP=+rate_vALD - rate_TPI1 - rate_GP3DH
        dICG3P=+rate_GP3DH-rate_HOR2
        dICGlyc=+rate_HOR2-rate_vGLycT

        dICGAP=+rate_vALD +rate_TPI1  -rate_GAPDH +rate_vsinkGAP
        dICBPG=+rate_GAPDH -rate_PGK
        dIC3PG=+rate_PGK - rate_vsink3PGA -rate_PGM
        dIC2PG=+rate_PGM-rate_ENO

        dICPEP=+rate_ENO -rate_vsinkPEP-rate_PYK1 
        
        dICPYR= +rate_PYK1 -rate_vPDC -rate_vsinkPYR 
        
        dICACE= +rate_vPDC -rate_ADH -rate_vsinkACE

        dICETOH=+rate_ADH -rate_ETOH_transport
        


        dICNAD=+rate_GP3DH -rate_GAPDH   +rate_ADH +rate_vmitoNADH 
        dICNADH=-rate_GP3DH +rate_GAPDH   -rate_ADH -rate_vmitoNADH
        dATP=+rate_ADK1-rate_vHXK -rate_ATPase -rate_vPFK + rate_PGK +rate_PYK1 -rate_vTPS1 +rate_ATPmito
        dADP=-2*rate_ADK1+rate_vHXK +rate_ATPase +rate_vPFK -rate_PGK - rate_PYK1 +rate_vTPS2 -rate_ATPmito
        dAMP=+rate_ADK1  -rate_AMD1 +rate_ADE1312
        
        dPHOS=-rate_GAPDH +rate_ATPase +rate_HOR2 +2*rate_vTPS1 +rate_vTPS2 -rate_ATPmito +rate_ISN1-rate_PNP1 +rate_VacPi +rate_vsinkG6P -rate_vsinkF6P +rate_vsink3PGA +rate_vsinkPEP-rate_vsinkGAP
        dIMP=rate_AMD1-rate_ADE1312+rate_HPT1-rate_ISN1
        dINO=rate_ISN1-rate_PNP1
        dHYP=+rate_PNP1 -rate_HPT1

        ##!! 
        dECETOH=+rate_ETOH_transport*self.ECbiomass*0.002 - (y['ECETOH']/3600)*self.dilution_rate #ethanol production
        dECglyc=+rate_vGLycT*self.ECbiomass*0.002 - (y['ECETOH']/3600)*self.dilution_rate #glycerol 
        ##!!  

        return jnp.stack([dG1P,dT6P,dICTRE,dICglci,dICG6P,dICF6P,
                          dICFBP,dICDHAP,dICG3P,
                          dICGlyc,dICGAP,dICBPG,dIC3PG,
                          dIC2PG,dICPEP,dICPYR,dICACE,dICETOH,dECETOH,dECglyc,dICNADH,
                          dICNAD,dATP,dADP,dAMP,dPHOS,dIMP,dINO,dHYP])#,dICPEP,dICPYR,dICACE])
```
#### Parameterization
For a script that parameterizes this model, we refer to the training scripts in [scripts/experiments/](https://github.com/AbeelLab/jaxkineticmodel/blob/main/scripts/experiments/1709_train_gp_onedataset.py), or you can use the object-oriented training process as described in [Training models](training_models.md)


### References
[1] Lao-Martil, D., Schmitz, J. P., Teusink, B., & van Riel, N. A. (2023). Elucidating yeast glycolytic dynamics at steady state growth and glucose pulses through kinetic metabolic modeling. Metabolic engineering, 77, 128-142.
[2] Arxiv reference
