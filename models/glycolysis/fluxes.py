# INITIALIZE FLUXES

from kinetic_mechanisms.KineticMechanisms import *
from kinetic_mechanisms.KineticMechanismsCustom import *
from kinetic_mechanisms.KineticModifiers import *


# DEFINE PARAMETERS

# GLT
p_GLT_KeqGLT = 1
p_GLT_KmGLTGLCi = 1.0078
p_GLT_KmGLTGLCo = 1.0078
p_GLT_VmGLT = 8.1327


#HXK
p_HXK_ExprsCor = 1
p_HXK1_Kadp = 0.3492
p_HXK1_Katp = 0.0931
p_HXK1_Keq = 3.7213e+03
p_HXK1_Kg6p = 34.7029
p_HXK1_Kglc = 0.3483
p_HXK1_Kt6p = 0.0073
p_HXK1_kcat = 6.2548
f_HXK1 = 1; 
f_HXK2 = 0; 
p_HXK_Vmax = p_HXK_ExprsCor * p_HXK1_kcat * (f_HXK1 + f_HXK2)


# ATPase
p_ATPase_ratio = 0.23265
# p_ATPase_ratio = 0.23265 # adjusted ATPase activity upon perturbation
# p_ATPase_ratio := piecewise(0.205, time > 3000, 0.23265)

# Vacu
p_vacuolePi_k = 0.1699
p_vacuolePi_steadyStatePi = 10

#PFK
f_PFK = 1; 
p_PFK_ExprsCor = 1
p_PFK_Camp = 0.0287
p_PFK_Catp = 1.2822
p_PFK_Cf16bp = 2.3638
p_PFK_Cf26bp = 0.0283
p_PFK_Ciatp = 40.3824
p_PFK_Kamp = 0.0100
p_PFK_Katp = 1.9971
p_PFK_Kf16bp = 0.0437
p_PFK_Kf26bp = 0.0012
p_PFK_Kf6p = 0.9166
p_PFK_Kiatp = 4.9332
p_PFK_L = 1.3886
p_PFK_gR = 1.8127
p_PFK_kcat = 8.7826
p_PFK_F26BP = 1e-3  
p_PFK_Vmax = p_PFK_ExprsCor * p_PFK_kcat * f_PFK


# GADPH
p_TDH1_Keq = 0.0054
p_TDH1_Kglyceral3p = 4.5953
p_TDH1_Kglycerate13bp = 0.9076
p_TDH1_Knad = 1.1775
p_TDH1_Knadh = 0.0419
p_TDH1_Kpi = 0.7731
p_TDH1_kcat = 78.6422
f_TDH1 = 1
f_TDH2 = 0
f_TDH3 = 0 
p_GAPDH_ExprsCor = 1
p_GAPDH_Vmax = p_GAPDH_ExprsCor * p_TDH1_kcat * (f_TDH1 + f_TDH2 + f_TDH3)


# NTH1
p_NTH1_Ktre = 2.1087
p_NTH1_kcat = 4.5132
f_NTH1 = 0.0020
p_NTH1_Vmax = p_NTH1_kcat * f_NTH1

# PGI
p_PGI_ExprsCor = 1
p_PGI1_Keq = 0.9564  # 0.956375672911768; #0.9564; # number of decimals this important
p_PGI1_Kf6p = 7.2433  # 7.243331730145231; #7.2433; # number of decimals this important
p_PGI1_Kg6p = 33.0689  # 33.068946195264843; #33.0689; # number of decimals this important
p_PGI1_kcat = 2.3215  # 2.321459895423278; #2.3215; # number of decimals this important
f_PGI1 = 1
p_PGI1_Vmax = p_PGI_ExprsCor * p_PGI1_kcat*f_PGI1

# TPI

p_TPI1_Kdhap = 205.9964
p_TPI1_Keq = 0.0515
p_TPI1_Kglyceral3p = 8.8483
p_TPI1_kcat = 16.1694
f_TPI1 = 1

# TPS1
f_TPS1 = 0.0014
p_TPS1_Kg6p = 4.5359
p_TPS1_Kudp_glc = 0.1268
p_TPS1_kcat = 9.6164e+03
p_TPS1_Kpi = 0.7890
p_TPS1_KmF6P = 1.5631
p_TPS1_Vmax = p_TPS1_kcat * f_TPS1

# TPS2
p_TPS2_Kt6p = 0.3686
p_TPS2_kcat = 28.4097
p_TPS2_Kpi = 0.7023
p_NTH1_Ktre = 2.1087
p_NTH1_kcat = 4.5132
f_TPS2 = 0.0013
p_TPS2_Vmax = p_TPS2_kcat * f_TPS2

# GlycT
p_GlycerolTransport = 0.1001
f_GLYCEROL_e = 0


# PGM1
p_PGM1_Keq = 21.3955
p_PGM1_Kg1p = 0.0653
p_PGM1_Kg6p = 0.0324
p_PGM1_kcat = 8.4574
f_PGM1 = 1
f_PGM2 = 0
f_PGM3 = 0
p_PGM1_Vmax = p_PGM1_kcat * (f_PGM1 + f_PGM2 + f_PGM3)

# PGM
p_GPM1_K2pg = 0.0750
p_GPM1_K3pg = 1.4151
p_GPM1_Keq = 0.1193
p_GPM1_kcat = 11.3652
f_GPM1 = 1
f_GPM2 = 0
f_GPM3 = 0


# HOR2
p_HOR2_Kglyc3p = 2.5844
p_HOR2_Kpi = 2.5491
p_HOR2_kcat = 1.2748
f_HOR2 = 1
p_HOR2_Vmax = p_HOR2_kcat*f_HOR2


# ENO
p_ENO1_K2pg = 0.0567
p_ENO1_Keq = 4.3589
p_ENO1_Kpep = 0.4831
p_ENO1_kcat = 3.3018
f_ENO1 = 1
f_ENO2 = 0

# ETOH
p_kETOHtransport = 0.0328
f_ETOH_e = 0

# mitoNADH
p_mitoNADHVmax = 0.2401
p_mitoNADHKm = 1.0000e-03

# PYK
p_PYK1_Kadp = 0.2430
p_PYK1_Katp = 9.3000
p_PYK1_Kf16bp = 0.1732
p_PYK1_Kpep = 0.2810
p_PYK1_L = 60000
p_PYK1_hill = 4
p_PYK1_kcat = 9.3167
f_PYK1 = 1
f_PYK2 = 0
p_PYK_ExprsCor = 1

#PGK
p_PGK_KeqPGK = 3.2348e+03
p_PGK_KmPGKADP = 0.2064
p_PGK_KmPGKATP = 0.2859
p_PGK_KmPGKBPG = 0.0031
p_PGK_KmPGKP3G = 0.4759
p_PGK_VmPGK = 55.1626
p_PGK_ExprsCor = 1

# mito
p_mitoVmax = 1.6034
p_mitoADPKm = 0.3394
p_mitoPiKm = 0.4568

# ALD
p_FBA1_Kdhap = 0.0300
p_FBA1_Keq = 0.1223
p_FBA1_Kf16bp = 0.6872
p_FBA1_Kglyceral3p = 3.5582
p_FBA1_kcat = 4.4067
f_FBA1 = 1
p_FBA1_Vmax = p_FBA1_kcat * f_FBA1



# ADH
p_ADH_KeqADH = 6.8487e-05
p_ADH_KiADHACE = 0.6431
p_ADH_KiADHETOH = 59.6935 
p_ADH_KiADHNAD = 0.9677
p_ADH_KiADHNADH = 0.0316 
p_ADH_KmADHACE = 1.1322 
p_ADH_KmADHETOH = 4.8970 
p_ADH_KmADHNAD = 0.1534 
p_ADH_KmADHNADH = 0.1208 
p_ADH_VmADH = 13.2581


# G3PDH
p_GPD1_Kadp = 1.1069
p_GPD1_Katp = 0.5573
p_GPD1_Kdhap = 2.7041
p_GPD1_Keq = 1.0266e+04
p_GPD1_Kf16bp = 12.7519
p_GPD1_Kglyc3p = 3.2278
p_GPD1_Knad = 0.6902
p_GPD1_Knadh = 0.0322
p_GPD1_kcat = 1.7064
f_GPD1 = 1
p_GPD1_Vmax = p_GPD1_kcat * f_GPD1


# ADK1
p_ADK1_k = 77.3163
p_ADK1_Keq = 0.2676

# PDC
p_PDC_ExprsCor = 1
f_PDC1 = 0.5290
p_PDC1_Kpi = 9.4294
p_PDC1_Kpyr = 12.9680
p_PDC1_hill = 0.7242
p_PDC1_kcat = 8.3613
p_PDC1_Vmax = p_PDC_ExprsCor * p_PDC1_kcat * f_PDC1


p_Ade13_Ade12_k = 0.6298  # NON-CONSTANT
p_Isn1_k = 0  # 0.3654
p_Pnp1_k = 0  # 0.0149
p_Hpt1_k = 0  # 0.0112
#   p_Ade13_Ade12_k := piecewise(0.6298, time > 3000, 0)
#   p_Isn1_k := piecewise(0.3654, time > 3000, 0)
#   p_Pnp1_k := piecewise(0.0149, time > 3000, 0)
#   p_Hpt1_k := piecewise(0.0112, time > 3000, 0)


# Amd1
p_Amd1_K50 = 10.9184
p_Amd1_Kpi = 1.6184e+03
p_Amd1_Katp = 5000
p_Amd1_Vmax = 0

#sinks
poly_sinkACE = -0.034836166800000
poly_sinkF6P = 0.024574614000000 
poly_sinkG6P = -0.077853600000000 
poly_sinkGAP = 0.012626909700000 
poly_sinkP3G = -0.007881000000000 
poly_sinkPEP = -0.007607000000000 
poly_sinkPYR = -0.161328300000000 
km_sinkACE = 1e-04
km_sinkF6P = 1e-04 
km_sinkG6P = 1e-02
km_sinkGAP = 5e-04
km_sinkP3G = 1e-03
km_sinkPEP = 1e-03
km_sinkPYR = 1e-03

# CREATE FLUXES

# Notes:
# - K_eq treated as a given!
def create_fluxes():

    
    
    
    # GLT
    substrates_GLT = ['ICglucose', 'ECglucose']
    v_GLT = Torch_Facilitated_Diffusion(vmax=p_GLT_VmGLT, k_equilibrium=p_GLT_KeqGLT,
                                        km_internal=p_GLT_KmGLTGLCi, km_external=p_GLT_KmGLTGLCo,
                                        substrate_names=substrates_GLT, to_be_learned=[True, False, True, True])

    # GLK/HXK
    substrates_GLK = ['ICATP','ICglucose']
    products_GLK = ['ICADP', 'ICG6P']
    modifiers_GLK = ['ICT6P']
    v_GLK = Torch_Rev_BiBi_MM_w_Inhibition(vmax=p_HXK_Vmax, k_equilibrium=p_HXK1_Keq, km_substrate1=p_HXK1_Katp, 
                                           km_substrate2=p_HXK1_Kglc,
                                           km_product1=p_HXK1_Kadp, km_product2=p_HXK1_Kg6p, ki_inhibitor=p_HXK1_Kt6p, 
                                           substrate_names=substrates_GLK,
                                           product_names=products_GLK,
                                           modifier_names=modifiers_GLK,
                                           to_be_learned=[True, False, True, True, True, True, True])
    # PGM1
    substrates_PGM1 = ['ICG1P']
    products_PGM1 = ['ICG6P']
    v_PGM1 = Torch_Rev_UniUni_MM(vmax=p_PGM1_Vmax, k_equilibrium=p_PGM1_Keq,
                                 km_substrate=p_PGM1_Kg1p, km_product=p_PGM1_Kg6p,
                                 substrate_names=substrates_PGM1,
                                 product_names=products_PGM1,
                                 to_be_learned=[True, False, True, True])

    # TPS1
    substrates_TPS1 = ['ICG6P', 'ICUDPG']
    modifiers_TPS1  = ['ICF6P', 'ICPI']
    TPS1_inhibition_Pi = SimpleInhibitor(k_I=p_TPS1_Kpi, to_be_learned=[True])
    TPS1_activation_F6P = SimpleActivator(k_A=p_TPS1_KmF6P, to_be_learned=[True])
    v_TPS1 = Torch_Irrev_MM_Bi_w_Modifiers(vmax=p_TPS1_Vmax, km_substrate1=p_TPS1_KmF6P,
                                           km_substrate2=0.2, modifiers=[TPS1_inhibition_Pi, TPS1_activation_F6P], 
                                           substrate_names=substrates_TPS1,
                                           modifier_names=modifiers_TPS1,
                                           to_be_learned=[True, True, True])
    
    # TPS2
    #TPS2_inhibition_Pi = SimpleInhibitor(k_I=p_TPS2_Kpi, to_be_learned=[True])
    
    substrates_TPS2 = ['ICT6P']
    products_TPS2 = ['ICPI']
    v_TPS2 = Torch_Irrev_MM_Bi_w_Inhibition(vmax=p_TPS2_Vmax, km_substrate1=p_TPS2_Kt6p, ki=p_TPS2_Kpi,
                                            substrate_names=substrates_TPS2,
                                            product_names=products_TPS2,
                                            to_be_learned=[True, True, True])

    
    # NTH1
    substrates_NTH1 = ['ICtreh']
    v_NTH1 = Torch_Irrev_MM_Uni(p_NTH1_Vmax, p_NTH1_Ktre, 
                                substrate_names=substrates_NTH1,
                                to_be_learned=[True, True])

    # PGI
    substrates_PGI = ['ICF6P']
    products_PGI = ['ICATP']
    v_PGI = Torch_Rev_UniUni_MM(vmax=p_PGI1_Vmax, k_equilibrium=p_PGI1_Keq, km_substrate=p_PGI1_Kg6p,
                                km_product=p_PGI1_Kf6p, 
                                substrate_names=substrates_PGI,
                                product_names=products_PGI,
                                to_be_learned=[True, False, True, True])
    
    # PFK
    substrates_PFK = ['ICATP', 'ICglucose']
    products_PFK = ['ICFBP']
    modifiers_PFK = ['ICAMP']
    v_PFK = Torch_Specific(vmax=p_PFK_Vmax, kr_F6P=p_PFK_Kf6p, kr_ATP=p_PFK_Katp, gr=p_PFK_gR,
                           c_ATP=p_PFK_Catp, ci_ATP=p_PFK_Ciatp, ci_AMP=p_PFK_Camp, 
                           ci_F26BP=p_PFK_Cf26bp, ci_F16BP=p_PFK_Cf16bp, l=p_PFK_L, 
                           kATP=p_PFK_Kiatp, kAMP=p_PFK_Kamp, F26BP = p_PFK_F26BP,
                           kF26BP = p_PFK_Kf26bp, kF16BP = p_PFK_Kf16bp,
                           substrate_names=substrates_PFK,
                           product_names=products_PFK,
                           modifier_names=modifiers_PFK,
                           to_be_learned=[True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True])
    
    # ALD
    substrates_ALD = ['ICFBP']
    products_ALD = ['ICGAP', 'ICDHAP']
    v_ALD = Torch_MM_unibi(vmax=p_FBA1_Vmax, k_equilibrium=p_FBA1_Keq, km_substrate=p_FBA1_Kf16bp,
                                    km_product1=p_FBA1_Kglyceral3p, km_product2=p_FBA1_Kdhap, 
                                    substrate_names=substrates_ALD,
                                    product_names=products_ALD,
                                    to_be_learned=[True, False, True, True, True])

    # TPI
    substrates_TPI = ['ICDHAP']
    products_TPI = ['ICGAP']
    v_TPI = Torch_Rev_UniUni_MM(vmax=p_TPI1_kcat*f_TPI1, k_equilibrium=p_TPI1_Keq, km_substrate=p_TPI1_Kdhap,
                                km_product=p_TPI1_Kglyceral3p,
                                substrate_names=substrates_TPI,
                                product_names=products_TPI,
                                to_be_learned=[True, False, True, True])
    
    # G3PDH
    substrates_G3PDH = ['ICDHAP', 'ICNADH']
    products_G3PDH = ['ICG3P', 'ICNAD']
    modifiers_G3PDH = ['ICFBP', 'ICATP', 'ICADP']
    v_G3PDH = Torch_Rev_BiBi_MM_w_Activation(vmax=p_GPD1_Vmax, k_equilibrium=p_GPD1_Keq, 
                                             km_substrate1=p_GPD1_Kdhap, km_substrate2=p_GPD1_Knadh,
                                             km_product1=p_GPD1_Kglyc3p, km_product2=p_GPD1_Knad,
                                               ka1=p_GPD1_Kf16bp, ka2=p_GPD1_Katp, ka3=p_GPD1_Kadp,
                                               substrate_names=substrates_G3PDH,
                                               product_names=products_G3PDH,
                                               modifier_names=modifiers_G3PDH,
                                             to_be_learned=[True, False, True, True, True, True, True, True, True])
    
    # HOR2
    substrates_HOR2 = ['ICG3P']
    modifiers_HOR2 = ['ICPI']
    HOR2_inhibition_Pi = SimpleInhibitor(k_I=p_HOR2_Kpi, to_be_learned=[True])
    v_HOR2 = Torch_Irrev_MM_Uni_w_Modifiers(vmax=p_HOR2_Vmax, km_substrate=p_HOR2_Kglyc3p,
                                            modifiers=[HOR2_inhibition_Pi], 
                                            substrate_names=substrates_HOR2,
                                            modifier_names=modifiers_HOR2,
                                            to_be_learned=[True, True])
    
    # GlycT
    substrates_GlycT = ['ICglyc']
    v_GlycT = Torch_Diffusion(
        enzyme=f_GLYCEROL_e, transport_coef=p_GlycerolTransport,
        substrate_names=substrates_GlycT,
        to_be_learned=[True])
    

    # GAPDH
    substrates_GAPDH = ['ICGAP', 'ICNAD', 'ICPI']
    products_GAPDH = ['ICBPG', 'ICNADH']
    v_GAPDH = Torch_MM_Ordered_Bi_Tri(vmax=p_GAPDH_Vmax, k_equilibrium=p_TDH1_Keq, km_substrate1=p_TDH1_Kglyceral3p,
                                      km_substrate2=p_TDH1_Knad, ki=p_TDH1_Kpi, 
                                      km_product1=p_TDH1_Kglycerate13bp, km_product2=p_TDH1_Knadh,
                                      substrate_names=substrates_GAPDH,
                                      product_names=products_GAPDH,
                                      to_be_learned=[True, False, True, True, True, True, True])
    
    # PGK
    substrates_PGK = ['ICBPG', 'ICADP']
    products_PGK = ['IC3PG', 'ICATP']
    v_PGK = Torch_Rev_BiBi_MM_Vr(vmax=p_PGK_VmPGK, k_equilibrium=p_PGK_KeqPGK, 
                               km_substrate1=p_PGK_KmPGKBPG, km_substrate2=p_PGK_KmPGKADP, 
                               km_product1=p_PGK_KmPGKP3G, km_product2=p_PGK_KmPGKATP,
                               substrate_names=substrates_PGK,
                               product_names=products_PGK,
                              to_be_learned=[True, False, True, True, True, True])

    # PGM
    substrates_PGM = ['IC3PG']
    products_PGM = ['IC2PG']
    v_PGM = Torch_Rev_UniUni_MM(vmax=p_GPM1_kcat*f_GPM1, k_equilibrium=p_GPM1_Keq, km_substrate=p_GPM1_K3pg,\
                                substrate_names=substrates_PGM,
                                product_names=products_PGM,
                                km_product=p_GPM1_K2pg, to_be_learned=[True, False, True, True])
    # ENO
    substrates_ENO = ['IC2PG']
    products_ENO = ['ICPEP']
    v_ENO = Torch_Rev_UniUni_MM(vmax=p_ENO1_kcat*(f_ENO1+f_ENO2), k_equilibrium=p_ENO1_Keq,
                                km_substrate=p_ENO1_K2pg, km_product=p_ENO1_Kpep, 
                                substrate_names=substrates_ENO,
                                product_names=products_ENO,
                                to_be_learned=[True, False, True, True])
    
    # PYK
    substrates_PYK = ['ICPEP', 'ICADP']
    products_PYK = ['ICATP']
    modifiers_PYK = ['ICFBP']
    v_PYK = Torch_Hill_Bi_Irreversible_Activation(vmax=p_PYK1_kcat*(f_PYK1+f_PYK2), hill=p_PYK1_hill,
                                                  k_substrate1=p_PYK1_Kpep, k_substrate2=p_PYK1_Kadp,
                                                  k_product=p_PYK1_Katp, k_activator=p_PYK1_Kf16bp, l=p_PYK1_L,
                                                  substrate_names=substrates_PYK,
                                                  product_names=products_PYK,
                                                  modifier_names=modifiers_PYK,
                                                  to_be_learned=[True, True, True, True, True, True, True])
    # PDC
    substrates_PDC = ['ICPYR']
    modifiers_PDC = ['ICPI']
    v_PDC = Torch_Hill_Irreversible_Inhibition(vmax=p_PDC1_Vmax, k_half_substrate=p_PDC1_Kpyr,
                                               hill=p_PDC1_hill, ki=p_PDC1_Kpi, 
                                               substrate_names=substrates_PDC,
                                               modifier_names=modifiers_PDC,
                                               to_be_learned=[True, True, True, True])

    # ADH
    substrates_ADH = ['ICNAD', 'ICETOH']
    products_ADH = ['ICACE', 'ICNADH']
    v_ADH = Torch_MM_Ordered_Bi_Bi(vmax=p_ADH_VmADH, k_equilibrium=p_ADH_KeqADH, km_substrate1=p_ADH_KmADHNAD, km_substrate2=p_ADH_KmADHETOH,
                                   km_product1=p_ADH_KmADHACE, km_product2=p_ADH_KmADHNADH, 
                                   ki_substrate1=p_ADH_KiADHNAD, ki_substrate2=p_ADH_KiADHETOH, 
                                   ki_product1=p_ADH_KiADHACE,ki_product2=p_ADH_KiADHNADH,
                                   substrate_names=substrates_ADH,
                                   product_names=products_ADH,
                                   to_be_learned=[True, False, True, True, True, True, True, True, True, True])
   
    # EtohT
    substrates_EtohT = ['ICETOH']
    v_EtohT = Torch_Diffusion(
        enzyme=f_ETOH_e, transport_coef=p_kETOHtransport, 
        substrate_names=substrates_EtohT,
        to_be_learned=[True])
    
    # ATPmito
    substrates_ATPmito = ['ICPI', 'ICADP']
    v_ATPmito = Torch_Irrev_MM_Bi(vmax=p_mitoVmax, km_substrate1=p_mitoADPKm,
                                  km_substrate2=p_mitoPiKm,
                                  substrate_names=substrates_ATPmito,
                                  to_be_learned=[True, True, True])
    
    # ATPase
    substrates_ATPase = ['ICATP']
    products_ATPase = ['ICADP']
    v_ATPase = Torch_ATPase(ATPase_ratio=p_ATPase_ratio, substrate_names=substrates_ATPase,
                            product_names=products_ATPase, to_be_learned=[True])
    
    # ADK1
    substrates_ADK1 = ['ICADP']
    products_ADK1 = ['ICATP', 'ICAMP']
    v_ADK1 = Torch_MA_Rev_Bi(k_equilibrium=p_ADK1_k, k_fwd=p_ADK1_Keq, 
                             substrate_names=substrates_ADK1,
                             product_names=products_ADK1,
                             to_be_learned=[False, True])
    
    # vacPi
    substrates_vacPi = ['ICPI']
    v_vacPi = Torch_MA_Rev(k=p_vacuolePi_k, steady_state_substrate=p_vacuolePi_steadyStatePi, 
                           substrate_names=substrates_vacPi,
                           to_be_learned=[True])
    
    # Amd1
    substrates_Amd1 = ['ICAMP']
    products_Amd1 = ['ICATP']
    modifiers_Amd1 = ['ICPI']
    v_Amd1 = Torch_Amd1(vmax=p_Amd1_Vmax, k50=p_Amd1_K50, ki=p_Amd1_Kpi, k_atp=p_Amd1_Katp, 
                        substrate_names=substrates_Amd1,
                        product_names=products_Amd1,
                        modifier_names=modifiers_Amd1,
                        to_be_learned=[True, True, True, True]) 
    
    # Ade1312
    substrates_Ade1312 = ['ICIMP']
    v_Ade1312 = Torch_MA_Irrev(k_fwd=p_Ade13_Ade12_k,
                               substrate_names=substrates_Ade1312,
                               to_be_learned=[True])

    # Isn1
    substrates_Isn1 = ['ICIMP']
    v_Isn1 = Torch_MA_Irrev(k_fwd=p_Isn1_k,
                            substrate_names=substrates_Isn1,
                            to_be_learned=[True])
    
    # Pnp1
    
    substrates_Pnp1 = ['ICINO']
    v_Pnp1 = Torch_MA_Irrev(k_fwd=p_Pnp1_k, 
                            substrate_names=substrates_Pnp1,
                            to_be_learned=[True])  # ISSUES??
    
    # Hpt1
    substrates_Hpt1 = ['ICHYP']
    v_Hpt1 = Torch_MA_Irrev(k_fwd=p_Hpt1_k,
                            substrate_names=substrates_Hpt1,
                            to_be_learned=[True])  # ISSUES??
    
    # NADHmito
    substrates_NADHmito = ['ICNADH']
    v_NADHmito = Torch_MM(p_mitoNADHVmax, p_mitoNADHKm, 
                          substrate_names=substrates_NADHmito,
                          to_be_learned=[True, True])

    # sinks
    substrates_sinkG6P = ['ICG6P']
    substrates_sinkF6P = ['ICF6P']
    substrates_sinkGAP = ['ICGAP']
    substrates_sinkP3G = ['IC3PG']
    substrates_sinkPEP = ['ICPEP']
    substrates_sinkPYR = ['ICPYR']
    substrates_sinkACE = ['ICACE']



    vsinkG6P = Torch_MM_Sink(v_sink=poly_sinkG6P, km_sink=km_sinkG6P, 
                             substrate_names=substrates_sinkG6P, to_be_learned=[True, True])
    vsinkF6P = Torch_MM_Sink(v_sink=poly_sinkF6P, km_sink=km_sinkF6P,
                             substrate_names=substrates_sinkF6P, to_be_learned=[True, True])
    vsinkGAP = Torch_MM_Sink(v_sink=poly_sinkGAP, km_sink=km_sinkGAP, 
                             substrate_names=substrates_sinkGAP, to_be_learned=[True, True])
    vsinkP3G = Torch_MM_Sink(v_sink=poly_sinkP3G, km_sink=km_sinkP3G, 
                             substrate_names=substrates_sinkP3G, to_be_learned=[True, True])
    vsinkPEP = Torch_MM_Sink(v_sink=poly_sinkPEP, km_sink=km_sinkPEP,
                             substrate_names=substrates_sinkPEP, to_be_learned=[True, True])
    vsinkPYR = Torch_MM_Sink(v_sink=poly_sinkPYR, km_sink=km_sinkPYR,
                             substrate_names=substrates_sinkPYR, to_be_learned=[True, True])
    vsinkACE = Torch_MM_Sink(v_sink=poly_sinkACE, km_sink=km_sinkACE,
                             substrate_names=substrates_sinkACE, to_be_learned=[True, True])
    v = {
        'v_GLT': v_GLT,
        'v_GLK': v_GLK,
        'v_PGM1': v_PGM1,
        'v_TPS1': v_TPS1,
        'v_TPS2': v_TPS2,
        'v_NTH1': v_NTH1,
        'v_PGI': v_PGI,
        'v_PFK': v_PFK,
        'v_ALD': v_ALD,
        'v_TPI': v_TPI,
        'v_G3PDH': v_G3PDH,
        'v_HOR2': v_HOR2,
        'v_GlycT': v_GlycT,
        'v_GAPDH': v_GAPDH,
        'v_PGK': v_PGK,
        'v_PGM': v_PGM,
        'v_ENO': v_ENO,
        'v_PYK': v_PYK,
        'v_PDC': v_PDC,
        'v_ADH': v_ADH,
        'v_EtohT': v_EtohT,
        'v_ATPmito': v_ATPmito,
        'v_ATPase': v_ATPase,
        'v_ADK1': v_ADK1,
        'v_vacPi': v_vacPi,
        'v_Amd1': v_Amd1,
        'v_Ade1312': v_Ade1312,
        'v_Isn1': v_Isn1,
        'v_Pnp1': v_Pnp1,
        'v_Hpt1': v_Hpt1,
        'v_NADHmito': v_NADHmito,
        'vsinkG6P': vsinkG6P,
        'vsinkF6P': vsinkF6P,
        'vsinkGAP': vsinkGAP,
        'vsinkP3G': vsinkP3G,
        'vsinkPEP': vsinkPEP,
        'vsinkPYR': vsinkPYR,
        'vsinkACE': vsinkACE
    }
    return v


