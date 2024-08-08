

import sys

sys.path.insert(0,"/home/plent/Documenten/Gitlab/NeuralODEs/jax_neural_odes")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import diffrax
import jax.numpy as jnp
import jax
import equinox as eqx
import optax

from source.kinetic_mechanisms.JaxKineticMechanisms import *
from source.kinetic_mechanisms.JaxKineticMechanismsCustom import *
from source.parameter_estimation.training import *
jax.config.update("jax_enable_x64", True)


#get time points
glycolysis_data=pd.read_csv("datasets/VanHeerden_Glucose_Pulse/FF1_timeseries_format.csv",index_col=0).T
time_points=[int(i) for i in glycolysis_data.index.to_list()]



GLT_params={'p_GLT_KmGLTGLCi':1.0078,'p_GLT_KmGLTGLCo':1.0078,'p_GLT_VmGLT':8.1327}
HXK1_params={'p_HXK1_Kglc':0.3483,'p_HXK_Vmax':1 * 6.2548 *(1 + 0)}
NTH1_params={"p_NTH1_Ktre" : 2.1087,"p_NTH1_kcat" : 4.5132,"f_NTH1" : 0.0020,"p_NTH1_Vmax" : 4.5132 * 0.0020}
PGI_params={ "p_PGI1_Keq": 0.9564,"p_PGI1_Kf6p": 7.2433,"p_PGI1_Kg6p": 33.0689,'p_PGI1_Vmax':1*2.3215*1}
v_sinkG6P_params={'poly_sinkG6P':0.077853600000000,'km_sinkG6P':1e-02} #important, this should be negative in stoichiometry!
v_sinkF6P_params={'poly_sinkF6P':0.024574614000000 ,'km_sinkF6P':1e-04}
PGM1_params={"p_PGM1_Keq": 21.3955,"p_PGM1_Kg1p": 0.0653,"p_PGM1_Kg6p": 0.0324,"p_PGM1_Vmax":8.4574*(1+0+0)}
TPS1_params={ "p_TPS1_Kg6p": 4.5359,"p_TPS1_Kudp_glc": 0.1268,"p_TPS1_Kpi": 0.7890,"p_TPS1_KmF6P": 1.5631,"p_TPS1_Vmax":9.6164e+03*0.0014}
PFK_params=pfk_parameters = {"p_PFK_Camp": 0.0287,"p_PFK_Catp": 1.2822,"p_PFK_Cf16bp": 2.3638,"p_PFK_Cf26bp": 0.0283,"p_PFK_Ciatp": 40.3824,"p_PFK_Kamp": 0.0100,
    "p_PFK_Katp": 1.9971,"p_PFK_Kf16bp": 0.0437,"p_PFK_Kf26bp": 0.0012,"p_PFK_Kf6p": 0.9166,"p_PFK_Kiatp": 4.9332,"p_PFK_L": 1.3886,
    "p_PFK_gR": 1.8127,"p_PFK_F26BP": 1e-3,"p_PFK_Vmax": 1 * 8.7826 * 1 } # Calculated as p_PFK_ExprsCor * p_PFK_kcat * f_PFK}

v_sinkF16P={'poly_sinkFBP':0.024574614000000 ,'km_sinkFBP':1e-04} #modelling reaction

params={**GLT_params,**HXK1_params,**NTH1_params,**NTH1_params,**PGI_params,**v_sinkG6P_params,**PGM1_params,**TPS1_params,**v_sinkF6P_params,**PFK_params,**v_sinkF16P}







v_GLT=Jax_Facilitated_Diffusion(substrate_extracellular='ECglucose',product_intracellular='ICglucose',vmax="p_GLT_VmGLT",km_internal='p_GLT_KmGLTGLCi',km_external='p_GLT_KmGLTGLCo')
v_HXK=Jax_Irrev_MM_Uni(substrate='ICglucose',vmax='p_HXK_Vmax',km_substrate='p_HXK1_Kglc')
v_NTH1=Jax_Irrev_MM_Uni(substrate='ICtreh',vmax='p_NTH1_Vmax',km_substrate='p_NTH1_Ktre')
v_PGI=Jax_Rev_UniUni_MM(substrate='ICG6P',product='ICF6P',vmax='p_PGI1_Vmax',k_equilibrium='p_PGI1_Keq',km_substrate='p_PGI1_Kg6p',km_product='p_PGI1_Kf6p')
v_sinkG6P=Jax_MM_Sink(substrate='ICG6P',v_sink='poly_sinkG6P',km_sink='km_sinkG6P')
v_sinkF6P=Jax_MM_Sink(substrate='ICF6P',v_sink='poly_sinkF6P',km_sink='km_sinkF6P')
v_PGM1=Jax_Rev_UniUni_MM(substrate='ICG1P',product='ICG6P',vmax='p_PGM1_Vmax',k_equilibrium='p_PGM1_Keq',km_substrate='p_PGM1_Kg1p',km_product='p_PGM1_Kg6p') #to do v_TPS1 for 2nd rate law
v_TPS1=Jax_Irrev_MM_Bi(substrate1="ICG6P",substrate2="ICG1P",vmax="p_TPS1_Vmax",km_substrate1="p_TPS1_Kg6p",
                                   km_substrate2="p_TPS1_Kudp_glc")
v_PFK=Jax_Specific(substrate1="ICF6P",substrate2="ICATP",product1="ICFBP",modifier="ICAMP",vmax="p_PFK_Vmax",kr_F6P="p_PFK_Kf6p", kr_ATP="p_PFK_Katp", gr="p_PFK_gR",c_ATP="p_PFK_Catp", ci_ATP="p_PFK_Ciatp", ci_AMP="p_PFK_Camp", 
                           ci_F26BP="p_PFK_Cf26bp", ci_F16BP="p_PFK_Cf16bp", l="p_PFK_L", 
                           kATP="p_PFK_Kiatp", kAMP="p_PFK_Kamp", F26BP ="p_PFK_F26BP",
                           kF26BP = "p_PFK_Kf26bp", kF16BP = "p_PFK_Kf16bp")

v_sinkF16P=Jax_MM_Sink(substrate='ICFBP',v_sink='poly_sinkFBP',km_sink='km_sinkFBP')

#interpolate things we do not wish to model    

#Extracellular Glucose interpolation
coeffs_ECglucose=diffrax.backward_hermite_coefficients(ts=jnp.array(time_points),ys=jnp.array(glycolysis_data['ECglucose']),
                                             fill_forward_nans_at_end=True)
EC_glucose_interpolation_cubic=diffrax.CubicInterpolation(ts=jnp.array(time_points),coeffs=coeffs_ECglucose)


#Trehalose interpolation
coeffs_ICtreh=diffrax.backward_hermite_coefficients(ts=jnp.array(time_points),ys=jnp.array(glycolysis_data['ICtreh']),
                                             fill_forward_nans_at_end=True)
ECtreh_interpolation_cubic=diffrax.CubicInterpolation(ts=jnp.array(time_points),coeffs=coeffs_ICtreh)


interpolated_mets={'ECglucose':EC_glucose_interpolation_cubic,
                    'ICtreh':ECtreh_interpolation_cubic}


class glycolysis():
    def __init__(self,
                  interpolate_dict):
       self.interpolate_dict=interpolate_dict

    def __call__(self,t,y,args):
        
        params=args


        met_names=['ICglucose','ICG6P','ICF6P',"ICFBP"]
        y=dict(zip(met_names,y))


        y['ECglucose']=self.interpolate_dict['ECglucose'].evaluate(t)
        y['ICtreh']=self.interpolate_dict['ICtreh'].evaluate(t)
        # y['ICF6P']=self.interpolate_dict['ICF6P'].evaluate(t)
        # y['ICG1P']=self.interpolate_dict['ICG1P'].evaluate(t)
        y['ICG1P']=0.130
        y['ICATP']=4.52
        y['ICAMP']=0.31
        


        eval_dict={**y,**params}
        #modifiers and stuff
        # eval_dict['ICPHOS']

        rate_vGLT=v_GLT(eval_dict)
        rate_vHXK=v_HXK(eval_dict)
        rate_vNTH1=v_NTH1(eval_dict)
        rate_vPGI=v_PGI(eval_dict)

        rate_vsinkG6P=v_sinkG6P(eval_dict)
        rate_vsinkF6P=v_sinkF6P(eval_dict)
        rate_vPGM1=v_PGM1(eval_dict)
        rate_vTPS1=v_TPS1(eval_dict)
        rate_vPFK=v_PFK(eval_dict)
        rate_vsinkF16P=v_sinkF16P(eval_dict)

        dICglci=+rate_vGLT - rate_vHXK +2*rate_vNTH1
        dICG6P=+rate_vHXK-rate_vPGI -rate_vsinkG6P +rate_vPGM1-rate_vTPS1
        dICF6P=+rate_vPGI +rate_vsinkF6P-rate_vPFK
        dICFBP=rate_vPFK -rate_vsinkF16P #vsinkF16P is a modeling reaction, needs to be removed later



        return jnp.stack([dICglci,dICG6P,dICF6P,dICFBP])
    


glycolyse=jax.jit(glycolysis(interpolated_mets))
term=diffrax.ODETerm(glycolyse)

ts=jnp.linspace(0,400,1000)

solver = diffrax.Kvaerno5()
saveat=diffrax.SaveAt(ts=ts)
stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)


y0=jnp.array([0.2,0.716385,0.202293,0.057001])


class NeuralODE():
    def __init__(self,func):

        
        self.func=func
        self.rtol=1e-3
        self.atol=1e-6
        self.max_steps=100000
    def __call__(self,ts,y0,params):
        solution = diffrax.diffeqsolve(
        diffrax.ODETerm(self.func),
        diffrax.Kvaerno5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0,
        args=(params),
        stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol,pcoeff=0.4,icoeff=0.3,dcoeff=0),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=self.max_steps)
        return solution.ys


glycolyse=jax.jit(NeuralODE(glycolysis(interpolated_mets)))



def create_log_params_loss_func(model,to_include:list):
    """Loss function for log transformed parameters """
    def loss_func(params,ts,ys):
        params=exponentiate_parameters(params)
        mask=~jnp.isnan(jnp.array(ys))
        ys=jnp.atleast_2d(ys)
        y0=ys[0,:]
        y_pred=model(ts,y0,params)
        ys = jnp.where(mask, ys, 0)
        y_pred = jnp.where(mask, y_pred, 0)


        ys=ys[:,to_include]
        y_pred=y_pred[:,to_include]

        non_nan_count = jnp.sum(mask)
        
        loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
        return loss
    return loss_func

log_loss_func=jax.jit(create_log_params_loss_func(glycolyse,[0,1,2]))
loss_func=jax.jit(create_loss_func(glycolyse))


@jax.jit
def update_log(opt_state,params,ts,ys):
    """Update rule for the gradients for log-transformed parameters. Can only be applied
    to nonnegative parameters"""
    log_params=log_transform_parameters(params)
    loss=log_loss_func(log_params,ts,ys)

    grads=jax.jit(jax.grad(log_loss_func,0))(log_params,ts,ys) #loss w.r.t. parameters
    updates,opt_state=optimizer.update(grads,opt_state)

    #we perform updates in log space, but only return params in lin space
    log_params = optax.apply_updates(log_params, updates)
    lin_params = exponentiate_parameters(log_params) 
    return opt_state,lin_params,loss,grads







params_init=params
lr=1e-3
optimizer = optax.adabelief(lr)
clip_by_global=optax.clip_by_global_norm(np.log(4))
optimizer = optax.chain(optimizer,clip_by_global)
opt_state = optimizer.init(params_init)


print("round 3")
# alpha1=np.linspace(0.2,1.0,2500)
for step in range(5000):
    opt_state,params_init,loss,grads=update_log(opt_state,params_init,time_points,
                                                jnp.array(glycolysis_data[['ICglucose','ICG6P','ICF6P','ICFBP']]))
    if step% 50==0:
        
#           # Scale step to range [0, 1]
#         # print(f"global norm: {global_norm(grads)}")
        print(f"Step {step}, Loss {loss}")




plt.plot(ts,glycolyse(ts,y0,params_init)[:,0],c="red",label="Intracellular Glucose (in loss)")
plt.plot(ts,glycolyse(ts,y0,params_init)[:,1],c="blue",label="Glucose-6-Phosphate (in loss)")
plt.plot(ts,glycolyse(ts,y0,params_init)[:,2],c="green",label="Fructose-6-phosphate (in loss)")
plt.plot(ts,glycolyse(ts,y0,params_init)[:,3],c="black",label="Fructose-Bi-phosphate (not in loss)")

plt.scatter(time_points,glycolysis_data['ICglucose'],label="ICglucose (true data)",c="red",alpha=0.8)
plt.scatter(time_points,glycolysis_data['ICG6P'],label="G6P (true data)",c="blue",alpha=0.8)
plt.scatter(time_points,glycolysis_data['ICF6P'],label="F6P (true data)",c="green",alpha=0.8)
plt.scatter(time_points,glycolysis_data['ICFBP'],label="FBP (true data)",c="black",alpha=0.8)
plt.legend()
plt.title("Glucose simulation")
# plt.yscale("log")
plt.savefig("figures/fitting_glycolysis.png")
plt.show()