import torch
from torch import nn


class Pollu_Model(torch.nn.Module):
    def __init__(self,fluxes,metabolites):
        super(Pollu_Model,self).__init__()
        self.fluxes=nn.ParameterDict(fluxes)
        self.metabolites=metabolites
        

    def calculate_fluxes(self,concentrations):
        self.fluxes['r1'].value=self.fluxes['r1'].calculate(concentrations[self.metabolites['y_1']])
        self.fluxes['r2'].value=self.fluxes['r2'].calculate(concentrations[self.metabolites['y_2']]*concentrations[self.metabolites['y_4']])
        self.fluxes['r3'].value=self.fluxes['r3'].calculate(concentrations[self.metabolites['y_5']]*concentrations[self.metabolites['y_2']])
        self.fluxes['r4'].value=self.fluxes['r4'].calculate(concentrations[self.metabolites['y_7']])
        self.fluxes['r5'].value=self.fluxes['r5'].calculate(concentrations[self.metabolites['y_7']])

        self.fluxes['r6'].value=self.fluxes['r6'].calculate(concentrations[self.metabolites['y_7']]*concentrations[self.metabolites['y_6']])
        self.fluxes['r7'].value=self.fluxes['r7'].calculate(concentrations[self.metabolites['y_9']])
        self.fluxes['r8'].value=self.fluxes['r8'].calculate(concentrations[self.metabolites['y_9']]*concentrations[self.metabolites['y_6']])
        self.fluxes['r9'].value=self.fluxes['r9'].calculate(concentrations[self.metabolites['y_11']]*concentrations[self.metabolites['y_2']])
        self.fluxes['r10'].value=self.fluxes['r10'].calculate(concentrations[self.metabolites['y_11']]*concentrations[self.metabolites['y_1']])

        self.fluxes['r11'].value=self.fluxes['r11'].calculate(concentrations[self.metabolites['y_13']])
        self.fluxes['r12'].value=self.fluxes['r12'].calculate(concentrations[self.metabolites['y_10']]*concentrations[self.metabolites['y_2']])
        self.fluxes['r13'].value=self.fluxes['r13'].calculate(concentrations[self.metabolites['y_14']])
        self.fluxes['r14'].value=self.fluxes['r14'].calculate(concentrations[self.metabolites['y_1']]*concentrations[self.metabolites['y_6']])
        self.fluxes['r15'].value=self.fluxes['r15'].calculate(concentrations[self.metabolites['y_3']])

        self.fluxes['r16'].value=self.fluxes['r16'].calculate(concentrations[self.metabolites['y_4']])     
   
        self.fluxes['r17'].value=self.fluxes['r17'].calculate(concentrations[self.metabolites['y_4']])
        self.fluxes['r18'].value=self.fluxes['r18'].calculate(concentrations[self.metabolites['y_16']])
        self.fluxes['r19'].value=self.fluxes['r19'].calculate(concentrations[self.metabolites['y_16']])
        
        self.fluxes['r20'].value=self.fluxes['r20'].calculate(concentrations[self.metabolites['y_17']]*concentrations[self.metabolites['y_6']])
        self.fluxes['r21'].value=self.fluxes['r21'].calculate(concentrations[self.metabolites['y_19']])
        self.fluxes['r22'].value=self.fluxes['r22'].calculate(concentrations[self.metabolites['y_19']])
        self.fluxes['r23'].value=self.fluxes['r23'].calculate(concentrations[self.metabolites['y_1']]*concentrations[self.metabolites['y_4']])
        self.fluxes['r24'].value=self.fluxes['r24'].calculate(concentrations[self.metabolites['y_19']]*concentrations[self.metabolites['y_1']])
        self.fluxes['r25'].value=self.fluxes['r25'].calculate(concentrations[self.metabolites['y_20']])


    def forward(self,_,conc_in):
        #conc_in=conc_in[0]

        self.calculate_fluxes(conc_in)
        # dy=torch.zeros_like(conc_in)
        
        # dy[0]  = -self.fluxes['r1'].value - self.fluxes['r10'].value - self.fluxes['r14'].value - self.fluxes['r23'].value - self.fluxes['r24'].value + self.fluxes['r2'].value + self.fluxes['r3'].value + self.fluxes['r9'].value + self.fluxes['r11'].value + self.fluxes['r12'].value + self.fluxes['r22'].value + self.fluxes['r25'].value
        # dy[1]  = -self.fluxes['r2'].value - self.fluxes['r3'].value - self.fluxes['r9'].value - self.fluxes['r12'].value + self.fluxes['r1'].value + self.fluxes['r21'].value
        # dy[2]  = -self.fluxes['r15'].value + self.fluxes['r1'].value + self.fluxes['r17'].value + self.fluxes['r19'].value + self.fluxes['r22'].value
        # dy[3]  = -self.fluxes['r2'].value - self.fluxes['r16'].value - self.fluxes['r17'].value - self.fluxes['r23'].value + self.fluxes['r15'].value
        # dy[4]  = -self.fluxes['r3'].value + self.fluxes['r4'].value + self.fluxes['r4'].value + self.fluxes['r6'].value + self.fluxes['r7'].value + self.fluxes['r13'].value + self.fluxes['r20'].value
        # dy[5]  = -self.fluxes['r6'].value - self.fluxes['r8'].value - self.fluxes['r14'].value - self.fluxes['r20'].value + self.fluxes['r3'].value + self.fluxes['r18'].value + self.fluxes['r18'].value
        # dy[6]  = -self.fluxes['r4'].value - self.fluxes['r5'].value - self.fluxes['r6'].value + self.fluxes['r13'].value
        # dy[7]  = self.fluxes['r4'].value + self.fluxes['r5'].value + self.fluxes['r6'].value + self.fluxes['r7'].value
        # dy[8]  = -self.fluxes['r7'].value - self.fluxes['r8'].value
        # dy[9]  = -self.fluxes['r12'].value + self.fluxes['r7'].value + self.fluxes['r9'].value
        # dy[10] = -self.fluxes['r9'].value - self.fluxes['r10'].value + self.fluxes['r8'].value + self.fluxes['r11'].value
        # dy[11] = self.fluxes['r9'].value
        # dy[12] = -self.fluxes['r11'].value + self.fluxes['r10'].value
        # dy[13] = -self.fluxes['r13'].value + self.fluxes['r12'].value
        # dy[14] = self.fluxes['r14'].value
        # dy[15] = -self.fluxes['r18'].value - self.fluxes['r19'].value + self.fluxes['r16'].value
        # dy[16] = -self.fluxes['r20'].value
        # dy[17] = self.fluxes['r20'].value
        # dy[18] = -self.fluxes['r21'].value - self.fluxes['r22'].value - self.fluxes['r24'].value + self.fluxes['r23'].value + self.fluxes['r25'].value
        # dy[19] = -self.fluxes['r25'].value + self.fluxes['r24'].value

        dy0  = -self.fluxes['r1'].value - self.fluxes['r10'].value - self.fluxes['r14'].value - self.fluxes['r23'].value - self.fluxes['r24'].value + self.fluxes['r2'].value + self.fluxes['r3'].value + self.fluxes['r9'].value + self.fluxes['r11'].value + self.fluxes['r12'].value + self.fluxes['r22'].value + self.fluxes['r25'].value
        dy1  = -self.fluxes['r2'].value - self.fluxes['r3'].value - self.fluxes['r9'].value - self.fluxes['r12'].value + self.fluxes['r1'].value + self.fluxes['r21'].value
        dy2  = -self.fluxes['r15'].value + self.fluxes['r1'].value + self.fluxes['r17'].value + self.fluxes['r19'].value + self.fluxes['r22'].value
        dy3 = -self.fluxes['r2'].value - self.fluxes['r16'].value - self.fluxes['r17'].value - self.fluxes['r23'].value + self.fluxes['r15'].value
        dy4  = -self.fluxes['r3'].value + self.fluxes['r4'].value + self.fluxes['r4'].value + self.fluxes['r6'].value + self.fluxes['r7'].value + self.fluxes['r13'].value + self.fluxes['r20'].value
        dy5  = -self.fluxes['r6'].value - self.fluxes['r8'].value - self.fluxes['r14'].value - self.fluxes['r20'].value + self.fluxes['r3'].value + self.fluxes['r18'].value + self.fluxes['r18'].value
        dy6  = -self.fluxes['r4'].value - self.fluxes['r5'].value - self.fluxes['r6'].value + self.fluxes['r13'].value
        dy7  = self.fluxes['r4'].value + self.fluxes['r5'].value + self.fluxes['r6'].value + self.fluxes['r7'].value
        dy8  = -self.fluxes['r7'].value - self.fluxes['r8'].value
        dy9  = -self.fluxes['r12'].value + self.fluxes['r7'].value + self.fluxes['r9'].value
        dy10 = -self.fluxes['r9'].value - self.fluxes['r10'].value + self.fluxes['r8'].value + self.fluxes['r11'].value
        dy11 = self.fluxes['r9'].value
        dy12 = -self.fluxes['r11'].value + self.fluxes['r10'].value
        dy13 = -self.fluxes['r13'].value + self.fluxes['r12'].value
        dy14 = self.fluxes['r14'].value
        dy15 = -self.fluxes['r18'].value - self.fluxes['r19'].value + self.fluxes['r16'].value
        dy16 = -self.fluxes['r20'].value
        dy17 = self.fluxes['r20'].value
        dy18 = -self.fluxes['r21'].value - self.fluxes['r22'].value - self.fluxes['r24'].value + self.fluxes['r23'].value + self.fluxes['r25'].value
        dy19 = -self.fluxes['r25'].value + self.fluxes['r24'].value


        # print(dy.shape)
        #If I get detach problems: dXdt=torch.cat([LACT,ACT,PYR,X],dim=0)
        dy=torch.stack([dy0,dy1,dy2,dy3,dy4,dy5,dy6,dy7,dy8,dy9,dy10,dy11,dy12,dy13,dy14,dy15,dy16,dy17,dy18,dy19],-1)

        return dy
