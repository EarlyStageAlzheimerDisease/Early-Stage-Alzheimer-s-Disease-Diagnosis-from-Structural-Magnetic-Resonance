import random
import numpy as np
from Global_Vars import Global_Vars
from Model_3D_EfficientNet import Model_3D_EfficientNet


def Obj_fun_CLS(Soln):
    Feat = Global_Vars.Feat
    Target = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    Batch_size = 16
    if dimension == 2:
        for i in range(Soln.shape[0]):
            learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            sol = np.round(Soln[i]).astype('uint8')
            Eval = Model_3D_EfficientNet(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size, sol)
            Fitn[i] = 1 / Eval[7]
        return Fitn
    else:
        sol = np.round(Soln).astype('uint8')
        learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = Model_3D_EfficientNet(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size, sol)
        Fitn = 1 / Eval[7]
        return Fitn

