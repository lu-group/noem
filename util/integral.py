import torch
import time
def Integration1D(U, X, LB, UB):
    TempX = X.squeeze()
    TempU = U.squeeze()
    TempX, Index = torch.sort(TempX)
    TempU = TempU[Index]
    Index1 = TempX >= LB
    Index2 = TempX <= UB
    Index = Index1 * Index2
    TempX = TempX[Index]
    TempU = TempU[Index]
    Integral = torch.trapz(TempU, TempX)
    return Integral

def GetGaussPointWeight(Num):
    if Num <= 1:
        return [1]
    if Num == 2:
        return [0.5, 0.5]
    if Num == 3:
        return [0.277777777775, 0.4444444, 0.277777775]
    if Num == 4:
        return [0.1739274, 0.3260725775, 0.3260725775, 0.1739274]
    if Num == 5:
        return [0.1184634425, 0.239314335, 0.2844444445, 0.239314335, 0.1184634425]
    if Num == 6:
        return [0.085662245, 0.1803807865, 0.233956967, 0.233956967, 0.1803807865, 0.085662245]
    if Num >= 7:
        return [0.064742483, 0.139852696, 0.1909150255, 0.2089795915, 0.1909150255, 0.139852696, 0.064742483]
def GetGaussPointLocation(Num):
    if Num <= 1:
        return [0.5]
    if Num == 2:
        return [0.21132,0.78868]
    if Num == 3:
        return [0.112701666,	0.5,	0.887298335]
    if Num == 4:
        return [0.06943185,	0.33000948,	0.66999052,	0.93056815]
    if Num == 5:
        return [0.0469101,0.230765345,0.5,	0.769234655,	0.9530899]
    if Num == 6:
        return [0.033765243, 	0.169395307, 	0.380690407, 	0.619309593,0.830604693,0.966234757]
    if Num >= 7:
        return [0.025446,0.129234,0.297077,0.5,0.702923,0.870766,0.974554]
def GaussianIntergration1D(U, X, LB, UB, Num_GP=5):
    # ========================================
    # U = U(X) Integrand
    # LB / UB: Lower bound/ Upper bound (n x 1 tensor)
    # Num_GP: Number of Gauss points
    # ========================================
    GPW = torch.tensor(GetGaussPointWeight(Num_GP)).view(Num_GP, 1)
    GPL = torch.tensor(GetGaussPointLocation(Num_GP)).view(Num_GP, 1)
    Length = UB - LB
    for ii in range(len(UB)):
        if ii == 0:
            Loc = LB[ii] + Length[ii] * GPL
            continue
        tLoc = LB[ii] + Length[ii] * GPL
        Loc = torch.cat((Loc, tLoc))
    Sorted_Loc, Loc_Index = torch.sort(Loc, dim=0)
    jj = 0
    tU_Loc = torch.tensor([])
    for ii in range(len(Sorted_Loc)):
        while X[jj].item() < Sorted_Loc[ii].item():
            jj = jj + 1
        tU = (U[jj - 1] + (U[jj] - U[jj - 1]) * (Sorted_Loc[ii] - X[jj - 1]) / (X[jj] - X[jj - 1]))
        tU_Loc = torch.cat((tU_Loc, tU))
    tU_Loc = tU_Loc.view(Num_GP * len(LB), 1)
    U_Loc = torch.zeros((Num_GP * len(LB), 1))
    for ii in range(len(Loc_Index)):
        U_Loc[Loc_Index[ii].item()] = tU_Loc[ii]
    U_Loc = U_Loc.view(int(len(Sorted_Loc) / Num_GP), Num_GP)
    Results = torch.mm(U_Loc, GPW) * Length
    return Results

if __name__ == '__main__':
    n = 50
    x = torch.linspace(0, 2, n)
    x = torch.unsqueeze(x, dim=-1)
    x.requires_grad = True
    u = -x * x * x
    LB = torch.zeros(n,1)
    UB = x
    st = time.time()
    result1 = GaussianIntergration1D(u, x, LB, UB)
    et = time.time()
    print(et-st)
    st = time.time()
    result2 = []
    for ii in range(n):
        result2.append(Integration1D(u, x, 0, UB[ii].item()))
    et = time.time()
    print(et - st)
    result1 = result1.squeeze()
    results2 = torch.tensor(result2)
    diff = result1 - results2
    print(diff)
    # print(result2)




