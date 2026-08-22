import math

def compute_gamma_xiong(w_Mn, w_Fe, w_Si, w_C):
    if w_Mn == 0.0:
        w_Mn = 0.0001
    if w_Fe == 0.0:
        w_Fe = 0.0001
    if w_Si == 0.0:
        w_Si = 0.0001
    if w_C == 0.0:
        w_C = 0.0001

    sigma = 0.009
    dG_Fe = -950.68
    dG_Mn = -663.1
    dG_Si = -2960.0
    dG_C = -22166.0
    Om_FeC = 42500.0
    Om_MnC = 26910.0
    a_FCC = 3.6e-10
    N = 6.023e+23
    T = 300.0
    M_Mn = 54.94
    M_Fe = 55.85
    M_Si = 28.09
    M_C = 12.01
    R = 8.314
    # D γ: =(518/1125)+(11692/15975)*(1/0.28-1)
    D_gamma = (518/1125)+(11692/15975)*(1/0.28-1)
    # D ε: =(518/1125)+(11692/15975)*(1/0.28-1)
    D_epsilon = (518/1125)+(11692/15975)*(1/0.28-1)
    # a2: =B41^2
    a2 = a_FCC**2
    # a2*N: =B42*B43
    a2_N = a2*N
    # n Mn: =E2/B46
    n_Mn = w_Mn/M_Mn
    # n Fe: =F2/B47
    n_Fe = w_Fe/M_Fe
    # n Si: =G2/B48
    n_Si = w_Si/M_Si
    # n C: =H2/B49
    n_C = w_C/M_C
    # n total: =SUM(B50:B53)
    n_total = (n_Mn + n_Fe + n_Si + n_C)
    # 𝜌: =2.3094/B44
    rho = 2.3094/a2_N
    # X Fe: =B51/B54
    X_Fe = n_Fe/n_total
    # X Mn: =B50/B54
    X_Mn = n_Mn/n_total
    # X Si: =B52/B54
    X_Si = n_Si/n_total
    # X C: =B53/B54
    X_C = n_C/n_total
    # β ε / μB: =(0.7*B3)+(0.62*B4)-(0.64*B3*B4)-(4*B6)
    beta_epsilon_muB = (0.7*X_Fe)+(0.62*X_Mn)-(0.64*X_Fe*X_Mn)-(4*X_C)
    # β γ / μB: =(0.62*B4)-(4*B6)
    beta_gamma_muB = (0.62*X_Mn)-(4*X_C)
    # Ω FeMn: =2873-(717*(B3-B4))
    Om_FeMn = 2873-(717*(X_Fe-X_Mn))
    # Ω FeSi: =2850+(3520*(B3-B5))
    Om_FeSi = 2850+(3520*(X_Fe-X_Si))
    # Tneel ε: =580*B4
    Tneel_epsilon = 580*X_Mn
    # Tneel γ: =(10*B4^3)+(898.4*B4^2)+(1176*B4)-(1992*B6)-(1272*B5)+152.4
    Tneel_gamma = (10*X_Mn**3)+(898.4*X_Mn**2)+(1176*X_Mn)-(1992*X_C)-(1272*X_Si)+152.4
    # 𝜏 γ: =B45/B18
    tau_gamma = T/Tneel_gamma
    # 𝜏 ε: =B45/B17
    tau_epsilon = T/Tneel_epsilon
    # f γ (𝜏 γ<1): =1-((1/B19)*((79*B21^-1)/(140*0.28)+(474/497)*((1/0.28)-1)*((B21^3)/6+(B21^9)/135+(B21^15)/600)))
    f_gamma_tau_gammalt1 = 1-((1/D_gamma)*((79*tau_gamma**-1)/(140*0.28)+(474/497)*((1/0.28)-1)*((tau_gamma**3)/6+(tau_gamma**9)/135+(tau_gamma**15)/600)))
    # f γ (𝜏 γ>1): =-1/B19*(((B21^-5)/10)+((B21^-15)/315)+((B21^-25)/1500))
    f_gamma_tau_gammagt1 = -1/D_gamma*(((tau_gamma**-5)/10)+((tau_gamma**-15)/315)+((tau_gamma**-25)/1500))
    # f ε (𝜏 ε<1): =1-((1/B20)*((79*B22^-1)/(140*0.28)+(474/497)*((1/0.28)-1)*((B22^3)/6+(B22^9)/135+(B22^15)/600)))
    f_epsilon_tau_epsilonlt1 = 1-((1/D_epsilon)*((79*tau_epsilon**-1)/(140*0.28)+(474/497)*((1/0.28)-1)*((tau_epsilon**3)/6+(tau_epsilon**9)/135+(tau_epsilon**15)/600)))
    # f ε (𝜏 ε>1): =-1/B20*(((B22^-5)/10)+((B22^-15)/315)+((B22^-25)/1500))
    f_epsilon_tau_epsilongt1 = -1/D_epsilon*(((tau_epsilon**-5)/10)+((tau_epsilon**-15)/315)+((tau_epsilon**-25)/1500))
    # f γ: =IF(B21<1,B23,B24)
    f_gamma = (f_gamma_tau_gammalt1 if tau_gamma<1 else f_gamma_tau_gammagt1)
    # f ε: =IF(B22<1,B25,B26)
    f_epsilon = (f_epsilon_tau_epsilonlt1 if tau_epsilon<1 else f_epsilon_tau_epsilongt1)
    # X Fe*𝛥G Fe: =B3*B9
    X_Fe_dG_Fe = X_Fe*dG_Fe
    # X Mn*𝛥G Mn: =B4*B10
    X_Mn_dG_Mn = X_Mn*dG_Mn
    # X Si*𝛥G Si: =B5*B11
    X_Si_dG_Si = X_Si*dG_Si
    # X C*𝛥G C: =B6*B12
    X_C_dG_C = X_C*dG_C
    # X Fe*X Mn*Ω FeMn: =B3*B4*B13
    X_Fe_X_Mn_Om_FeMn = X_Fe*X_Mn*Om_FeMn
    # X Fe*X Si*Ω FeSi: =B3*B5*B14
    X_Fe_X_Si_Om_FeSi = X_Fe*X_Si*Om_FeSi
    # X Fe*X C*Ω FeC: =B3*B6*B15
    X_Fe_X_C_Om_FeC = X_Fe*X_C*Om_FeC
    # X Mn*X C*Ω MnC: =B4*B6*B16
    X_Mn_X_C_Om_MnC = X_Mn*X_C*Om_MnC
    # 𝛥G Chemical: =SUM(B29:B36)
    dG_Chemical = (X_Fe_dG_Fe + X_Mn_dG_Mn + X_Si_dG_Si + X_C_dG_C + X_Fe_X_Mn_Om_FeMn + X_Fe_X_Si_Om_FeSi + X_Fe_X_C_Om_FeC + X_Mn_X_C_Om_MnC)
    # 𝛥G Magnetic: =(B55*B45*B28*LN(B7+1))-(B55*B45*B27*LN(B8+1))
    dG_Magnetic = (R*T*f_epsilon*math.log(beta_epsilon_muB+1))-(R*T*f_gamma*math.log(beta_gamma_muB+1))
    # 𝛥G Total: =B37+B38
    dG_Total = dG_Chemical+dG_Magnetic
    # 𝛄: =2*B1*B39+2*B2
    gamma = 2*rho*dG_Total+2*sigma
    # SFE: =$B$40*1000
    SFE = gamma*1000
    return SFE

def main():
    print("Enter wt.% for Mn, Fe, Si, C:")
    w_Mn = float(input("Mn: "))
    w_Fe = float(input("Fe: "))
    w_Si = float(input("Si: "))
    w_C = float(input("C: "))

    gamma_mJ = compute_gamma_xiong(
        w_Mn,
        w_Fe,
        w_Si,
        w_C
    )
    print(f"\nComputed γ = {gamma_mJ:.7f} mJ/m²\n")

if __name__ == "__main__":
    main()
