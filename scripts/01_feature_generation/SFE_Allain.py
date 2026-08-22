import math

def compute_gamma_allain(w_Mn, w_Fe, w_C):
    if w_Mn == 0.0:
        w_Mn = 0.0001
    if w_Fe == 0.0:
        w_Fe = 0.0001
    if w_C == 0.0:
        w_C = 0.0001

    sigma = 0.009
    dG_C = -22166.0
    Om_MnC = 26910.0
    Om_FeC = 42500.0
    D_FCC = 2.34245617
    D_HCP = 2.34245617
    T = 300.0
    M_Mn = 54.94
    M_Fe = 55.85
    M_C = 12.01
    R = 8.314
    # 𝛥G Fe: =-2243.38+ (4.309*B39)
    dG_Fe = -2243.38+ (4.309*T)
    # 𝛥G Mn: =-1000+(1.123*B39)
    dG_Mn = -1000+(1.123*T)
    # 𝑎◦(T): =0.361*10^-9
    a0_T = 0.361*10**-9
    # a2: =B35^2
    a2 = a0_T**2
    # N: =6.023*10^23
    N = 6.023*10**23
    # a2*N: =B36*B37
    a2_N = a2*N
    # n Mn: =E2/B40
    n_Mn = w_Mn/M_Mn
    # n Fe: =F2/B41
    n_Fe = w_Fe/M_Fe
    # n C: =G2/B42
    n_C = w_C/M_C
    # n total: =B43+B44+B45
    n_total = n_Mn+n_Fe+n_C
    # 𝜌: =(2.3094/B38)
    rho = (2.3094/a2_N)
    # X Fe: =B44/B46
    X_Fe = n_Fe/n_total
    # X Mn: =B43/B46
    X_Mn = n_Mn/n_total
    # X C: =B45/B46
    X_C = n_C/n_total
    # ℬ HCP: =0.62*B4
    B_HCP = 0.62*X_Mn
    # ℬ FCC: =(0.7*B3)+(0.62*B4)-(0.5*B3*B4)
    B_FCC = (0.7*X_Fe)+(0.62*X_Mn)-(0.5*X_Fe*X_Mn)
    # Ω FeMn: =2180+(532*(B3-B4))
    Om_FeMn = 2180+(532*(X_Fe-X_Mn))
    # Tneel HCP: =580*B4
    Tneel_HCP = 580*X_Mn
    # Tneel FCC: =(669.27*(1-EXP(-5.46*B4)))-((0.241*B5)*(1-B4-B5))-109
    Tneel_FCC = (669.27*(1-math.exp(-5.46*X_Mn)))-((0.241*X_C)*(1-X_Mn-X_C))-109
    # 𝜏 FCC: =B39/B15
    tau_FCC = T/Tneel_FCC
    # 𝜏 HCP: =B39/B14
    tau_HCP = T/Tneel_HCP
    # f FCC (𝜏 FCC<1): =1-((1/B16)*(((79*B18^-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((B18^3)/6+(B18^9)/135+(B18^15/600))))
    f_FCC_tau_FCClt1 = 1-((1/D_FCC)*(((79*tau_FCC**-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((tau_FCC**3)/6+(tau_FCC**9)/135+(tau_FCC**15/600))))
    # f FCC (𝜏 FCC>1): =-1/B16*(((B18^-5)/10)+((B18^-15)/315)+((B18^-25)/1500))
    f_FCC_tau_FCCgt1 = -1/D_FCC*(((tau_FCC**-5)/10)+((tau_FCC**-15)/315)+((tau_FCC**-25)/1500))
    # f HCP (𝜏 HCP<1): =1-((1/B17)*(((79*B19^-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((B19^3)/6+(B19^9)/135+(B19^15/600))))
    f_HCP_tau_HCPlt1 = 1-((1/D_HCP)*(((79*tau_HCP**-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((tau_HCP**3)/6+(tau_HCP**9)/135+(tau_HCP**15/600))))
    # f HCP (𝜏 HCP>1): =-1/B17*(((B19^-5)/10)+((B19^-15)/315)+((B19^-25)/1500))
    f_HCP_tau_HCPgt1 = -1/D_HCP*(((tau_HCP**-5)/10)+((tau_HCP**-15)/315)+((tau_HCP**-25)/1500))
    # f FCC: =IF(B18<1,B20,B21)
    f_FCC = (f_FCC_tau_FCClt1 if tau_FCC<1 else f_FCC_tau_FCCgt1)
    # f HCP: =IF(B19<1,B22,B23)
    f_HCP = (f_HCP_tau_HCPlt1 if tau_HCP<1 else f_HCP_tau_HCPgt1)
    # X Fe*𝛥G Fe: =B3*B8
    X_Fe_dG_Fe = X_Fe*dG_Fe
    # X Mn*𝛥G Mn: =B4*B9
    X_Mn_dG_Mn = X_Mn*dG_Mn
    # X C*𝛥G C: =B5*B10
    X_C_dG_C = X_C*dG_C
    # X Fe*X Mn*Ω FeMn: =B3*B4*B11
    X_Fe_X_Mn_Om_FeMn = X_Fe*X_Mn*Om_FeMn
    # X Fe*X C*Ω FeC: =B3*B5*B13
    X_Fe_X_C_Om_FeC = X_Fe*X_C*Om_FeC
    # X Mn*X C*Ω MnC: =B4*B5*B12
    X_Mn_X_C_Om_MnC = X_Mn*X_C*Om_MnC
    # 𝛥G Chemical: =SUM(B26,B27,B28,B29,B30,B31)
    dG_Chemical = (X_Fe_dG_Fe + X_Mn_dG_Mn + X_C_dG_C + X_Fe_X_Mn_Om_FeMn + X_Fe_X_C_Om_FeC + X_Mn_X_C_Om_MnC)
    # 𝛥G Magnetic: =(B47*B39*B25*LN(B6+1))-(B47*B39*B24*LN(B7+1))
    dG_Magnetic = (R*T*f_HCP*math.log(B_HCP+1))-(R*T*f_FCC*math.log(B_FCC+1))
    # 𝛄: =2*B1*SUM(B32:B33)+(2*B2)
    gamma = 2*rho*(dG_Chemical + dG_Magnetic)+(2*sigma)
    # SFE: =$B$34*1000
    SFE = gamma*1000
    return SFE

def main():
    print("Enter wt.% for Mn, Fe, C:")
    w_Mn = float(input("Mn: "))
    w_Fe = float(input("Fe: "))
    w_C = float(input("C: "))

    gamma_mJ = compute_gamma_allain(
        w_Mn,
        w_Fe,
        w_C
    )
    print(f"\nComputed γ = {gamma_mJ:.7f} mJ/m²\n")

if __name__ == "__main__":
    main()
