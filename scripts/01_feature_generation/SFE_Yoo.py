import math

def compute_gamma_yoo(w_Mn, w_Fe, w_Al, w_C):
    if w_Mn == 0.0:
        w_Mn = 0.0001
    if w_Fe == 0.0:
        w_Fe = 0.0001
    if w_Al == 0.0:
        w_Al = 0.0001
    if w_C == 0.0:
        w_C = 0.0001

    sigma = 0.009
    B_Fe = 0.7
    B_Mn = 0.62
    B_Al = 0.0
    B_FeMn = 0.64
    B_C = 4.0
    dG_C = -22166.0
    Om_FeAl = 3339.0
    Om_FeC = 42500.0
    Om_MnC = 26910.0
    D_FCC = 2.34245617
    D_HCP = 2.34245617
    T = 300.0
    d = 40.0
    M_Mn = 54.94
    M_Fe = 55.85
    M_Al = 26.98
    M_C = 12.01
    R = 8.314
    # 𝜌: =2.94*10^-5
    rho = 2.94*10**-5
    # 𝛥G Fe: =-2243.38+(4.309*B46)
    dG_Fe = -2243.38+(4.309*T)
    # 𝛥G Mn: =-1000+(1.123*B46)
    dG_Mn = -1000+(1.123*T)
    # 𝛥G Al: =2800+(5*B46)
    dG_Al = 2800+(5*T)
    # 𝛥G ex: =170.06*EXP(-B47/18.55)
    dG_ex = 170.06*math.exp(-d/18.55)
    # n Mn: =E2/B48
    n_Mn = w_Mn/M_Mn
    # n Fe: =F2/B49
    n_Fe = w_Fe/M_Fe
    # n Al: =G2/B50
    n_Al = w_Al/M_Al
    # n C: =H2/B51
    n_C = w_C/M_C
    # n total: =SUM(B52:B55)
    n_total = (n_Mn + n_Fe + n_Al + n_C)
    # X Fe: =B53/B56
    X_Fe = n_Fe/n_total
    # X Mn: =B52/B56
    X_Mn = n_Mn/n_total
    # X Al: =B54/B56
    X_Al = n_Al/n_total
    # X C: =B55/B56
    X_C = n_C/n_total
    # ℬ HCP: =(B4*B8)-(B6*B11)
    B_HCP = (X_Mn*B_Mn)-(X_C*B_C)
    # ℬ FCC: =(B3*B7)+(B4*B8)-(B6*B11)-(B10*B3*B4)
    B_FCC = (X_Fe*B_Fe)+(X_Mn*B_Mn)-(X_C*B_C)-(B_FeMn*X_Fe*X_Mn)
    # Ω FeMn: =2180+(532*(B3-B4))
    Om_FeMn = 2180+(532*(X_Fe-X_Mn))
    # Tneel HCP: =580*B4
    Tneel_HCP = 580*X_Mn
    # Tneel FCC: =(250*LN(B4))-(4750*B6*B4)-(6.2*B5)+720
    Tneel_FCC = (250*math.log(X_Mn))-(4750*X_C*X_Mn)-(6.2*X_Al)+720
    # 𝜏 FCC: =B46/B23
    tau_FCC = T/Tneel_FCC
    # 𝜏 HCP: =B46/B22
    tau_HCP = T/Tneel_HCP
    # f FCC (𝜏 FCC<1): =1-((1/B24)*(((79*B26^-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((B26^3)/6+(B26^9)/135+(B26^15/600))))
    f_FCC_tau_FCClt1 = 1-((1/D_FCC)*(((79*tau_FCC**-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((tau_FCC**3)/6+(tau_FCC**9)/135+(tau_FCC**15/600))))
    # f FCC (𝜏 FCC>1): =-1/B24*(((B26^-5)/10)+((B26^-15)/315)+((B26^-25)/1500))
    f_FCC_tau_FCCgt1 = -1/D_FCC*(((tau_FCC**-5)/10)+((tau_FCC**-15)/315)+((tau_FCC**-25)/1500))
    # f HCP (𝜏 HCP<1): =1-((1/B25)*(((79*B27^-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((B27^3)/6+(B27^9)/135+(B27^15/600))))
    f_HCP_tau_HCPlt1 = 1-((1/D_HCP)*(((79*tau_HCP**-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((tau_HCP**3)/6+(tau_HCP**9)/135+(tau_HCP**15/600))))
    # f HCP (𝜏 HCP>1): =-1/B25*(((B27^-5)/10)+((B27^-15)/315)+((B27^-25)/1500))
    f_HCP_tau_HCPgt1 = -1/D_HCP*(((tau_HCP**-5)/10)+((tau_HCP**-15)/315)+((tau_HCP**-25)/1500))
    # f FCC: =IF(B26<1,B28,B29)
    f_FCC = (f_FCC_tau_FCClt1 if tau_FCC<1 else f_FCC_tau_FCCgt1)
    # f HCP: =IF(B27<1,B30,B31)
    f_HCP = (f_HCP_tau_HCPlt1 if tau_HCP<1 else f_HCP_tau_HCPgt1)
    # X Fe*𝛥G Fe: =B3*B14
    X_Fe_dG_Fe = X_Fe*dG_Fe
    # X Mn*𝛥G Mn: =B4*B15
    X_Mn_dG_Mn = X_Mn*dG_Mn
    # X Al*𝛥G Al: =B5*B16
    X_Al_dG_Al = X_Al*dG_Al
    # X C*𝛥G C: =B6*B17
    X_C_dG_C = X_C*dG_C
    # X Fe*X Mn*Ω FeMn: =B3*B4*B18
    X_Fe_X_Mn_Om_FeMn = X_Fe*X_Mn*Om_FeMn
    # X Fe*X Al*Ω FeAl: =B3*B5*B19
    X_Fe_X_Al_Om_FeAl = X_Fe*X_Al*Om_FeAl
    # X Fe*X C*Ω FeC: =B3*B6*B20
    X_Fe_X_C_Om_FeC = X_Fe*X_C*Om_FeC
    # X Mn*X C*Ω MnC: =B4*B6*B21
    X_Mn_X_C_Om_MnC = X_Mn*X_C*Om_MnC
    # 𝛥G Chemical: =SUM(B34:B41)
    dG_Chemical = (X_Fe_dG_Fe + X_Mn_dG_Mn + X_Al_dG_Al + X_C_dG_C + X_Fe_X_Mn_Om_FeMn + X_Fe_X_Al_Om_FeAl + X_Fe_X_C_Om_FeC + X_Mn_X_C_Om_MnC)
    # 𝛥G Magnetic: =((B57*B46*B33*LN(B12+1))-(B57*B46*B32*LN(B13+1)))
    dG_Magnetic = ((R*T*f_HCP*math.log(B_HCP+1))-(R*T*f_FCC*math.log(B_FCC+1)))
    # 𝛄: =2*B1*SUM(B42:B44)+(2*B2)
    gamma = 2*rho*(dG_Chemical + dG_Magnetic + dG_ex)+(2*sigma)
    # SFE: =$B$45*1000
    SFE = gamma*1000
    return SFE

def main():
    print("Enter wt.% for Mn, Fe, Al, C:")
    w_Mn = float(input("Mn: "))
    w_Fe = float(input("Fe: "))
    w_Al = float(input("Al: "))
    w_C = float(input("C: "))

    gamma_mJ = compute_gamma_yoo(
        w_Mn,
        w_Fe,
        w_Al,
        w_C
    )
    print(f"\nComputed γ = {gamma_mJ:.7f} mJ/m²\n")

if __name__ == "__main__":
    main()
