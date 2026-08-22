import math

def compute_gamma_wang(w_Mn, w_Fe, w_Ni, w_Al, w_C):
    if w_Mn == 0.0:
        w_Mn = 0.0001
    if w_Fe == 0.0:
        w_Fe = 0.0001
    if w_Ni == 0.0:
        w_Ni = 0.0001
    if w_Al == 0.0:
        w_Al = 0.0001
    if w_C == 0.0:
        w_C = 0.0001

    sigma = 0.009
    B_Fe = 0.7
    B_Mn = 0.62
    B_Ni = 0.62
    B_Al = 0.0
    B_FeMn = 0.64
    B_C = 4.0
    dG_C = -22166.0
    Om_FeNi = 2095.0
    Om_FeAl = 3339.0
    Om_FeC = 42500.0
    Om_MnC = 26910.0
    D_FCC = 2.342456517
    D_HCP = 2.342456517
    T = 300.0
    M_Mn = 54.94
    M_Fe = 55.85
    M_Ni = 58.69
    M_Al = 26.98
    M_C = 12.01
    R = 8.314
    # 𝜌: =0.0000294
    rho = 0.0000294
    # 𝛥G Fe: =-2243.38+(4.309*B54)
    dG_Fe = -2243.38+(4.309*T)
    # 𝛥G Mn: =-1000+(1.123*B54)
    dG_Mn = -1000+(1.123*T)
    # 𝛥G Ni: =1046+(1.255*B54)
    dG_Ni = 1046+(1.255*T)
    # 𝛥G Al: =2800+(5*B54)
    dG_Al = 2800+(5*T)
    # a2: =3.6*10^-10*3.6*10^-10
    a2 = 3.6*10**-10*3.6*10**-10
    # N: =6.023*10^23
    N = 6.023*10**23
    # a2*N: =B51*B52
    a2_N = a2*N
    # n Mn: =E2/B55
    n_Mn = w_Mn/M_Mn
    # n Fe: =F2/B56
    n_Fe = w_Fe/M_Fe
    # n Ni: =G2/B57
    n_Ni = w_Ni/M_Ni
    # n Al: =H2/B58
    n_Al = w_Al/M_Al
    # n C: =I2/B59
    n_C = w_C/M_C
    # n total: =B60+B61+B62+B63+B64
    n_total = n_Mn+n_Fe+n_Ni+n_Al+n_C
    # X Fe: =B61/B65
    X_Fe = n_Fe/n_total
    # X Mn: =B60/B65
    X_Mn = n_Mn/n_total
    # X Ni: =B62/B65
    X_Ni = n_Ni/n_total
    # X Al: =B63/B65
    X_Al = n_Al/n_total
    # X C: =B64/B65
    X_C = n_C/n_total
    # ℬ HCP: =(B4*B9)-(B7*B13)
    B_HCP = (X_Mn*B_Mn)-(X_C*B_C)
    # ℬ FCC: =(0.7*B3)+(0.62*B4)+(0.62*B5)-(0.64*B3*B4)-(4*B7)
    B_FCC = (0.7*X_Fe)+(0.62*X_Mn)+(0.62*X_Ni)-(0.64*X_Fe*X_Mn)-(4*X_C)
    # Ω FeMn: =2180+(532*(B3-B4))
    Om_FeMn = 2180+(532*(X_Fe-X_Mn))
    # Tneel HCP: =580*B4
    Tneel_HCP = 580*X_Mn
    # Tneel FCC: =251.71+(681*B4)-(1800*B5)-(1151*B6)-(1740*B7)
    Tneel_FCC = 251.71+(681*X_Mn)-(1800*X_Ni)-(1151*X_Al)-(1740*X_C)
    # 𝜏 FCC: =B54/B27
    tau_FCC = T/Tneel_FCC
    # 𝜏 HCP: =B54/B26
    tau_HCP = T/Tneel_HCP
    # f FCC (𝜏 FCC<1): =1-((1/B28)*(((79*B30^-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((B30^3)/6+(B30^9)/135+(B30^15/600))))
    f_FCC_tau_FCClt1 = 1-((1/D_FCC)*(((79*tau_FCC**-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((tau_FCC**3)/6+(tau_FCC**9)/135+(tau_FCC**15/600))))
    # f FCC (𝜏 FCC>1): =-1/B28*(((B30^-5)/10)+((B30^-15)/315)+((B30^-25)/1500))
    f_FCC_tau_FCCgt1 = -1/D_FCC*(((tau_FCC**-5)/10)+((tau_FCC**-15)/315)+((tau_FCC**-25)/1500))
    # f HCP (𝜏 HCP<1): =1-((1/B29)*(((79*B31^-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((B31^3)/6+(B31^9)/135+(B31^15/600))))
    f_HCP_tau_HCPlt1 = 1-((1/D_HCP)*(((79*tau_HCP**-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((tau_HCP**3)/6+(tau_HCP**9)/135+(tau_HCP**15/600))))
    # f HCP (𝜏 HCP>1): =-1/B29*(((B31^-5)/10)+((B31^-15)/315)+((B31^-25)/1500))
    f_HCP_tau_HCPgt1 = -1/D_HCP*(((tau_HCP**-5)/10)+((tau_HCP**-15)/315)+((tau_HCP**-25)/1500))
    # f FCC: =IF(B30<1,B32,B33)
    f_FCC = (f_FCC_tau_FCClt1 if tau_FCC<1 else f_FCC_tau_FCCgt1)
    # f HCP: =IF(B31<1,B34,B35)
    f_HCP = (f_HCP_tau_HCPlt1 if tau_HCP<1 else f_HCP_tau_HCPgt1)
    # X Fe*𝛥G Fe: =B3*B16
    X_Fe_dG_Fe = X_Fe*dG_Fe
    # X Mn*𝛥G Mn: =B4*B17
    X_Mn_dG_Mn = X_Mn*dG_Mn
    # X Ni*𝛥G Ni: =B5*B18
    X_Ni_dG_Ni = X_Ni*dG_Ni
    # X Al*𝛥G Al: =B6*B19
    X_Al_dG_Al = X_Al*dG_Al
    # X C*𝛥G C: =B7*B20
    X_C_dG_C = X_C*dG_C
    # X Fe*X Mn*Ω FeMn: =B3*B4*B21
    X_Fe_X_Mn_Om_FeMn = X_Fe*X_Mn*Om_FeMn
    # X Fe*X Ni*Ω FeNi: =B3*B5*B22
    X_Fe_X_Ni_Om_FeNi = X_Fe*X_Ni*Om_FeNi
    # X Fe*X Al*Ω FeAl: =B3*B6*B23
    X_Fe_X_Al_Om_FeAl = X_Fe*X_Al*Om_FeAl
    # X Fe*X C*Ω FeC: =B3*B7*B24
    X_Fe_X_C_Om_FeC = X_Fe*X_C*Om_FeC
    # X Mn*X C*Ω MnC: =B4*B7*B25
    X_Mn_X_C_Om_MnC = X_Mn*X_C*Om_MnC
    # 𝛥G Chemical: =SUM(B38:B47)
    dG_Chemical = (X_Fe_dG_Fe + X_Mn_dG_Mn + X_Ni_dG_Ni + X_Al_dG_Al + X_C_dG_C + X_Fe_X_Mn_Om_FeMn + X_Fe_X_Ni_Om_FeNi + X_Fe_X_Al_Om_FeAl + X_Fe_X_C_Om_FeC + X_Mn_X_C_Om_MnC)
    # 𝛥G Magnetic: =((B66*B54*B37*LN(B14+1))-(B66*B54*B36*LN(B15+1)))
    dG_Magnetic = ((R*T*f_HCP*math.log(B_HCP+1))-(R*T*f_FCC*math.log(B_FCC+1)))
    # 𝛄: =2*B1*(B48+B49)+2*B2
    gamma = 2*rho*(dG_Chemical+dG_Magnetic)+2*sigma
    # SFE: =$B$50*1000
    SFE = gamma*1000
    return SFE

def main():
    print("Enter wt.% for Mn, Fe, Ni, Al, C:")
    w_Mn = float(input("Mn: "))
    w_Fe = float(input("Fe: "))
    w_Ni = float(input("Ni: "))
    w_Al = float(input("Al: "))
    w_C = float(input("C: "))

    gamma_mJ = compute_gamma_wang(
        w_Mn,
        w_Fe,
        w_Ni,
        w_Al,
        w_C
    )
    print(f"\nComputed γ = {gamma_mJ:.7f} mJ/m²\n")

if __name__ == "__main__":
    main()
