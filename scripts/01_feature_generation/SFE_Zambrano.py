import math

def compute_gamma_zambrano(w_Mn, w_Fe, w_Al, w_Si, w_C):
    if w_Mn == 0.0:
        w_Mn = 0.0001
    if w_Fe == 0.0:
        w_Fe = 0.0001
    if w_Al == 0.0:
        w_Al = 0.0001
    if w_Si == 0.0:
        w_Si = 0.0001
    if w_C == 0.0:
        w_C = 0.0001

    sigma = 0.01
    B_Fe = 0.7
    B_Mn = 0.62
    B_Al = 0.0
    B_FeMn = 0.64
    B_C = 4.0
    dG_C = -22166.0
    Om_FeAl = 3326.28
    Om_FeC = 42500.0
    Om_FeSi = 1780.0
    D_FCC = 2.34245617
    D_HCP = 2.34245617
    T = 300.0
    d = 40.0
    M_Mn = 54.94
    M_Fe = 55.85
    M_Al = 26.98
    M_Si = 28.09
    M_C = 12.01
    R = 8.314
    # 𝛥G Fe: =-1828.4+ (4.686*B54)
    dG_Fe = -1828.4+ (4.686*T)
    # 𝛥G Mn: =3970-(1.6667*B54)
    dG_Mn = 3970-(1.6667*T)
    # 𝛥G Al: =5481.04-(1.79912*B54)
    dG_Al = 5481.04-(1.79912*T)
    # 𝛥G Si: =IF(H2<3,-560-(8*B54),-1800+B54)
    dG_Si = (-560-(8*T) if w_Si<3 else -1800+T)
    # 𝛥G ex: =170.06*EXP(-B55/18.55)
    dG_ex = 170.06*math.exp(-d/18.55)
    # 𝑎◦(FCC): =3.5945+0.00125*(E2-20)+0.00594*G2+0.0272*I2
    a0_FCC = 3.5945+0.00125*(w_Mn-20)+0.00594*w_Al+0.0272*w_C
    # 𝑎◦(T): =B49*(1+2.065*10^-9*(B54-300))*10^-10
    a0_T = a0_FCC*(1+2.065*10**-9*(T-300))*10**-10
    # a2: =B50^2
    a2 = a0_T**2
    # N: =6.023*10^23
    N = 6.023*10**23
    # a2*N: =B51*B52
    a2_N = a2*N
    # n Mn: =E2/B56
    n_Mn = w_Mn/M_Mn
    # n Fe: =F2/B57
    n_Fe = w_Fe/M_Fe
    # n Al: =G2/B58
    n_Al = w_Al/M_Al
    # n Si: =H2/B59
    n_Si = w_Si/M_Si
    # n C: =I2/B60
    n_C = w_C/M_C
    # n total: =SUM(B61:B65)
    n_total = (n_Mn + n_Fe + n_Al + n_Si + n_C)
    # 𝜌: =(2.3094/B53)
    rho = (2.3094/a2_N)
    # X Fe: =B62/B66
    X_Fe = n_Fe/n_total
    # X Mn: =B61/B66
    X_Mn = n_Mn/n_total
    # X Al: =B63/B66
    X_Al = n_Al/n_total
    # X Si: =B64/B66
    X_Si = n_Si/n_total
    # X C: =B65/B66
    X_C = n_C/n_total
    # ℬ HCP: =(B4*B9)-(B7*B12)
    B_HCP = (X_Mn*B_Mn)-(X_C*B_C)
    # ℬ FCC: =(B3*B8)+(B4*B9)-(B7*B12)-(B11*B3*B4)
    B_FCC = (X_Fe*B_Fe)+(X_Mn*B_Mn)-(X_C*B_C)-(B_FeMn*X_Fe*X_Mn)
    # Ω FeMn: =-9135.5+(15282.1*B4)
    Om_FeMn = -9135.5+(15282.1*X_Mn)
    # Tneel HCP: =580*B4
    Tneel_HCP = 580*X_Mn
    # Tneel FCC: =(0.00001*B4^3)+(0.08984*B4^2)+(11.76*B4)-(19.92*B7)-(12.72*B6)-(6.61*B5)+152.4
    Tneel_FCC = (0.00001*X_Mn**3)+(0.08984*X_Mn**2)+(11.76*X_Mn)-(19.92*X_C)-(12.72*X_Si)-(6.61*X_Al)+152.4
    # 𝜏 FCC: =B54/B25
    tau_FCC = T/Tneel_FCC
    # 𝜏 HCP: =B54/B24
    tau_HCP = T/Tneel_HCP
    # f FCC (𝜏 FCC<1): =1-((1/B26)*(((79*B28^-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((B28^3)/6+(B28^9)/135+(B28^15/600))))
    f_FCC_tau_FCClt1 = 1-((1/D_FCC)*(((79*tau_FCC**-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((tau_FCC**3)/6+(tau_FCC**9)/135+(tau_FCC**15/600))))
    # f FCC (𝜏 FCC>1): =-1/B26*(((B28^-5)/10)+((B28^-15)/315)+((B28^-25)/1500))
    f_FCC_tau_FCCgt1 = -1/D_FCC*(((tau_FCC**-5)/10)+((tau_FCC**-15)/315)+((tau_FCC**-25)/1500))
    # f HCP (𝜏 HCP<1): =1-((1/B27)*(((79*B29^-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((B29^3)/6+(B29^9)/135+(B29^15/600))))
    f_HCP_tau_HCPlt1 = 1-((1/D_HCP)*(((79*tau_HCP**-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((tau_HCP**3)/6+(tau_HCP**9)/135+(tau_HCP**15/600))))
    # f HCP (𝜏 HCP>1): =-1/B27*(((B29^-5)/10)+((B29^-15)/315)+((B29^-25)/1500))
    f_HCP_tau_HCPgt1 = -1/D_HCP*(((tau_HCP**-5)/10)+((tau_HCP**-15)/315)+((tau_HCP**-25)/1500))
    # f FCC: =IF(B28<=1,B30,B31)
    f_FCC = (f_FCC_tau_FCClt1 if tau_FCC<=1 else f_FCC_tau_FCCgt1)
    # f HCP: =IF(B29<=1,B32,B33)
    f_HCP = (f_HCP_tau_HCPlt1 if tau_HCP<=1 else f_HCP_tau_HCPgt1)
    # X Fe*𝛥G Fe: =B3*B15
    X_Fe_dG_Fe = X_Fe*dG_Fe
    # X Mn*𝛥G Mn: =B4*B16
    X_Mn_dG_Mn = X_Mn*dG_Mn
    # X Al*𝛥G Al: =B5*B17
    X_Al_dG_Al = X_Al*dG_Al
    # X Si*𝛥G Si: =B6*B18
    X_Si_dG_Si = X_Si*dG_Si
    # X C*𝛥G C: =B7*B19
    X_C_dG_C = X_C*dG_C
    # X Fe*X Mn*Ω FeMn: =B3*B4*B20
    X_Fe_X_Mn_Om_FeMn = X_Fe*X_Mn*Om_FeMn
    # X Fe*X Al*Ω FeAl: =B3*B5*B21
    X_Fe_X_Al_Om_FeAl = X_Fe*X_Al*Om_FeAl
    # X Fe*X C*Ω FeC: =B3*B7*B22
    X_Fe_X_C_Om_FeC = X_Fe*X_C*Om_FeC
    # X Fe*X Si*Ω FeSi: =B3*B6*B23
    X_Fe_X_Si_Om_FeSi = X_Fe*X_Si*Om_FeSi
    # 𝛥G Chemical: =SUM(B36,B37,B38,B39,B40,B41,B42,B43,B44)
    dG_Chemical = (X_Fe_dG_Fe + X_Mn_dG_Mn + X_Al_dG_Al + X_Si_dG_Si + X_C_dG_C + X_Fe_X_Mn_Om_FeMn + X_Fe_X_Al_Om_FeAl + X_Fe_X_C_Om_FeC + X_Fe_X_Si_Om_FeSi)
    # 𝛥G Magnetic: =((B67*B54*B35*LN(B13+1))-(B67*B54*B34*LN(B14+1)))
    dG_Magnetic = ((R*T*f_HCP*math.log(B_HCP+1))-(R*T*f_FCC*math.log(B_FCC+1)))
    # 𝛄: =2*B1*SUM(B45:B47)+(2*B2)
    gamma = 2*rho*(dG_Chemical + dG_Magnetic + dG_ex)+(2*sigma)
    # SFE: =$B$48*1000
    SFE = gamma*1000
    return SFE

def main():
    print("Enter wt.% for Mn, Fe, Al, Si, C:")
    w_Mn = float(input("Mn: "))
    w_Fe = float(input("Fe: "))
    w_Al = float(input("Al: "))
    w_Si = float(input("Si: "))
    w_C = float(input("C: "))

    gamma_mJ = compute_gamma_zambrano(
        w_Mn,
        w_Fe,
        w_Al,
        w_Si,
        w_C
    )
    print(f"\nComputed γ = {gamma_mJ:.7f} mJ/m²\n")

if __name__ == "__main__":
    main()
