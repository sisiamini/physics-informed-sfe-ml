import math

def compute_gamma_akbari(w_Mo, w_Cu, w_V, w_Mn, w_Fe, w_Ni, w_Al, w_Cr, w_Si, w_C):
    if w_Mo == 0.0:
        w_Mo = 0.0001
    if w_Cu == 0.0:
        w_Cu = 0.0001
    if w_V == 0.0:
        w_V = 0.0001
    if w_Mn == 0.0:
        w_Mn = 0.0001
    if w_Fe == 0.0:
        w_Fe = 0.0001
    if w_Ni == 0.0:
        w_Ni = 0.0001
    if w_Al == 0.0:
        w_Al = 0.0001
    if w_Cr == 0.0:
        w_Cr = 0.0001
    if w_Si == 0.0:
        w_Si = 0.0001
    if w_C == 0.0:
        w_C = 0.0001

    sigma = 0.015
    B_Fe = 0.7
    B_Mn = 0.62
    B_Al = 0.0
    B_FeMn = 0.64
    B_C = 4.0
    dG_C = -24595.12
    Om_FeAl = 3323.0
    Om_FeC = 42500.0
    Om_MnC = 26910.0
    D_FCC = 2.34245617
    D_HCP = 2.34245617
    T = 300.0
    d = 40.0
    M_Mn = 54.94
    M_Fe = 55.85
    M_Ni = 58.69
    M_Al = 26.98
    M_Cr = 52.0
    M_Si = 28.09
    M_Cu = 63.546
    M_V = 50.9415
    M_Mo = 95.95
    M_C = 12.01
    R = 8.314
    # 𝛥G Fe: =-1828.4+ (4.686*B57)
    dG_Fe = -1828.4+ (4.686*T)
    # 𝛥G Mn: =3970-(1.7*B57)
    dG_Mn = 3970-(1.7*T)
    # 𝛥G Al: =5481.04-(1.799*B57)
    dG_Al = 5481.04-(1.799*T)
    # 𝛥G ex: =170.06*EXP(-B58/18.55)
    dG_ex = 170.06*math.exp(-d/18.55)
    # N: =6.023*10^23
    N = 6.023*10**23
    # n Mn: =H2/B59
    n_Mn = w_Mn/M_Mn
    # n Fe: =I2/B60
    n_Fe = w_Fe/M_Fe
    # n Ni: =J2/B61
    n_Ni = w_Ni/M_Ni
    # n Al: =K2/B62
    n_Al = w_Al/M_Al
    # n Cr: =L2/B63
    n_Cr = w_Cr/M_Cr
    # n Si: =M2/B64
    n_Si = w_Si/M_Si
    # n Cu: =F2/B65
    n_Cu = w_Cu/M_Cu
    # n V: =G2/B66
    n_V = w_V/M_V
    # n Mo: =E2/B67
    n_Mo = w_Mo/M_Mo
    # n C: =N2/B68
    n_C = w_C/M_C
    # n total: =SUM(B69:B78)
    n_total = (n_Mn + n_Fe + n_Ni + n_Al + n_Cr + n_Si + n_Cu + n_V + n_Mo + n_C)
    # X Fe: =B70/B79
    X_Fe = n_Fe/n_total
    # X Mn: =B69/B79
    X_Mn = n_Mn/n_total
    # X Al: =B72/B79
    X_Al = n_Al/n_total
    # X Cr: =B73/B79
    X_Cr = n_Cr/n_total
    # X Ni: =B71/B79
    X_Ni = n_Ni/n_total
    # X Cu: =B75/B79
    X_Cu = n_Cu/n_total
    # X V: =B76/B79
    X_V = n_V/n_total
    # X Si: =B74/B79
    X_Si = n_Si/n_total
    # X Mo: =B77/B79
    X_Mo = n_Mo/n_total
    # X C: =B78/B79
    X_C = n_C/n_total
    # ℬ HCP: =(B4*B14)-(B12*B17)
    B_HCP = (X_Mn*B_Mn)-(X_C*B_C)
    # ℬ FCC: =(B3*B13)+(B4*B14)-(B12*B17)-(B16*B3*B4)
    B_FCC = (X_Fe*B_Fe)+(X_Mn*B_Mn)-(X_C*B_C)-(B_FeMn*X_Fe*X_Mn)
    # Ω FeMn: =-9135.5+(15282.1*B4)
    Om_FeMn = -9135.5+(15282.1*X_Mn)
    # Tneel HCP: =580*B4
    Tneel_HCP = 580*X_Mn
    # Tneel FCC: =(250*LN(B4))-(4750*B12*B4)-(222*B8)-(2.6*B6)-(6.2*B5)-(13*B10)+720
    Tneel_FCC = (250*math.log(X_Mn))-(4750*X_C*X_Mn)-(222*X_Cu)-(2.6*X_Cr)-(6.2*X_Al)-(13*X_Si)+720
    # 𝜏 FCC: =B57/B29
    tau_FCC = T/Tneel_FCC
    # 𝜏 HCP: =B57/B28
    tau_HCP = T/Tneel_HCP
    # f FCC (𝜏 FCC<1): =1-((1/B30)*(((79*B32^-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((B32^3)/6+(B32^9)/135+(B32^15/600))))
    f_FCC_tau_FCClt1 = 1-((1/D_FCC)*(((79*tau_FCC**-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((tau_FCC**3)/6+(tau_FCC**9)/135+(tau_FCC**15/600))))
    # f FCC (𝜏 FCC>1): =-1/B30*(((B32^-5)/10)+((B32^-15)/315)+((B32^-25)/1500))
    f_FCC_tau_FCCgt1 = -1/D_FCC*(((tau_FCC**-5)/10)+((tau_FCC**-15)/315)+((tau_FCC**-25)/1500))
    # f HCP (𝜏 HCP<1): =1-((1/B31)*(((79*B33^-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((B33^3)/6+(B33^9)/135+(B33^15/600))))
    f_HCP_tau_HCPlt1 = 1-((1/D_HCP)*(((79*tau_HCP**-1)/(140*0.28))+(474/497)*((1/0.28)-1)*((tau_HCP**3)/6+(tau_HCP**9)/135+(tau_HCP**15/600))))
    # f HCP (𝜏 HCP>1): =-1/B31*(((B33^-5)/10)+((B33^-15)/315)+((B33^-25)/1500))
    f_HCP_tau_HCPgt1 = -1/D_HCP*(((tau_HCP**-5)/10)+((tau_HCP**-15)/315)+((tau_HCP**-25)/1500))
    # f FCC: =IF(B32<1,B34,B35)
    f_FCC = (f_FCC_tau_FCClt1 if tau_FCC<1 else f_FCC_tau_FCCgt1)
    # f HCP: =IF(B33<1,B36,B37)
    f_HCP = (f_HCP_tau_HCPlt1 if tau_HCP<1 else f_HCP_tau_HCPgt1)
    # X Fe*𝛥G Fe: =B3*B20
    X_Fe_dG_Fe = X_Fe*dG_Fe
    # X Mn*𝛥G Mn: =B4*B21
    X_Mn_dG_Mn = X_Mn*dG_Mn
    # X Al*𝛥G Al: =B5*B22
    X_Al_dG_Al = X_Al*dG_Al
    # X C*𝛥G C: =B12*B23
    X_C_dG_C = X_C*dG_C
    # X Fe*X Mn*Ω FeMn: =B3*B4*B24
    X_Fe_X_Mn_Om_FeMn = X_Fe*X_Mn*Om_FeMn
    # X Fe*X Al*Ω FeAl: =B3*B5*B25
    X_Fe_X_Al_Om_FeAl = X_Fe*X_Al*Om_FeAl
    # X Fe*X C*Ω FeC: =B3*B12*B26
    X_Fe_X_C_Om_FeC = X_Fe*X_C*Om_FeC
    # X Mn*X C*Ω MnC: =B4*B12*B27
    X_Mn_X_C_Om_MnC = X_Mn*X_C*Om_MnC
    # 𝛥G Chemical: =SUM(B40,B41,B42,B43,B44,B45,B46,B47)
    dG_Chemical = (X_Fe_dG_Fe + X_Mn_dG_Mn + X_Al_dG_Al + X_C_dG_C + X_Fe_X_Mn_Om_FeMn + X_Fe_X_Al_Om_FeAl + X_Fe_X_C_Om_FeC + X_Mn_X_C_Om_MnC)
    # 𝛥G Magnetic: =((B80*B57*B39*LN(B18+1))-(B80*B57*B38*LN(B19+1)))
    dG_Magnetic = ((R*T*f_HCP*math.log(B_HCP+1))-(R*T*f_FCC*math.log(B_FCC+1)))
    # 𝑎◦(FCC): =3.578+(0.033*B12)+(0.00095*B4)-(0.0002*B7)+(0.0006*B6)+(0.0056*B5)+(0.0031*B11)+(0.0018*B9)
    a0_FCC = 3.578+(0.033*X_C)+(0.00095*X_Mn)-(0.0002*X_Ni)+(0.0006*X_Cr)+(0.0056*X_Al)+(0.0031*X_Mo)+(0.0018*X_V)
    # 𝑎◦(T): =B52*(1+2.065*10^-9*(B57-300))*10^-10
    a0_T = a0_FCC*(1+2.065*10**-9*(T-300))*10**-10
    # a2: =B53^2
    a2 = a0_T**2
    # a2*N: =B54*B55
    a2_N = a2*N
    # 𝜌: =(2.3094/B56)
    rho = (2.3094/a2_N)
    # 𝛄: =2*B1*SUM(B48:B50)+(2*B2)
    gamma = 2*rho*(dG_Chemical + dG_Magnetic + dG_ex)+(2*sigma)
    # SFE: =$B$51*1000
    SFE = gamma*1000
    return SFE

def main():
    print("Enter wt.% for Mo, Cu, V, Mn, Fe, Ni, Al, Cr, Si, C:")
    w_Mo = float(input("Mo: "))
    w_Cu = float(input("Cu: "))
    w_V = float(input("V: "))
    w_Mn = float(input("Mn: "))
    w_Fe = float(input("Fe: "))
    w_Ni = float(input("Ni: "))
    w_Al = float(input("Al: "))
    w_Cr = float(input("Cr: "))
    w_Si = float(input("Si: "))
    w_C = float(input("C: "))

    gamma_mJ = compute_gamma_akbari(
        w_Mo,
        w_Cu,
        w_V,
        w_Mn,
        w_Fe,
        w_Ni,
        w_Al,
        w_Cr,
        w_Si,
        w_C
    )
    print(f"\nComputed γ = {gamma_mJ:.7f} mJ/m²\n")

if __name__ == "__main__":
    main()
