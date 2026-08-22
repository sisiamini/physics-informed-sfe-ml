import math

def compute_gamma_dumay(w_Mn, w_Fe, w_C, w_Cu, w_Cr, w_Al, w_Si):
    if w_Mn == 0.0:
        w_Mn = 0.0001
    if w_Fe == 0.0:
        w_Fe = 0.0001
    if w_C == 0.0:
        w_C = 0.0001
    if w_Cu == 0.0:
        w_Cu = 0.0001
    if w_Cr == 0.0:
        w_Cr = 0.0001
    if w_Al == 0.0:
        w_Al = 0.0001
    if w_Si == 0.0:
        w_Si = 0.0001

    rho = 2.94e-05
    sigmagamma_epsilon = 0.008
    T = 300.0
    M_Mn = 54.94
    M_Fe = 55.85
    M_C = 12.01
    M_Cu = 63.546
    M_Cr = 52.0
    M_Al = 26.98
    M_Si = 28.09
    R = 8.314
    a = 1246.0
    b = 24.29
    c = -17.175
    C_FeMn = 2873.0
    D_FeMn = -717.0
    E_FeSi = 2850.0
    F_FeSi = 3520.0
    pgamma = 0.28
    pepsilon = 0.28
    # ΔG Fe: =-2243.38+4.309*$B$48
    dG_Fe = -2243.38+4.309*T
    # ΔG Mn: =-1000+1.123*$B$48
    dG_Mn = -1000+1.123*T
    # ΔG Cu: =2273
    dG_Cu = 2273
    # ΔG Cr: =1370-10*$B$48
    dG_Cr = 1370-10*T
    # ΔG Al: =2800+5*$B$48
    dG_Al = 2800+5*T
    # ΔG Si: =-560-8*$B$48
    dG_Si = -560-8*T
    # D γ: =(518/1125)+(11692/15975)*(1/B73-1)
    D_gamma = (518/1125)+(11692/15975)*(1/pgamma-1)
    # D ε: =(518/1125)+(11692/15975)*(1/$B$74-1)
    D_epsilon = (518/1125)+(11692/15975)*(1/pepsilon-1)
    # n Mn: =E2/B49
    n_Mn = w_Mn/M_Mn
    # n Fe: =F2/B50
    n_Fe = w_Fe/M_Fe
    # n C: =G2/B51
    n_C = w_C/M_C
    # n Cu: =H2/B52
    n_Cu = w_Cu/M_Cu
    # n Cr: =I2/B53
    n_Cr = w_Cr/M_Cr
    # n Al: =J2/B54
    n_Al = w_Al/M_Al
    # n Si: =K2/B55
    n_Si = w_Si/M_Si
    # n total: =SUM(B56:B62)
    n_total = (n_Mn + n_Fe + n_C + n_Cu + n_Cr + n_Al + n_Si)
    # X Fe: =B57/B63
    X_Fe = n_Fe/n_total
    # X Mn: =B56/B63
    X_Mn = n_Mn/n_total
    # X C: =B58/B63
    X_C = n_C/n_total
    # X Cu: =B59/B63
    X_Cu = n_Cu/n_total
    # X Cr: =B60/B63
    X_Cr = n_Cr/n_total
    # X Al: =B61/B63
    X_Al = n_Al/n_total
    # X Si: =B62/B63
    X_Si = n_Si/n_total
    # β ε/μB: =0.62*B4-4*B5
    beta_epsilon_muB = 0.62*X_Mn-4*X_C
    # β γ/μB: =0.7*B3+0.62*B4-0.64*B3*B4-4*B5
    beta_gamma_muB = 0.7*X_Fe+0.62*X_Mn-0.64*X_Fe*X_Mn-4*X_C
    # Ω FeMn: =$B$69+$B$70*(B3-B4)
    Om_FeMn = C_FeMn+D_FeMn*(X_Fe-X_Mn)
    # Ω FeSi: =B71+B72*(B3-B9)
    Om_FeSi = E_FeSi+F_FeSi*(X_Fe-X_Si)
    # ΔG FeMnX/C: =IF(B5=0,$B$66*$B$67+$B$68*B4,($B$66/B5)*(1-EXP(-$B$67*B5))+$B$68*B4)
    dG_FeMnX_C = (a*b+c*X_Mn if X_C==0 else (a/X_C)*(1-math.exp(-b*X_C))+c*X_Mn)
    # Tneel ε: =580*B4
    Tneel_epsilon = 580*X_Mn
    # Tneel γ: =250*LN(B4)-4750*B5*B4-222*B6-2.6*B7-6.2*B8-13*B9+720
    Tneel_gamma = 250*math.log(X_Mn)-4750*X_C*X_Mn-222*X_Cu-2.6*X_Cr-6.2*X_Al-13*X_Si+720
    # τ γ: =$B$48/B22
    tau_gamma = T/Tneel_gamma
    # τ ε: =$B$48/B21
    tau_epsilon = T/Tneel_epsilon
    # f γ (τ<1): =1-(1/B23)*((79*B25)/(140*$B$73)+(474/497)*(1/$B$73-1)*(B25^3/6+B25^9/135+B25^15/600))
    f_gamma_tault1 = 1-(1/D_gamma)*((79*tau_gamma)/(140*pgamma)+(474/497)*(1/pgamma-1)*(tau_gamma**3/6+tau_gamma**9/135+tau_gamma**15/600))
    # f γ (τ>1): =-(1/B23)*(B25^-5/10+B25^-15/315+B25^-25/1500)
    f_gamma_taugt1 = -(1/D_gamma)*(tau_gamma**-5/10+tau_gamma**-15/315+tau_gamma**-25/1500)
    # f ε (τ<1): =1-(1/B24)*((79*B26)/(140*$B$74)+(474/497)*(1/$B$74-1)*(B26^3/6+B26^9/135+B26^15/600))
    f_epsilon_tault1 = 1-(1/D_epsilon)*((79*tau_epsilon)/(140*pepsilon)+(474/497)*(1/pepsilon-1)*(tau_epsilon**3/6+tau_epsilon**9/135+tau_epsilon**15/600))
    # f ε (τ>1): =-(1/B24)*(B26^-5/10+B26^-15/315+B26^-25/1500)
    f_epsilon_taugt1 = -(1/D_epsilon)*(tau_epsilon**-5/10+tau_epsilon**-15/315+tau_epsilon**-25/1500)
    # f γ: =IF(B25<1,B27,B28)
    f_gamma = (f_gamma_tault1 if tau_gamma<1 else f_gamma_taugt1)
    # f ε: =IF(B26<1,B29,B30)
    f_epsilon = (f_epsilon_tault1 if tau_epsilon<1 else f_epsilon_taugt1)
    # X Fe*ΔG Fe: =B3*B12
    X_Fe_dG_Fe = X_Fe*dG_Fe
    # X Mn*ΔG Mn: =B4*B13
    X_Mn_dG_Mn = X_Mn*dG_Mn
    # X Cu*ΔG Cu: =B6*B14
    X_Cu_dG_Cu = X_Cu*dG_Cu
    # X Cr*ΔG Cr: =B7*B15
    X_Cr_dG_Cr = X_Cr*dG_Cr
    # X Al*ΔG Al: =B8*B16
    X_Al_dG_Al = X_Al*dG_Al
    # X Si*ΔG Si: =B9*B17
    X_Si_dG_Si = X_Si*dG_Si
    # X C*ΔG FeMnX/C: =B5*B20
    X_C_dG_FeMnX_C = X_C*dG_FeMnX_C
    # X Fe*X Mn*Ω FeMn: =B3*B4*B18
    X_Fe_X_Mn_Om_FeMn = X_Fe*X_Mn*Om_FeMn
    # X Fe*X Si*Ω FeSi: =B3*B9*B19
    X_Fe_X_Si_Om_FeSi = X_Fe*X_Si*Om_FeSi
    # ΔG Chemical: =SUM(B33:B41)
    dG_Chemical = (X_Fe_dG_Fe + X_Mn_dG_Mn + X_Cu_dG_Cu + X_Cr_dG_Cr + X_Al_dG_Al + X_Si_dG_Si + X_C_dG_FeMnX_C + X_Fe_X_Mn_Om_FeMn + X_Fe_X_Si_Om_FeSi)
    # ΔG Magnetic: =$B$64*$B$48*B32*LN(1+B10)-$B$64*$B$48*B31*LN(1+B11)
    dG_Magnetic = R*T*f_epsilon*math.log(1+beta_epsilon_muB)-R*T*f_gamma*math.log(1+beta_gamma_muB)
    # ΔG total: =B42+B43
    dG_total = dG_Chemical+dG_Magnetic
    # γ: =2*B1*B44+2*B2
    gamma = 2*rho*dG_total+2*sigmagamma_epsilon
    # SFE γ: =B45*1000
    SFE_gamma = gamma*1000
    # Mechanism indicator: =IF(B46<12,"<12 mJ/m²: α-martensite risk",IF(B46<18,"12–18 mJ/m²: ε-martensite range",">18 mJ/m²: twinning/glide likely"))
    Mechanism_indicator = ("<12 mJ/m²: α-martensite risk" if SFE_gamma<12 else ("12–18 mJ/m²: ε-martensite range" if SFE_gamma<18 else ">18 mJ/m²: twinning/glide likely"))
    # SFE: =$B$46
    SFE = SFE_gamma
    return SFE

def main():
    print("Enter wt.% for Mn, Fe, C, Cu, Cr, Al, Si:")
    w_Mn = float(input("Mn: "))
    w_Fe = float(input("Fe: "))
    w_C = float(input("C: "))
    w_Cu = float(input("Cu: "))
    w_Cr = float(input("Cr: "))
    w_Al = float(input("Al: "))
    w_Si = float(input("Si: "))

    gamma_mJ = compute_gamma_dumay(
        w_Mn,
        w_Fe,
        w_C,
        w_Cu,
        w_Cr,
        w_Al,
        w_Si
    )
    print(f"\nComputed γ = {gamma_mJ:.7f} mJ/m²\n")

if __name__ == "__main__":
    main()
