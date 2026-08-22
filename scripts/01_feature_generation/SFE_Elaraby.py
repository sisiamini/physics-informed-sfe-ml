import math

def compute_gamma_elaraby(w_C, w_Mn, w_Si, w_Al, w_Mo, w_N, w_Cr, w_Ni, w_Cu, w_Fe):
    if w_C == 0.0:
        w_C = 0.0001
    if w_Mn == 0.0:
        w_Mn = 0.0001
    if w_Si == 0.0:
        w_Si = 0.0001
    if w_Al == 0.0:
        w_Al = 0.0001
    if w_Mo == 0.0:
        w_Mo = 0.0001
    if w_N == 0.0:
        w_N = 0.0001
    if w_Cr == 0.0:
        w_Cr = 0.0001
    if w_Ni == 0.0:
        w_Ni = 0.0001
    if w_Cu == 0.0:
        w_Cu = 0.0001
    if w_Fe == 0.0:
        w_Fe = 0.0001

    sigma = 10.0
    T = 300.0
    grain_size = 10.0
    a = 3.6e-10
    # N A: =6.0221413E+23
    N_A = 6.0221413E+23
    # R: =8.3144621
    R = 8.3144621
    # p: =0.28
    p = 0.28
    # D: =2.342456517
    D = 2.342456517
    # ΔG Fe: =-2243.38+4.309*$B$3
    dG_Fe = -2243.38+4.309*T
    # ΔG C: =-24595.12
    dG_C = -24595.12
    # ΔG Mn: =-1000+1.123*$B$3
    dG_Mn = -1000+1.123*T
    # ΔG Al: =5481.04-1.799*$B$3
    dG_Al = 5481.04-1.799*T
    # ΔG Si: =-560-8*$B$3
    dG_Si = -560-8*T
    # ΔG Ni: =1046+1.2552*$B$3
    dG_Ni = 1046+1.2552*T
    # ΔG Cr: =1370-0.163*$B$3
    dG_Cr = 1370-0.163*T
    # ΔG Mo: =-3650+0.63*$B$3
    dG_Mo = -3650+0.63*T
    # ΔG Cu: =600+0.2*$B$3
    dG_Cu = 600+0.2*T
    # Ω FeC: =42500
    Om_FeC = 42500
    # Ω FeAl: =3328
    Om_FeAl = 3328
    # Ω FeNi: =2095
    Om_FeNi = 2095
    # Ω FeCr: =2095
    Om_FeCr = 2095
    # Ω CrNi: =4190
    Om_CrNi = 4190
    # Ω MnC: =26910
    Om_MnC = 26910
    # Ω FeMo: =0
    Om_FeMo = 0
    # N parameter: G CrN^FCC: =18800
    N_parameter_G_CrNFCC = 18800
    # N parameter: G NiN^FCC: =-17000
    N_parameter_G_NiNFCC = -17000
    # N parameter: G CrNi^FCC: =-35800
    N_parameter_G_CrNiFCC = -35800
    # N parameter: G MnN^FCC: =8800
    N_parameter_G_MnNFCC = 8800
    # N parameter: G CrMn^FCC: =-10000
    N_parameter_G_CrMnFCC = -10000
    # N parameter: G NiMn^FCC: =25800
    N_parameter_G_NiMnFCC = 25800
    # U CrN: =$B$57-($B$22-$B$28)
    U_CrN = N_parameter_G_CrNFCC-(dG_Fe-dG_Cr)
    # U NiN: =$B$58-($B$22-$B$27)
    U_NiN = N_parameter_G_NiNFCC-(dG_Fe-dG_Ni)
    # U CrNi: =$B$59-($B$28-$B$27)
    U_CrNi = N_parameter_G_CrNiFCC-(dG_Cr-dG_Ni)
    # U MnN: =$B$60-($B$22-$B$24)
    U_MnN = N_parameter_G_MnNFCC-(dG_Fe-dG_Mn)
    # U CrMn: =$B$61-($B$28-$B$24)
    U_CrMn = N_parameter_G_CrMnFCC-(dG_Cr-dG_Mn)
    # U NiMn: =$B$62-($B$27-$B$24)
    U_NiMn = N_parameter_G_NiMnFCC-(dG_Ni-dG_Mn)
    # M C: =12.011
    M_C = 12.011
    # M Mn: =54.93805
    M_Mn = 54.93805
    # M Si: =28.0855
    M_Si = 28.0855
    # M Al: =26.981539
    M_Al = 26.981539
    # M Mo: =95.95
    M_Mo = 95.95
    # M N: =14.00674
    M_N = 14.00674
    # M Cr: =51.9961
    M_Cr = 51.9961
    # M Ni: =58.6934
    M_Ni = 58.6934
    # M Cu: =63.546
    M_Cu = 63.546
    # M Fe: =55.847
    M_Fe = 55.847
    # n total in 100 g alloy: =E2/$B$80+F2/$B$81+G2/$B$82+H2/$B$83+I2/$B$84+J2/$B$85+K2/$B$86+L2/$B$87+M2/$B$88+N2/$B$89
    n_total_in_100_g_alloy = w_C/M_C+w_Mn/M_Mn+w_Si/M_Si+w_Al/M_Al+w_Mo/M_Mo+w_N/M_N+w_Cr/M_Cr+w_Ni/M_Ni+w_Cu/M_Cu+w_Fe/M_Fe
    # ΔG exgs: =IF($B$4=0,0,1000*170.06*EXP(-$B$4/18.55))
    dG_exgs = (0 if grain_size==0 else 1000*170.06*math.exp(-grain_size/18.55))
    # γ Nb term: =0
    gamma_Nb_term = 0
    # ρ: =4/(SQRT(3)*B5^2*B6)
    rho = 4/(math.sqrt(3)*a**2*N_A)
    # X C: =(E2/$B$80)/$B$90
    X_C = (w_C/M_C)/n_total_in_100_g_alloy
    # X Mn: =(F2/$B$81)/$B$90
    X_Mn = (w_Mn/M_Mn)/n_total_in_100_g_alloy
    # X Si: =(G2/$B$82)/$B$90
    X_Si = (w_Si/M_Si)/n_total_in_100_g_alloy
    # X Al: =(H2/$B$83)/$B$90
    X_Al = (w_Al/M_Al)/n_total_in_100_g_alloy
    # X Mo: =(I2/$B$84)/$B$90
    X_Mo = (w_Mo/M_Mo)/n_total_in_100_g_alloy
    # X N: =(J2/$B$85)/$B$90
    X_N = (w_N/M_N)/n_total_in_100_g_alloy
    # X Cr: =(K2/$B$86)/$B$90
    X_Cr = (w_Cr/M_Cr)/n_total_in_100_g_alloy
    # X Ni: =(L2/$B$87)/$B$90
    X_Ni = (w_Ni/M_Ni)/n_total_in_100_g_alloy
    # X Cu: =(M2/$B$88)/$B$90
    X_Cu = (w_Cu/M_Cu)/n_total_in_100_g_alloy
    # X Fe: =(N2/$B$89)/$B$90
    X_Fe = (w_Fe/M_Fe)/n_total_in_100_g_alloy
    # Ω FeMn: =2873-717*($B$20-$B$12)
    Om_FeMn = 2873-717*(X_Fe-X_Mn)
    # Ω FeSi: =2850+3520*($B$20-$B$13)
    Om_FeSi = 2850+3520*(X_Fe-X_Si)
    # ΔG chem: =$B$20*$B$22+$B$11*$B$23+$B$12*$B$24+$B$14*$B$25+$B$17*$B$28+$B$18*$B$27+$B$15*$B$29+$B$19*$B$30+$B$13*$B$26
    dG_chem = X_Fe*dG_Fe+X_C*dG_C+X_Mn*dG_Mn+X_Al*dG_Al+X_Cr*dG_Cr+X_Ni*dG_Ni+X_Mo*dG_Mo+X_Cu*dG_Cu+X_Si*dG_Si
    # ΔG interaction: =$B$20*$B$12*$B$33+$B$20*$B$14*$B$34+$B$20*$B$17*$B$37+$B$20*$B$18*$B$36+$B$20*$B$13*$B$35+$B$17*$B$18*$B$38+$B$12*$B$11*$B$39+$B$20*$B$11*$B$32
    dG_interaction = X_Fe*X_Mn*Om_FeMn+X_Fe*X_Al*Om_FeAl+X_Fe*X_Cr*Om_FeCr+X_Fe*X_Ni*Om_FeNi+X_Fe*X_Si*Om_FeSi+X_Cr*X_Ni*Om_CrNi+X_Mn*X_C*Om_MnC+X_Fe*X_C*Om_FeC
    # T N^γ: =10*$B$12^3+898.4*$B$12^2+1176*$B$12-1992*$B$11-1272*$B$13-661*$B$14-170*$B$17+152.4
    T_Ngamma = 10*X_Mn**3+898.4*X_Mn**2+1176*X_Mn-1992*X_C-1272*X_Si-661*X_Al-170*X_Cr+152.4
    # τ γ = T/T N^γ: =IF($B$45=0,0,$B$3/$B$45)
    tau_gamma_T_T_Ngamma = (0 if T_Ngamma==0 else T/T_Ngamma)
    # f(τ γ): =IF($B$46=0,0,IF($B$46<=1,1-(1/$B$9)*((79/(140*$B$8))*$B$46^-1+(474/497)*(1/$B$8-1)*($B$46^3/6+$B$46^9/135+$B$46^15/600)),-($B$46^-5/10+$B$46^-15/315+$B$46^-25/1500)/$B$9))
    f_tau_gamma = (0 if tau_gamma_T_T_Ngamma==0 else (1-(1/D)*((79/(140*p))*tau_gamma_T_T_Ngamma**-1+(474/497)*(1/p-1)*(tau_gamma_T_T_Ngamma**3/6+tau_gamma_T_T_Ngamma**9/135+tau_gamma_T_T_Ngamma**15/600)) if tau_gamma_T_T_Ngamma<=1 else -(tau_gamma_T_T_Ngamma**-5/10+tau_gamma_T_T_Ngamma**-15/315+tau_gamma_T_T_Ngamma**-25/1500)/D))
    # β γ: =0.7*$B$20+0.62*$B$12+0.62*$B$18-0.8*$B$17-0.64*$B$20*$B$12-4*$B$11
    beta_gamma = 0.7*X_Fe+0.62*X_Mn+0.62*X_Ni-0.8*X_Cr-0.64*X_Fe*X_Mn-4*X_C
    # ΔG mag^γ: =IF($B$46=0,0,$B$7*$B$3*LN(1+$B$48)*$B$47)
    dG_maggamma = (0 if tau_gamma_T_T_Ngamma==0 else R*T*math.log(1+beta_gamma)*f_tau_gamma)
    # T N^ε: =580*$B$12
    T_Nepsilon = 580*X_Mn
    # τ ε = T/T N^ε: =IF($B$50=0,0,$B$3/$B$50)
    tau_epsilon_T_T_Nepsilon = (0 if T_Nepsilon==0 else T/T_Nepsilon)
    # f(τ ε): =IF($B$51=0,0,IF($B$51<=1,1-(1/$B$9)*((79/(140*$B$8))*$B$51^-1+(474/497)*(1/$B$8-1)*($B$51^3/6+$B$51^9/135+$B$51^15/600)),-($B$51^-5/10+$B$51^-15/315+$B$51^-25/1500)/$B$9))
    f_tau_epsilon = (0 if tau_epsilon_T_T_Nepsilon==0 else (1-(1/D)*((79/(140*p))*tau_epsilon_T_T_Nepsilon**-1+(474/497)*(1/p-1)*(tau_epsilon_T_T_Nepsilon**3/6+tau_epsilon_T_T_Nepsilon**9/135+tau_epsilon_T_T_Nepsilon**15/600)) if tau_epsilon_T_T_Nepsilon<=1 else -(tau_epsilon_T_T_Nepsilon**-5/10+tau_epsilon_T_T_Nepsilon**-15/315+tau_epsilon_T_T_Nepsilon**-25/1500)/D))
    # β ε: =0.62*$B$12-4*$B$11
    beta_epsilon = 0.62*X_Mn-4*X_C
    # ΔG mag^ε: =IF($B$51=0,0,$B$7*$B$3*LN(1+$B$53)*$B$52)
    dG_magepsilon = (0 if tau_epsilon_T_T_Nepsilon==0 else R*T*math.log(1+beta_epsilon)*f_tau_epsilon)
    # ΔG mag^γ→ε: =$B$54-$B$49
    dG_maggammaepsilon = dG_magepsilon-dG_maggamma
    # D1 N: =$B$20+$B$12*EXP(-$B$66/($B$7*$B$3))+$B$17*EXP(-$B$63/($B$7*$B$3))+$B$18*EXP(-$B$64/($B$7*$B$3))
    D1_N = X_Fe+X_Mn*math.exp(-U_MnN/(R*T))+X_Cr*math.exp(-U_CrN/(R*T))+X_Ni*math.exp(-U_NiN/(R*T))
    # D2 N: =$B$20*EXP($B$63/($B$7*$B$3))+$B$12*EXP(-$B$67/($B$7*$B$3))+$B$17+$B$18*EXP(-$B$65/($B$7*$B$3))
    D2_N = X_Fe*math.exp(U_CrN/(R*T))+X_Mn*math.exp(-U_CrMn/(R*T))+X_Cr+X_Ni*math.exp(-U_CrNi/(R*T))
    # D3 N: =$B$20*EXP($B$64/($B$7*$B$3))+$B$12*EXP(-$B$68/($B$7*$B$3))+$B$17*EXP($B$65/($B$7*$B$3))+$B$18
    D3_N = X_Fe*math.exp(U_NiN/(R*T))+X_Mn*math.exp(-U_NiMn/(R*T))+X_Cr*math.exp(U_CrNi/(R*T))+X_Ni
    # G N contribution: 6X N(...): =6*$B$16*($B$24+($B$66*$B$20)/$B$69+($B$67*$B$17)/$B$70+($B$68*$B$18)/$B$71)
    G_N_contribution_6X_N = 6*X_N*(dG_Mn+(U_MnN*X_Fe)/D1_N+(U_CrMn*X_Cr)/D2_N+(U_NiMn*X_Ni)/D3_N)
    # ΔG N: =-20705*$B$16^2+23097*$B$16
    dG_N = -20705*X_N**2+23097*X_N
    # Y N: =IF($B$16>0,1/(1+((1-$B$16)/$B$16)*EXP(-$B$73/($B$7*$B$3))),0)
    Y_N = (1/(1+((1-X_N)/X_N)*math.exp(-dG_N/(R*T))) if X_N>0 else 0)
    # G N^ideal: =IF(AND($B$16>0,$B$74>0,$B$74<1),$B$7*$B$3*($B$16*LN($B$74/$B$16)+(1-$B$16)*LN((1-$B$74)/(1-$B$16))),0)
    G_Nideal = (R*T*(X_N*math.log(Y_N/X_N)+(1-X_N)*math.log((1-Y_N)/(1-X_N))) if all([X_N>0, Y_N>0, Y_N<1]) else 0)
    # G N^excess: =0.25*$B$73*($B$74-$B$16)^2
    G_Nexcess = 0.25*dG_N*(Y_N-X_N)**2
    # ΔG total: =1000*($B$42+$B$43+$B$55)
    dG_total = 1000*(dG_chem+dG_interaction+dG_maggammaepsilon)
    # γ basic: =2*$B$1*$B$92+2*$B$2+2*$B$1*$B$93
    gamma_basic = 2*rho*dG_total+2*sigma+2*rho*dG_exgs
    # γ N term: =2000*$B$1*($B$72+$B$75+$B$76)
    gamma_N_term = 2000*rho*(G_N_contribution_6X_N+G_Nideal+G_Nexcess)
    # SFE Γ: =$B$94+$B$95+$B$96
    SFE_Gamma = gamma_basic+gamma_Nb_term+gamma_N_term
    # SFE: =$B$97
    SFE = SFE_Gamma
    return SFE

def main():
    print("Enter wt.% for C, Mn, Si, Al, Mo, N, Cr, Ni, Cu, Fe:")
    w_C = float(input("C: "))
    w_Mn = float(input("Mn: "))
    w_Si = float(input("Si: "))
    w_Al = float(input("Al: "))
    w_Mo = float(input("Mo: "))
    w_N = float(input("N: "))
    w_Cr = float(input("Cr: "))
    w_Ni = float(input("Ni: "))
    w_Cu = float(input("Cu: "))
    w_Fe = float(input("Fe: "))

    gamma_mJ = compute_gamma_elaraby(
        w_C,
        w_Mn,
        w_Si,
        w_Al,
        w_Mo,
        w_N,
        w_Cr,
        w_Ni,
        w_Cu,
        w_Fe
    )
    print(f"\nComputed γ = {gamma_mJ:.7f} mJ/m²\n")

if __name__ == "__main__":
    main()
