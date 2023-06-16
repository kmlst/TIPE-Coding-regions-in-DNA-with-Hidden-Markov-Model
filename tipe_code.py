import numpy as np
import matplotlib.pyplot as plt



# seulement deux etats

def epidemie_1(temps = 50, population = 10**6):
    propagation = np.array( [[0.9 , 0.1], [0.3, 0.7]]) # 0 -> infecte, 1 -> sain
    popu = np.array([0, 1])
    X_temps = np.linspace(0, temps, temps)
    Y_infectes = []
    for t in range(temps):
        Y_infectes.append(popu[0]*population)
        popu = np.dot(popu, propagation)
    plt.plot(X_temps, Y_infectes)


# illustration de markov deux etats
# modele irrealiste a cause de la pente, voir site worldometers.info juste pour donner une impression de ce que ca devrait donner pas pour donner 


#epidemie_1()
#plt.show()



# 5 etats cette fois:
def epidemie_2(temps = 100, population = 10**6):
    propagation = np.array( [
        [0.7, 0.2, 0, 0, 0.0001, 0.0999], # 0 -> infecte vaccine
        [0.2, 0.8, 0, 0, 0, 0], # 1 -> sain vaccine
        [0 , 0.2, 0.1, 0.7, 0, 0], # 2 -> sain non vaccin
        [0,0.2, 0, 0.7, 0.001, 0.099], # 3 -> infecte non vaccine
        [0, 0, 0, 0, 1, 0 ], # 4 -> mort
        [0, 0, 0, 0, 0, 1 ] # 5 -> immunise
    ])
    popu = np.array([0, 0, 1, 0, 0, 0])
    X_temps = np.linspace(0, temps, temps)
    Y_infectes = []
    for t in range(temps):
        Y_infectes.append(popu[0]*population)
        popu = np.dot(popu, propagation)
    plt.plot(X_temps, Y_infectes)

#epidemie_2()
#plt.show()

# modele bcp plus complique car bcp d'etats mais
# resultats bien plus satisfaisant
# on peut jouer sur la propagation de l'epidemie 
# qui prennent en compte la reaction des gens, des gouvernements, le confinement ( notamment probabilite d'infection qui descend, celle de vaccination monte, mais aussi mutation aleatoire)




# Chaine de Markov cachee
# changement de domaine 

def max_arg(liste):
    """ renvoie le max de la liste et le plus petit indice ou il a ete realise"""
    m = liste[0]
    i_max = 0
    for n in range(len(liste)):
        if liste[n] > m:
            m = liste[n]
            i_max = n
    return m, i_max


def Viterbi(A, B, Obs):
    # A matrice de transition
    # B les probabilites d'observation tq b_j(o_t) = B[j][t]
    # On travaille avec des logarithmes
    logA = np.log(A)
    logB = np.log(B)
    N = len(A)
    T = len(Obs) 
    pointeurs =  np.reshape(np.zeros(T*N), (N,T)) # sert a retracer le chemin a la fin
    alpha_prec =  np.array(B[:][Obs[0]])
    alpha_suiv = np.zeros(N) 
    for t in range(T):
        nouv_alpha = np.zeros(N)
        for j in range(N):
            nouv_alpha[j], pointeurs[j][t] = max_arg( np.array( np.log(alpha_suiv[i]) + logA[i][j] + logB[j][Obs[t]] for i in range(N))) 
            # log est croissante, conserve donc le max
            # on met en pointeur l'etat i qui realise le maximum : c'etait l'etat precedent
        alpha_prec = alpha_suiv[:]
        alpha_suiv = nouv_alpha[:]
    
    pmax, i_final = max_arg(alpha_suiv)
    pmax = np.exp(pmax) 
    etats_successifs = np.zeros(T)

    i = i_final
    for t in range(1, T+1, -1):
        etats_successifs[t] = i
        i = pointeurs[i][t-1]
    return pmax, etats_successifs



def forward(A, B, Obs):
    N = len(A)
    T = len(B)
    alpha_prec = np.array(B[:][Obs[0]])
    alpha_suiv = np.zeros(N)
    for t in range(T):
        nouv_alpha = np.zeros(N)
        for j in range(N):
            nouv_alpha[j] = B[j][Obs[t]] * sum( alpha_suiv[i] * A[i][j] for i in range(N)) 
        alpha_prec = alpha_suiv[:]
        alpha_suiv = nouv_alpha[:]
    return sum(alpha_suiv)




# def baum-welch(A,B, Obs):
#     if # condition de convergence

#     else:
#         N = len(A)
#         T = len(Obs) 
#         alphas = np.reshape(np.zeros(N*T), (N, T)) 
#         betas = np.reshape(np.zeros(N*T), (N, T))
#         # a initialiser correctement avec un for ICI
#         C = sum(alphas)
#         D = sum(beta)
#         alphas = alphas/C
#         beta = beta/D
#         for t in range(1, T): # on construit 
#             for i in range(N):
#                 alphas[i][t] = B[i][Obs[t]] *
#                 betas[i][t] = 

# ABANDON MOMOENTANE



def baum_welch_naif(A, B, Obs):
    N = len(A)
    T = len(Obs)
    alphas = np.reshape(np.zeros(N*T), (T, N)) 
    betas = np.reshape(np.zeros(N*T), (T, N))
    # trouver toutes les valeurs des alphas et betas
    alphas[:][0] = B[:][Obs[0]]
    betas[T-1][:] = np.ones(N)

    for t in range(1, T-2):
        for j in range(N):
            alphas[t][j] =  B[j][Obs[t]] * sum( alphas[t-1][i] * A[i][j]  for i in range(N)) 
            betas[T-1-t][j] = B[Obs[T-t]][j] * sum( betas[T-t][i] * A[j][i] for i in range(N))

    Pobs = sum(alphas[T-1][:])
    # step Expectations
    zeta = np.reshape(np.zeros(N*N*T), (T,N, N))
    gamma = np.reshape(np.zeros(N*T), (T,N))

    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                zeta[t][i][j] = alphas[t][i] * betas[t+1][j] * A[i][j] * B[Obs[t]][j] / Pobs
    for t in range(T):
        for i in range(N):
            gamma[t][i] = (alphas[t][j] * betas[t][j]) / Pobs

    #step S

    nouvA = np.reshape(np.zeros(N**2), (N,N))
    nouvB = np.reshape(np.zeros(N * len(B[0])), (N, len(B[0])))

    for i in range(N):
        denom = sum( sum( zeta[t][i][k] for k in range(N)) for t in range(T)) 
        for j in range(N):
            nouvA[i][j] = sum( zeta[t][i][j] for t in range(T)) / denom

    for j in range(N):
        for k in range(len(B)):
            denom = sum(gamma[t][j] for t in range(T))
            for t in range(T):
                if Obs[t] == k:
                    nouvB[j][k] = nouvB[j][k] + gamma[t][j] / denom
    
    return nouvA, nouvB





def traite_fichier_adn():
    nucleotide = open("adn_pur.txt", "r")
    nombres = open("adn_traite", "a")
    lignes = nucleotide.readlines()
    N = ['a', 'c', 't', 'g']

    for l in lignes:
        for carac in l:
            if carac == 'a':
                nombres.write("0 ")
            if carac == 'c':
                nombres.write("1 ")
            if carac == 't':
                nombres.write("2 ")
            if carac == 'g':
                nombres.write("3 ")
    nucleotide.close()
    nombres.close()


adn = open("adn_traite", "r")
sequence = adn.readlines()
Ob = []
for ligne in sequence:
    for nclt in ligne:
        if nclt in ['0', '1', '2', '3']:
            Ob.append(int(nclt))
adn.close()




def sequencageADN(Obs):
    precision = 0.1
    A =  0.25 * np.reshape(np.ones(16), (4, 4))
    B =  0.25 * np.reshape(np.ones(16), (4, 4))

    Ap, Bp = baum_welch_naif(A, B, Obs)
    while np.linalg.norm(A - Ap) < precision or np.linalg.norm(B - Bp)<precision:
        A = Ap
        B = Bp
        Ap, Bp = baum_welch_naif(A, B, Obs)
    
    return A, B


#print(sequencageADN(Ob))














def ruine_du_joueur(N, p, T = 100):
    X_t = np.zeros(2*N+1)
    X_t[N] = 1.0
    T = list(range(1, T))
    A = np.reshape(np.zeros((2*N+1)**2), ((2*N+1),(2*N+1)))
    A[0][0] = 1
    A[-1][-1] = 1

    for i in range(1, 2*N):
        A[i][i-1] = 1-p
        A[i][i+1] = p

    print(A)
    Argent = []
    for t in T:
        m = max(X_t)
        for k in range(2*N+1):
            if X_t[k] == m:
                Argent.append(k)
                break
        X_t = np.dot(X_t, A)
    plt.plot(T, Argent)



















import random as rd
def vol_du_joueur(N, p):
    for _ in range(3):
        X = [0]
        Y = [N]
        temps = 0
        A = N
        while A > 0 and A < 2*N:
            temps += 1
            if rd.random()< p:
                A += 1
            else:
                A -= 1
            X.append(temps)
            Y.append(A)
        plt.plot(X, Y)
    
    plt.xlabel("temps")
    plt.ylabel("Pieces")
    plt.show()


#vol_du_joueur(20, 0.5)



def temps_de_vol(N,p):
    Y = []
    nb_essais = 100000
    for k in range(nb_essais):
        temps = 0
        A = N
        while A > 0 and A < 2*N:
            temps += 1
            if rd.random()< p:
                A += 1
            else:
                A -= 1
        Y.append(temps)
    Yp = [0]*(max(Y)+1)
    for y in Y:
        Yp[y] += 1
    plt.bar(list(range(max(Y)+1)), Yp, width=1.0, edgecolor = "#981FFA")
    plt.show()

#temps_de_vol(200, 0.7)

import cmath


def mouvement_brownien(N):
    position = 0 + 0j
    X = [position]
    t = 1
    i = 1j
    direction = 1
    while t < N:
        dir = rd.random()
        dist = rd.random()

        if dir < 0.05 or  0.50> dir > 0.45:
            if dist <0.01:
                direction *= cmath.exp(dir*2*np.pi*1j)
                position = position + dist * direction 

                X.append(position)
                t += 1
    plt.plot( [ z.real for z in X], [z.imag for z in X])
    plt.show()


mouvement_brownien(10000)




def baum_welch(A, B, Obs):
    N = len(A)
    T = len(Obs)
    alphas = np.reshape(np.zeros(N*T), (T, N)) 
    betas = np.reshape(np.zeros(N*T), (T, N))
    # trouver toutes les valeurs des alphas et betas
    alphas[:][0] = B[:][Obs[0]]
    betas[T-1][:] = np.ones(N)
    for t in range(1, T-2):
        for j in range(N):
            alphas[t][j] =  B[j][Obs[t]] * sum( alphas[t-1][i] * A[i][j]  for i in range(N)) 
            betas[T-1-t][j] = B[Obs[T-t]][j] * sum( betas[T-t][i] * A[j][i] for i in range(N))
    constantesC = []
    for t in range(T):
        C = sum(alphas[t][:])**(-1)
        constantes.append(C)
        alphas[t][:] = alphas[t][:] / C
    constancesc = []
    for t in range(T):
        c = 0
        for y in range(N):
            c += np.dot(alphas[t][:], A[y]) * B[t][y]
        c = 1 / c
        constantesc.append(c)
    Pobs = sum(alphas[T-1][:])
    # step Expectations
    zeta = np.reshape(np.zeros(N*N*T), (T,N, N))
    gamma = np.reshape(np.zeros(N*T), (T,N))

    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                zeta[t][i][j] = alphas[t][i] * betas[t+1][j] * A[i][j] * B[Obs[t]][j] / Pobs
    for t in range(T):
        for i in range(N):
            gamma[t][i] = (alphas[t][j] * betas[t][j]) / Pobs
    #step S
    nouvA = np.reshape(np.zeros(N**2), (N,N))
    nouvB = np.reshape(np.zeros(N * len(B[0])), (N, len(B[0])))
    for i in range(N):
        denom = sum( sum( zeta[t][i][k] for k in range(N)) for t in range(T)) 
        for j in range(N):
            nouvA[i][j] = sum( zeta[t][i][j] for t in range(T)) / denom
    for j in range(N):
        for k in range(len(B)):
            denom = sum(gamma[t][j] for t in range(T))
            for t in range(T):
                if Obs[t] == k:
                    nouvB[j][k] = nouvB[j][k] + gamma[t][j] / denom
    
    return nouvA, nouvB


